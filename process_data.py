"""Filter on RouterBench (0-shot) data, add response lengths, energy cost, and carbon cost"""
import ast
import tomllib
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ROOT_DIR / "config.toml"
RAW_PATH = ROOT_DIR / "data" / "raw" / "routerbench_0shot.pkl"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

MODELS = [
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "meta/llama-2-70b-chat",
]

# Cover a variety of tasks ranging from
# knowledge across languages, commonsense, math, and coding problems
TASKS = ["mmlu", "hellaswag", "arc", "gsm8k", "mbpp"]


def load_config_data(path=CONFIG_PATH):
    """Open config file and return the variables used for processing."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    return {
        "seed": cfg["seed"],
        "n_prompts": cfg["n_prompts"],
        "grid_intensity": cfg["grid_intensity"],
        "carbon_price": cfg["carbon_price"],
        "energy": cfg["energy_wh_per_1k_queries"],
        "split": cfg["split"],
        "lambdas": cfg["lambdas"],
    }


def filter_data(routerbench, n_prompts, seed):
    """Keep the selected models' columns, add a task column, and sample n_prompts prompts."""
    # columns to filter on
    keep_cols = ["sample_id", "prompt", "eval_name"]
    for m in MODELS:
        keep_cols += [m, f"{m}|model_response", f"{m}|total_cost"]  # quality score, response, cost
    routerbench = routerbench[keep_cols].copy()

    # Create a task column that only specifies main parent group of eval_name
    # to help later with stratified sampling
    routerbench["task"] = routerbench["eval_name"].str.split("-").str[0]
    # Rename grade to gsm8k to be consistent with the paper
    routerbench["task"] = routerbench["task"].replace({"grade": "gsm8k"})

    # Select all mbpp tasks since there's the fewest (427)(and less than n_prompts/5)
    mbpp = routerbench[routerbench["task"] == "mbpp"]
    n_per_task = (n_prompts - len(mbpp)) // (len(TASKS) - 1)  # e.g. (5000 - 427) // 4 = 1143

    # Randomly select n_per_task prompts from each remaining task
    return pd.concat([
        routerbench[routerbench["task"].isin(TASKS[:-1])]
        .groupby("task")
        .sample(n=n_per_task, random_state=seed),
        mbpp,
    ])


def add_features(rb_sampled, energy, grid_intensity, carbon_price):
    """Reshape to prompt-model pairs and add response length, energy, and carbon cost columns."""
    # Reshape so that each row is a prompt, model pair which simplifies calculations
    prompt_model_pair = pd.concat([
        rb_sampled[["sample_id", "prompt", "eval_name", "task"]].assign(
            model=m,
            quality=rb_sampled[m],
            response=rb_sampled[f"{m}|model_response"],
            cost=rb_sampled[f"{m}|total_cost"],
        )
        for m in MODELS
    ], ignore_index=True)

    # Extract response text and calculate token length
    text = prompt_model_pair["response"].apply(ast.literal_eval).str[0]
    prompt_model_pair["response_length"] = text.str.split().str.len()

    # Calculate avg response length
    prompt_model_pair["avg_response_length"] = (
        prompt_model_pair.groupby("model")["response_length"].transform("mean")
    )

    # Add energy column and divide by 1000 to get wh per query
    prompt_model_pair["wh_per_query"] = prompt_model_pair["model"].map(energy) / 1000

    # Scale average energy by response's relative length
    prompt_model_pair["wh"] = (
        prompt_model_pair["wh_per_query"]
        * prompt_model_pair["response_length"]
        / prompt_model_pair["avg_response_length"]
    )
    # Add carbon cost
    prompt_model_pair["carbon_cost"] = (
        prompt_model_pair["wh"] / 1000      # Wh -> kWh
        * grid_intensity                    # -> kg CO2
        * carbon_price / 1000               # $/t -> $
    )

    return prompt_model_pair


def make_prompts_table(rb_sampled, split_fractions, seed):
    """One row per prompt w/ features the router can see, plus a train/val/test label."""
    rb_sampled = rb_sampled.reset_index(drop=True)
    assert rb_sampled["sample_id"].is_unique

    # The prompt is stored as a string like "['text...']", so pull the text out
    text = rb_sampled["prompt"].apply(ast.literal_eval).str[0]

    prompts = rb_sampled[["sample_id", "task"]].copy()
    prompts["prompt_len_words"] = text.str.split().str.len()
    prompts["has_numbers"] = text.str.contains(r"\d").astype(int)
    prompts["n_questions"] = text.str.count(r"\?")
    prompts["has_code"] = text.str.contains(r"def |import |assert |```").astype(int)

    # Split each task separately so every partition has the same task mix.
    # Each prompt gets one label, so its four models always stay together.
    train_frac, val_frac, _ = split_fractions
    prompts["split"] = "test"
    for task, rows in prompts.groupby("task"):
        shuffled = rows.sample(frac=1, random_state=seed).index
        n_train = round(len(rows) * train_frac)
        n_val = round(len(rows) * val_frac)
        prompts.loc[shuffled[:n_train], "split"] = "train"
        prompts.loc[shuffled[n_train:n_train + n_val], "split"] = "val"

    return prompts


def make_rewards(pairs, prompts, lam):
    """One row per prompt, one column per model: quality - lam_dollar * dollars - lam_carbon * carbon."""
    reward = (
        pairs["quality"].astype(float)
        - lam["dollar"] * pairs["cost"]
        - lam["carbon"] * pairs["carbon_cost"]
    )

    table = pairs.assign(reward=reward).pivot(
        index="sample_id", columns="model", values="reward"
    )
    table = table[MODELS]                                  # fixed column order
    table.columns = [m.split("/")[-1] for m in MODELS]     # drop the slashes for Julia

    # Same row order as prompts, so row i means the same prompt in both files
    return table.loc[prompts["sample_id"]]

def main():
    cfg = load_config_data()
    routerbench = pd.read_pickle(RAW_PATH)
    rb_sampled = filter_data(routerbench, cfg["n_prompts"], cfg["seed"])
    prompt_model_pair = add_features(
        rb_sampled, cfg["energy"], cfg["grid_intensity"], cfg["carbon_price"]
    )
    print(prompt_model_pair.shape)
    # needed for later evaluation
    prompt_model_pair.to_csv(PROCESSED_DIR / "pairs.csv", index=False)

    # feature matrix
    prompts = make_prompts_table(rb_sampled, cfg["split"], cfg["seed"])
    prompts.to_csv(PROCESSED_DIR / "prompts.csv", index=False)

    # reward matrix, one per lambda
    train_ids = prompts.loc[prompts["split"] == "train", "sample_id"]
    for scenario, lam in cfg["lambdas"].items():
        # computes reward: quality - lambda_dollar * cost - lambda_carbon * carbon_cost
        rewards = make_rewards(prompt_model_pair, prompts, lam)
        rewards.to_csv(PROCESSED_DIR / f"rewards_{scenario}.csv")

        # single model baseline
        print(f"\n{scenario} (lambda dollar = {lam['dollar']}, carbon = {lam['carbon']}): mean train reward per model")
        print(rewards.loc[train_ids].mean().round(3))

    # sanity check on dimensions of each task and split
    print("\nSplit sizes:")
    print(pd.crosstab(prompts["task"], prompts["split"]))


if __name__ == "__main__":
    main()
