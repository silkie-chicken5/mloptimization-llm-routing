"""Filter on RouterBench (0-shot) data, add response lengths, energy cost, and carbon cost"""
import ast
import tomllib
from pathlib import Path
import pandas as pd

PYTHON_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PYTHON_DIR.parent / "config.toml"
RAW_PATH = PYTHON_DIR / "data" / "raw" / "routerbench_0shot.pkl"

MODELS = [
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
    "meta/llama-2-70b-chat",
]

# Keep tasks that are central to the Routerbench paper and cover
# knowledge, commonsense, math, and code
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
    n_per_task = (n_prompts - len(mbpp)) // (len(TASKS) - 1)  # (2500 - 427) // 4 = 518

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


def main():
    cfg = load_config_data()
    routerbench = pd.read_pickle(RAW_PATH)
    rb_sampled = filter_data(routerbench, cfg["n_prompts"], cfg["seed"])
    prompt_model_pair = add_features(
        rb_sampled, cfg["energy"], cfg["grid_intensity"], cfg["carbon_price"]
    )
    print(prompt_model_pair.shape)
    return prompt_model_pair


if __name__ == "__main__":
    main()
