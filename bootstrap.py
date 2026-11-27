"""Bootstrap 95% intervals for the test-set results.

Reads results/test_picks.csv (from policy_trees.jl) and data/processed/pairs.csv.
Resamples test prompts with replacement within each task (the test set is stratified
by task), using the same resampled prompts for every method, so differences are paired.

Writes:
  results/bootstrap_reward.csv    RQ2: mean test reward per method, differences, share of oracle gain
  results/bootstrap_outcomes.csv  RQ1: quality, $ and kg CO2 per 1,000 queries, and % change vs always Yi-34B

Run from the repo root after policy_trees.jl:  python bootstrap.py
"""
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
N_BOOT = 2000
REFERENCE_MODEL = "Yi-34B-Chat"   # fixed reference for RQ1, the same in every scenario

with open(ROOT / "config.toml", "rb") as f:
    cfg = tomllib.load(f)
rng = np.random.default_rng(cfg["seed"])


def stratified_draws(tasks, n_boot):
    """n_boot x n matrix of row positions, resampled with replacement within each task."""
    tasks = np.asarray(tasks)
    draws = np.empty((n_boot, len(tasks)), dtype=int)
    for task in np.unique(tasks):
        rows = np.flatnonzero(tasks == task)
        draws[:, rows] = rng.choice(rows, size=(n_boot, len(rows)))
    return draws


def summarize(point, boot):
    """Point estimate with a 95% percentile interval from the bootstrap values."""
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"estimate": point, "ci_low": low, "ci_high": high}


# ---------- What each pick actually produced ----------
pairs = pd.read_csv(
    ROOT / "data" / "processed" / "pairs.csv",
    usecols=["sample_id", "task", "model", "quality", "cost", "wh", "carbon_cost"],
)
pairs["model"] = pairs["model"].str.split("/").str[-1]           # same short names as the Julia picks
pairs["kg_co2"] = pairs["wh"] / 1000 * cfg["grid_intensity"]     # Wh -> kWh -> kg CO2

picks = pd.read_csv(RESULTS / "test_picks.csv")
# Add the KNN router's test picks (from knn_router.py) so it is scored like every other method
picks = pd.concat([picks, pd.read_csv(RESULTS / "knn_picks.csv")], ignore_index=True)

# Add "always Yi-34B" as a method in every scenario, for the RQ1 comparison
test_ids = np.sort(picks["sample_id"].unique())
always_ref = pd.concat([
    pd.DataFrame({"scenario": s, "method": "always_yi", "sample_id": test_ids, "model": REFERENCE_MODEL})
    for s in picks["scenario"].unique()
])
picks = pd.concat([picks, always_ref], ignore_index=True)

chosen = picks.merge(pairs, on=["sample_id", "model"], how="left", validate="many_to_one")
assert chosen["quality"].notna().all(), "some picks did not match pairs.csv"

lam_dollar = chosen["scenario"].map(lambda s: cfg["lambdas"][s]["dollar"])
lam_carbon = chosen["scenario"].map(lambda s: cfg["lambdas"][s]["carbon"])
chosen["reward"] = chosen["quality"] - lam_dollar * chosen["cost"] - lam_carbon * chosen["carbon_cost"]

# ---------- Check: rebuilt rewards match the Julia results ----------
julia = pd.read_csv(RESULTS / "comparison.csv")[["scenario", "method", "test"]]
rebuilt = chosen.groupby(["scenario", "method"])["reward"].mean().rename("rebuilt").reset_index()
check = julia.merge(rebuilt, on=["scenario", "method"])
gap = (check["test"] - check["rebuilt"]).abs().max()
print(f"largest gap vs comparison.csv: {gap:.2e}")
assert gap < 1e-6, "picks do not reproduce the Julia test rewards; fix that before trusting the intervals"

# ---------- Bootstrap draws: one set, shared by every scenario and method ----------
task_of = pairs.drop_duplicates("sample_id").set_index("sample_id")["task"]
draws = stratified_draws(task_of.loc[test_ids], N_BOOT)

reward_rows = []
outcome_rows = []

for scenario, rows in chosen.groupby("scenario"):
    # one prompts x methods table per metric, rows in test_ids order
    wide = {
        metric: rows.pivot(index="sample_id", columns="method", values=metric).loc[test_ids]
        for metric in ["reward", "quality", "cost", "kg_co2"]
    }

    # ---- RQ2: reward, differences between methods, share of oracle gain ----
    reward = wide["reward"]
    point = reward.mean()
    boot = pd.DataFrame(reward.to_numpy()[draws].mean(axis=1), columns=reward.columns)

    for method in ["optimal_tree", "greedy_tree", "best_single", "knn", "cheapest", "oracle"]:
        reward_rows.append({"scenario": scenario, "quantity": f"reward: {method}",
                            **summarize(point[method], boot[method])})

    for a, b in [("optimal_tree", "greedy_tree"), ("optimal_tree", "best_single"), ("greedy_tree", "best_single"), ("optimal_tree", "knn"), ("knn", "best_single")]:
        reward_rows.append({"scenario": scenario, "quantity": f"difference: {a} - {b}",
                            **summarize(point[a] - point[b], boot[a] - boot[b])})

    for method in ["optimal_tree", "greedy_tree", "knn"]:
        share_point = (point[method] - point["best_single"]) / (point["oracle"] - point["best_single"])
        share_boot = (boot[method] - boot["best_single"]) / (boot["oracle"] - boot["best_single"])
        reward_rows.append({"scenario": scenario, "quantity": f"share of oracle gain: {method}",
                            **summarize(share_point, share_boot)})

    # ---- RQ1: quality, $ and CO2 per 1,000 queries, and % change vs always Yi-34B ----
    for metric, scale, label in [("quality", 1, "quality"),
                                 ("cost", 1000, "dollars_per_1k"),
                                 ("kg_co2", 1000, "kg_co2_per_1k")]:
        values = wide[metric]
        point = values.mean() * scale
        boot = pd.DataFrame(values.to_numpy()[draws].mean(axis=1) * scale, columns=values.columns)

        for method in values.columns:
            outcome_rows.append({"scenario": scenario, "method": method, "metric": label,
                                 "kind": "level", **summarize(point[method], boot[method])})
            if method == "always_yi":
                continue
            pct_point = 100 * (point[method] / point["always_yi"] - 1)
            pct_boot = 100 * (boot[method] / boot["always_yi"] - 1)
            outcome_rows.append({"scenario": scenario, "method": method, "metric": label,
                                 "kind": "% change vs always Yi-34B", **summarize(pct_point, pct_boot)})

reward_table = pd.DataFrame(reward_rows)
outcome_table = pd.DataFrame(outcome_rows)
reward_table.round(4).to_csv(RESULTS / "bootstrap_reward.csv", index=False)
outcome_table.round(4).to_csv(RESULTS / "bootstrap_outcomes.csv", index=False)

# ---------- Short printout of the numbers the report needs first ----------
pd.set_option("display.width", 200)
print("\n===== RQ2: differences in mean test reward (95% interval) =====")
print(reward_table[reward_table["quantity"].str.startswith("difference")].round(4).to_string(index=False))

print("\n===== RQ1: optimal tree, % change vs always Yi-34B (95% interval) =====")
mask = (outcome_table["method"] == "optimal_tree") & (outcome_table["kind"] != "level")
print(outcome_table[mask].round(1).to_string(index=False))