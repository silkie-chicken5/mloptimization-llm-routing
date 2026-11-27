"""k-nearest-neighbor router: a RouterBench-style comparison for the policy trees.

For each prompt, find the k most similar training prompts (the same five features the trees use),
average each model's reward over those neighbors, and route to the model with the highest average.
k is tuned on the validation prompts for each scenario.

Writes:
  results/knn_picks.csv    test picks, same format as test_picks.csv, so bootstrap.py can score them
  results/knn_summary.csv  chosen k and validation/test reward per scenario

Run from the repo root after process_data.py:  python knn_router.py
"""
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
K_GRID = [1, 5, 10, 25, 50, 100, 200, 400]

with open(ROOT / "config.toml", "rb") as f:
    cfg = tomllib.load(f)


def feature_matrix(prompts, train_mask):
    """One-hot task plus standardized numeric features, scaled using the training rows only."""
    task = pd.get_dummies(prompts["task"], dtype=float)
    numeric = prompts[["prompt_len_words", "has_numbers", "has_code", "n_questions"]].astype(float)
    # log length, so a few very long prompts don't dominate the distances
    numeric["prompt_len_words"] = np.log1p(numeric["prompt_len_words"])
    mean = numeric[train_mask].mean()
    std = numeric[train_mask].std().replace(0, 1)
    numeric = (numeric - mean) / std
    return pd.concat([task, numeric], axis=1).to_numpy()


def nearest_neighbors(X_query, X_train, k_max):
    """Positions of the k_max closest training rows for each query row (Euclidean distance)."""
    squared_dist = (
        (X_query ** 2).sum(axis=1)[:, None]
        - 2 * X_query @ X_train.T
        + (X_train ** 2).sum(axis=1)[None, :]
    )
    return np.argsort(squared_dist, axis=1, kind="stable")[:, :k_max]


def knn_picks(neighbors, R_train, k, model_names):
    """Route each query to the model with the highest mean reward among its k nearest neighbors."""
    neighbor_rewards = R_train[neighbors[:, :k]]       # queries x k x models
    predicted = neighbor_rewards.mean(axis=1)          # queries x models
    return np.array(model_names)[predicted.argmax(axis=1)]


def picked_reward(R, picks, model_names):
    """Mean reward when each row is sent to its picked model."""
    columns = [model_names.index(p) for p in picks]
    return R[np.arange(len(picks)), columns].mean()


# ---------- Data ----------
prompts = pd.read_csv(PROCESSED / "prompts.csv")
pairs = pd.read_csv(PROCESSED / "pairs.csv", usecols=["sample_id", "model", "quality", "cost", "carbon_cost"])
pairs["model"] = pairs["model"].str.split("/").str[-1]   # same short names as the Julia picks

# prompts x models tables, rows in prompts.csv order
wide = {
    column: pairs.pivot(index="sample_id", columns="model", values=column).loc[prompts["sample_id"]]
    for column in ["quality", "cost", "carbon_cost"]
}
model_names = list(wide["quality"].columns)

train = (prompts["split"] == "train").to_numpy()
val = (prompts["split"] == "val").to_numpy()
test = (prompts["split"] == "test").to_numpy()

# Neighbors depend only on the features, not on lambda, so find them once
X = feature_matrix(prompts, train)
neighbors_val = nearest_neighbors(X[val], X[train], max(K_GRID))
neighbors_test = nearest_neighbors(X[test], X[train], max(K_GRID))

# ---------- One KNN router per scenario ----------
summary_rows = []
pick_tables = []

for scenario, lam in cfg["lambdas"].items():
    R = (wide["quality"] - lam["dollar"] * wide["cost"] - lam["carbon"] * wide["carbon_cost"]).to_numpy()
    R_train, R_val, R_test = R[train], R[val], R[test]

    # tune k on validation
    val_scores = {}
    for k in K_GRID:
        picks = knn_picks(neighbors_val, R_train, k, model_names)
        val_scores[k] = picked_reward(R_val, picks, model_names)
    best_k = max(val_scores, key=val_scores.get)

    test_picks = knn_picks(neighbors_test, R_train, best_k, model_names)
    summary_rows.append({
        "scenario": scenario,
        "best_k": best_k,
        "val": val_scores[best_k],
        "test": picked_reward(R_test, test_picks, model_names),
        "models_used_on_test": len(set(test_picks)),
    })
    pick_tables.append(pd.DataFrame({
        "scenario": scenario,
        "method": "knn",
        "sample_id": prompts.loc[test, "sample_id"].to_numpy(),
        "model": test_picks,
    }))

summary = pd.DataFrame(summary_rows)
summary.round(4).to_csv(RESULTS / "knn_summary.csv", index=False)
pd.concat(pick_tables, ignore_index=True).to_csv(RESULTS / "knn_picks.csv", index=False)
print(summary.round(4).to_string(index=False))