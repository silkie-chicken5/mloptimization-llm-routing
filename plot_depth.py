"""Plot optimal vs greedy trees across depths, from results/depth_comparison.csv.

Writes results/depth_comparison.png with one column per scenario:
  row 1: test reward vs max depth
  row 2: test reward vs number of leaves (reward per unit of complexity)
  row 3: optimal minus greedy test reward, with 95% bootstrap interval
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
BLUE, ORANGE, MUTED = "#2a78d6", "#eb6834", "#8a8984"

runs = pd.read_csv(RESULTS / "depth_comparison.csv")
diffs = pd.read_csv(RESULTS / "depth_differences.csv")
scenarios = list(runs["scenario"].unique())

fig, axes = plt.subplots(3, len(scenarios), figsize=(4.5 * len(scenarios), 10), squeeze=False)

for c, scenario in enumerate(scenarios):
    r = runs[runs["scenario"] == scenario]
    for method, color, label in [("optimal_tree", BLUE, "Optimal tree"), ("greedy_tree", ORANGE, "Greedy tree")]:
        m = r[r["method"] == method].sort_values("depth")
        axes[0, c].plot(m["depth"], m["test"], marker="o", color=color, label=label)
        axes[1, c].plot(m["leaves"], m["test"], marker="o", color=color, label=label)

    d = diffs[diffs["scenario"] == scenario].sort_values("depth")
    below = (d["optimal_minus_greedy"] - d["ci_low"]).clip(lower=0)
    above = (d["ci_high"] - d["optimal_minus_greedy"]).clip(lower=0)
    axes[2, c].errorbar(d["depth"], d["optimal_minus_greedy"], yerr=[below, above],
                        fmt="o", color=BLUE, capsize=4)
    axes[2, c].axhline(0, color=MUTED, lw=1)

    axes[0, c].set_title(scenario, loc="left")
    axes[0, c].set_xlabel("Max depth")
    axes[1, c].set_xlabel("Leaves")
    axes[2, c].set_xlabel("Max depth")

axes[0, 0].set_ylabel("Test reward")
axes[1, 0].set_ylabel("Test reward")
axes[2, 0].set_ylabel("Optimal − greedy\n(95% interval)")
axes[0, 0].legend(frameon=False)

fig.tight_layout()
fig.savefig(RESULTS / "depth_comparison.png", dpi=200)
print("saved results/depth_comparison.png")