"""Plot where each policy tree routes the test prompts, per scenario and task, from results/routing_by_task.csv.

Writes results/routing_by_task_<method>.png for the optimal and greedy trees:
one panel per scenario, one 100% stacked bar per task, segments = share of prompts sent to each model.
"""
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"

# Models smallest -> largest; each keeps its color in every panel (color follows the model)
MODELS = ["mistral-7b-chat", "mixtral-8x7b-chat", "Yi-34B-Chat", "llama-2-70b-chat"]
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # reference palette slots 1-4 (validated)
LABEL_ON_FILL = ["#ffffff", "#ffffff", "#ffffff", "#0b0b0b"]  # in-segment text: white, near-black on yellow
TASKS = ["mmlu", "hellaswag", "arc", "gsm8k", "mbpp", "all"]  # multiple choice, then math / code, then overall
TASK_LABELS = {"mmlu": "MMLU", "hellaswag": "HellaSwag", "arc": "ARC", "gsm8k": "GSM8K (math)",
               "mbpp": "MBPP (code)", "all": "All tasks"}
METHOD_TITLES = {"optimal_tree": "Optimal policy tree", "greedy_tree": "Greedy (CART) policy tree"}
MIN_LABEL_SHARE = 0.12   # only label segments wide enough to hold the text

INK, INK_2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2,
    "ytick.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "legend.frameon": False,
})


def plot_method(routing, lambdas, method):
    d = routing[routing["method"] == method]
    # same scenario order as policy_trees.jl: by total weight
    scenarios = sorted(d["scenario"].unique(), key=lambda s: lambdas[s]["dollar"] + lambdas[s]["carbon"])

    fig, axes = plt.subplots(1, len(scenarios), figsize=(3.3 * len(scenarios), 4.6), sharey=True)
    y = [len(TASKS) - i for i in range(len(TASKS))]
    y[-1] -= 0.4   # small gap before the "All tasks" bar

    for ax, scenario in zip(axes, scenarios):
        share = d[d["scenario"] == scenario].pivot(index="task", columns="model", values="share").loc[TASKS, MODELS]
        left = pd.Series(0.0, index=TASKS)
        for model, color, text_color in zip(MODELS, MODEL_COLORS, LABEL_ON_FILL):
            w = share[model]
            ax.barh(y, w, left=left, height=0.72, color=color, edgecolor=SURFACE, linewidth=1.5, label=model)
            for yi, wi, li in zip(y, w, left):
                if wi >= MIN_LABEL_SHARE:
                    ax.text(li + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=8, color=text_color)
            left += w

        lam = lambdas[scenario]
        ax.set_title(f"{scenario}\nλ$ = {lam['dollar']:,}, λCO₂ = {lam['carbon']:,}", fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
        ax.tick_params(axis="y", length=0)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)

    axes[0].set_yticks(y, [TASK_LABELS[t] for t in TASKS])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.01),
               title="Share of test prompts routed to", title_fontsize=9)
    fig.suptitle(f"{METHOD_TITLES[method]}: where each task's test prompts are routed, by scenario",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(RESULTS_DIR / f"routing_by_task_{method}.png", dpi=200)
    plt.close(fig)


def main():
    routing = pd.read_csv(RESULTS_DIR / "routing_by_task.csv")
    with open(ROOT_DIR / "config.toml", "rb") as f:
        lambdas = tomllib.load(f)["lambdas"]
    for method in ["optimal_tree", "greedy_tree"]:
        plot_method(routing, lambdas, method)


if __name__ == "__main__":
    main()
