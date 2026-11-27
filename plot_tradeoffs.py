"""Plot the quality vs. dollar / CO2 tradeoff from results/lambda_sweep.csv (made by sweep_lambda.jl).

Writes two figures per split to results/:
  tradeoff_frontier_<split>.png  quality vs $ and vs CO2, one curve per method across lambda
  tradeoff_relative_<split>.png  % quality lost and % $ / CO2 saved vs always using Yi-34B, across lambda

Use the *_val figures to choose lambdas; report the *_test figures (test never informs the choice).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
REFERENCE_MODEL = "Yi-34B-Chat"
Y_FLOOR = -100   # relative chart: values below this are drawn as off-scale markers   

# Named scenarios from config.toml, located on the sweep they belong to
NAMED = {
    "joint": {100: "quality_first", 1000: "balanced", 3000: "efficiency_first"},
    "carbon_only": {30000: "carbon_first"},
}
# (x, y) label offsets in points, per x-axis column, to keep annotations from colliding
LABEL_OFFSETS = {
    "dollars_per_1k": {"quality_first": (-40, 22), "balanced": (-5, 30), "efficiency_first": (12, -38),
                       "carbon_first": (-40, 22)},
    "kg_co2_per_1k": {"quality_first": (-30, 22), "balanced": (-20, -42), "efficiency_first": (14, -40),
                      "carbon_first": (-20, 22)},
}
# balanced and efficiency_first sit close together on the log axis: anchor them on opposite sides
SCENARIO_HA = {"balanced": "right", "efficiency_first": "left"}
MODEL_LABEL_OFFSET = {"Yi-34B-Chat": (8, 6), "mistral-7b-chat": (8, 4)}
SWEEP_TITLES = {
    "joint": "Joint sweep (λ_dollar = λ_carbon = λ)",
    "carbon_only": "Carbon-only sweep (λ_dollar = 0, λ_carbon = λ)",
}

# Categorical slots 1-3 of the reference palette (validated light mode); neutrals for references
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "legend.frameon": False,
})


def lam(df):
    """The swept lambda (lambda_carbon is the swept value in both sweeps)."""
    return df["lambda_carbon"]


def plot_frontier(sweep_df, refs, split):
    cost_cols = [("dollars_per_1k", "$ per 1,000 queries", "linear"),
                 ("kg_co2_per_1k", "kg CO₂ per 1,000 queries (log scale)", "log")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=True)

    for r, sweep in enumerate(["joint", "carbon_only"]):
        d = sweep_df[sweep_df["sweep"] == sweep].sort_values("lambda_carbon")
        for c, (col, xlabel, xscale) in enumerate(cost_cols):
            ax = axes[r, c]
            ax.set_xscale(xscale)

            # single-model references: every model as a hollow marker, labeled
            for _, m in refs.iterrows():
                ax.scatter(m[col], m["quality"], s=60, facecolors="none", edgecolors=MUTED, linewidths=1.5, zorder=2)
                ax.annotate(m["model"], (m[col], m["quality"]), textcoords="offset points",
                            xytext=MODEL_LABEL_OFFSET.get(m["model"], (6, -12)),
                            fontsize=8, color=INK_2)

            # best single model at each lambda (filled square; sits on top of a reference marker)
            bs = d[d["method"] == "best_single"]
            ax.scatter(bs[col], bs["quality"], marker="s", s=70, color=INK_2, edgecolors=SURFACE,
                       linewidths=2, zorder=3, label="Best single model (per λ)")

            # oracle frontier
            o = d[d["method"] == "oracle"]
            ax.plot(o[col], o["quality"], ls="--", lw=1.5, color=MUTED, marker="o", ms=4, zorder=3,
                    label="Oracle (hindsight best per prompt)")

            # tree curves
            for method, color, label in [("optimal_tree", BLUE, "Optimal policy tree"),
                                         ("greedy_tree", ORANGE, "Greedy (CART) policy tree")]:
                t = d[d["method"] == method]
                ax.plot(t[col], t["quality"], lw=2, color=color, marker="o", ms=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4, label=label)

            # label the named scenarios on the optimal tree curve
            t = d[d["method"] == "optimal_tree"]
            for lam_value, name in NAMED[sweep].items():
                pt = t[lam(t) == lam_value]
                if len(pt):
                    ax.annotate(f"{name}\n(λ={lam_value:,})", (pt[col].iloc[0], pt["quality"].iloc[0]),
                                textcoords="offset points", xytext=LABEL_OFFSETS[col][name], fontsize=8, color=INK,
                                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))

            # named scenarios on neither sweep (carbon_realistic: lambda_dollar=1000, lambda_carbon=5000)
            if sweep == "joint":
                off = sweep_df[(sweep_df["sweep"] == "off_sweep") & (sweep_df["method"] == "optimal_tree")]
                for _, pt in off.iterrows():
                    ax.scatter(pt[col], pt["quality"], marker="*", s=180, color=BLUE, edgecolors=SURFACE,
                               linewidths=1.5, zorder=5,
                               label=f"{pt['scenario']}, optimal tree (λ$={pt['lambda_dollar']:,.0f}, "
                                     f"λCO₂={pt['lambda_carbon']:,.0f})")

            ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(f"Mean quality on {split} set")
            ax.set_title(SWEEP_TITLES[sweep] if c == 0 else "", loc="left", fontsize=11, color=INK)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"[{split.upper()} set] Quality vs. cost across λ: moving left saves $ / CO₂, moving down loses quality",
                 x=0.02, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(RESULTS_DIR / f"tradeoff_frontier_{split}.png", dpi=200)
    plt.close(fig)


def plot_relative(sweep_df, refs, split):
    base = refs.set_index("model").loc[REFERENCE_MODEL]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for ax, sweep in zip(axes, ["joint", "carbon_only"]):
        t = sweep_df[(sweep_df["sweep"] == sweep) & (sweep_df["method"] == "optimal_tree")].sort_values("lambda_carbon")
        x = lam(t)
        series = [
            ("Quality lost", BLUE, 100 * (1 - t["quality"] / base["quality"])),
            ("$ saved", ORANGE, 100 * (1 - t["dollars_per_1k"] / base["dollars_per_1k"])),
            ("CO₂ saved", AQUA, 100 * (1 - t["kg_co2_per_1k"] / base["kg_co2_per_1k"])),
        ]
        end_labels = []
        for name, color, y in series:
            shown = y.clip(lower=Y_FLOOR)
            ax.plot(x, shown, lw=2, color=color, marker="o", ms=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5, label=name, zorder=3)
            # points below the floor: down-triangle at the floor + the real value
            off = y < Y_FLOOR
            ax.scatter(x[off], shown[off], marker="v", s=70, color=color, edgecolors=SURFACE, zorder=4)
            for xi, yi in zip(x[off], y[off]):
                ax.annotate(f"{yi:+.0f}%", (xi, Y_FLOOR), textcoords="offset points", xytext=(0, -14),
                            ha="center", fontsize=8, color=INK_2)
            end_labels.append([name, x.iloc[-1], shown.iloc[-1]])

        # direct labels at the line ends (text ink, not series color), nudged apart vertically
        end_labels.sort(key=lambda e: e[2])
        for i in range(1, len(end_labels)):
            end_labels[i][2] = max(end_labels[i][2], end_labels[i - 1][2] + 9)
        for name, xe, ye in end_labels:
            ax.annotate(name, (xe, ye), textcoords="offset points", xytext=(10, 0), va="center", fontsize=9, color=INK)

        # named scenarios: faint vertical rule + name just above the plot area (keeps the plot area clear)
        for lam_value, scen in NAMED[sweep].items():
            ax.axvline(lam_value, color=GRID, lw=1.2, zorder=1)
            ax.annotate(scen, (lam_value, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 4),
                        textcoords="offset points", ha=SCENARIO_HA.get(scen, "center"), va="bottom", fontsize=8, color=INK_2)

        ax.axhline(0, color=MUTED, lw=1, zorder=2)
        ax.set_ylim(Y_FLOOR - 25, 100)
        ax.set_xscale("log")
        ax.set_xlabel("λ (log scale)")
        ax.set_title(SWEEP_TITLES[sweep], loc="left", fontsize=11, color=INK, pad=22)
        ax.margins(x=0.15)

    axes[0].set_ylabel(f"% vs. always using {REFERENCE_MODEL}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"[{split.upper()} set] Optimal policy tree: % saved vs. % quality lost, relative to always routing to {REFERENCE_MODEL}\n"
                 "Negative 'saved' = uses more than Yi; ▼ marks points below −100% with their actual value",
                 x=0.02, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(RESULTS_DIR / f"tradeoff_relative_{split}.png", dpi=200)
    plt.close(fig)


def main():
    df = pd.read_csv(RESULTS_DIR / "lambda_sweep.csv", keep_default_na=False, na_values=[""])
    df["scenario"] = df["scenario"].fillna("")
    for split in ["val", "test"]:
        d = df[df["split"] == split]
        refs = d[d["sweep"] == "reference"]
        sweep_df = d[d["sweep"] != "reference"]
        plot_frontier(sweep_df, refs, split)
        plot_relative(sweep_df, refs, split)


if __name__ == "__main__":
    main()
