"""Plot aggregate fixed-prefix results, without smoothing or individual examples."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    source = Path(os.environ["P2_SUMMARY"])
    summary = json.loads(source.read_text())
    output = Path(os.environ.get("P2_PLOT_DIR", str(source.parent)))
    output.mkdir(parents=True, exist_ok=True)
    if summary["common_n"] < 32:
        print("Engineering-scale sample: use exact-count tables instead of trend plots.")
        return
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 180, "pdf.fonttype": 42})
    colors = {"a": "#485563", "b": "#BE6B12", "c": "#147D92", "d": "#8B4F9D"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.1), sharey=True, constrained_layout=True)
    for axis, phase, title in zip(axes, ["greedy_before", "greedy_after"], ["Before current-case update", "After current-case update"]):
        for key, method in summary["methods"].items():
            metric = "consensus" if key == "b" else "greedy_before" if key == "a" else phase
            blocks = method["online_blocks"]
            axis.plot([point["cursor"] for point in blocks],
                      [point[metric + "_cumulative"]["percent"] for point in blocks],
                      marker="o", markersize=3.5, color=colors[key], label=method["name"])
        axis.set(title=title, xlabel="Completed stream cases")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Cumulative accuracy (%)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Fixed development stream · n={summary['common_n']} · seed {summary['seed']}", fontsize=12)
    fig.savefig(output / "continual_online.png")
    fig.savefig(output / "continual_online.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    for key, method in summary["methods"].items():
        points = method["probe"]
        for axis, metric in zip(axes, ["percent", "delta_from_initial_pp"]):
            axis.plot([point["cursor"] for point in points], [point[metric] for point in points],
                      marker="o", color=colors[key], linestyle="--" if key == "b" else "-", label=method["name"])
    axes[0].set(title="Greedy probe accuracy", ylabel="Accuracy (%)")
    axes[1].set(title="Change from initialization", ylabel="Change (percentage points)")
    axes[1].axhline(0, color="#999999", linewidth=0.6)
    for axis in axes:
        axis.set_xlabel("Completed stream cases")
        axis.grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Held-out probe · {summary['probe_n']} image-reference groups · greedy for all actors", fontsize=12)
    fig.savefig(output / "continual_probe.png")
    fig.savefig(output / "continual_probe.pdf")
    plt.close(fig)
    print("Wrote aggregate online and probe PNG/PDF figures.")


if __name__ == "__main__":
    main()
