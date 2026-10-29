
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "corruption_aware"

SUMMARY_CSV = RESULTS / "baseline_vs_proposed_summary.csv"
CONDITION_CSV = RESULTS / "baseline_vs_proposed_by_condition.csv"

PLOT_DIR = RESULTS / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    summary = pd.read_csv(SUMMARY_CSV)
    conditions = pd.read_csv(CONDITION_CSV)

    # Figure 1: Mean accuracy by corruption type.
    plot_data = summary.sort_values("proposed_accuracy")

    plt.figure(figsize=(11, 7))
    plt.barh(
        plot_data["corruption"],
        plot_data["baseline_accuracy"] * 100,
        label="Fine-tuned Swin baseline",
    )
    plt.barh(
        plot_data["corruption"],
        plot_data["proposed_accuracy"] * 100,
        alpha=0.75,
        label="Corruption-aware Swin",
    )
    plt.xlabel("Mean accuracy (%)")
    plt.ylabel("Corruption type")
    plt.title("Accuracy by corruption type")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "accuracy_by_corruption.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Figure 2: Improvement for each corruption type.
    improvement = summary.sort_values("improvement_pp")

    plt.figure(figsize=(10, 7))
    plt.barh(
        improvement["corruption"],
        improvement["improvement_pp"],
    )
    plt.axvline(0, linewidth=1)
    plt.xlabel("Accuracy improvement (percentage points)")
    plt.ylabel("Corruption type")
    plt.title("Robustness improvement over fine-tuned Swin")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "improvement_by_corruption.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Figure 3: Accuracy versus corruption severity.
    severity = (
        conditions.groupby("severity", as_index=False)
        .agg(
            baseline_accuracy=("baseline_accuracy", "mean"),
            proposed_accuracy=("proposed_accuracy", "mean"),
        )
    )

    plt.figure(figsize=(8, 5))
    plt.plot(
        severity["severity"],
        severity["baseline_accuracy"] * 100,
        marker="o",
        label="Fine-tuned Swin baseline",
    )
    plt.plot(
        severity["severity"],
        severity["proposed_accuracy"] * 100,
        marker="o",
        label="Corruption-aware Swin",
    )
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel("Corruption severity")
    plt.ylabel("Mean accuracy (%)")
    plt.title("Robustness across corruption severity levels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "accuracy_by_severity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("Plots saved to:", PLOT_DIR)
    for filename in [
        "accuracy_by_corruption.png",
        "improvement_by_corruption.png",
        "accuracy_by_severity.png",
    ]:
        print(PLOT_DIR / filename)


if __name__ == "__main__":
    main()

