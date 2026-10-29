
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results" / "controlled_comparison" / "controlled_results.csv"
OUT = ROOT / "results" / "controlled_comparison"
PLOTS = OUT / "plots"

def main():
    df = pd.read_csv(INPUT)
    PLOTS.mkdir(parents=True, exist_ok=True)

    # Summary by corruption type.
    by_corruption = (
        df.groupby("corruption", as_index=False)
        .agg(
            baseline_accuracy=("baseline_accuracy", "mean"),
            proposed_accuracy=("proposed_accuracy", "mean"),
            improvement_pp=("accuracy_improvement_pp", "mean"),
            baseline_macro_f1=("baseline_macro_f1", "mean"),
            proposed_macro_f1=("proposed_macro_f1", "mean"),
        )
        .sort_values("improvement_pp", ascending=False)
    )
    by_corruption.to_csv(OUT / "summary_by_corruption.csv", index=False)

    # Summary by severity.
    by_severity = (
        df.groupby("severity", as_index=False)
        .agg(
            baseline_accuracy=("baseline_accuracy", "mean"),
            proposed_accuracy=("proposed_accuracy", "mean"),
            improvement_pp=("accuracy_improvement_pp", "mean"),
        )
    )
    by_severity.to_csv(OUT / "summary_by_severity.csv", index=False)

    print("Number of conditions:", len(df))
    print(f"Baseline mean accuracy: {df['baseline_accuracy'].mean():.4%}")
    print(f"Proposed mean accuracy: {df['proposed_accuracy'].mean():.4%}")
    print(
        "Mean improvement: "
        f"{df['accuracy_improvement_pp'].mean():+.2f} percentage points"
    )

    print("\nMean accuracy by severity:")
    print(by_severity.to_string(index=False))

    print("\nLargest improvements by corruption:")
    print(by_corruption.head(5).to_string(index=False))

    print("\nLowest proposed accuracy by corruption:")
    print(
        by_corruption.sort_values("proposed_accuracy")
        .head(5)
        .to_string(index=False)
    )

    # Figure 1: Baseline versus proposed accuracy.
    plot_df = by_corruption.sort_values("proposed_accuracy")

    plt.figure(figsize=(11, 7))
    plt.barh(
        plot_df["corruption"],
        plot_df["baseline_accuracy"] * 100,
        label="Fine-tuned Swin",
    )
    plt.barh(
        plot_df["corruption"],
        plot_df["proposed_accuracy"] * 100,
        alpha=0.75,
        label="Corruption-aware Swin",
    )
    plt.xlabel("Mean accuracy (%)")
    plt.ylabel("Corruption")
    plt.title("Controlled robustness comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "controlled_accuracy_by_corruption.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 2: Accuracy by severity.
    plt.figure(figsize=(8, 5))
    plt.plot(
        by_severity["severity"],
        by_severity["baseline_accuracy"] * 100,
        marker="o",
        label="Fine-tuned Swin",
    )
    plt.plot(
        by_severity["severity"],
        by_severity["proposed_accuracy"] * 100,
        marker="o",
        label="Corruption-aware Swin",
    )
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel("Severity level")
    plt.ylabel("Mean accuracy (%)")
    plt.title("Controlled accuracy across severity levels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "controlled_accuracy_by_severity.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSummary files:", OUT)
    print("Plots:", PLOTS)


if __name__ == "__main__":
    main()
