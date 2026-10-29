
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

BASELINE_CSV = (
    ROOT / "results" / "corruption_evaluation"
    / "corruption_metrics_clean.csv"
)

PROPOSED_CSV = (
    ROOT / "results" / "corruption_aware" / "corruptions"
    / "corruption_metrics.csv"
)

OUTPUT_DIR = ROOT / "results" / "corruption_aware"


def main():
    baseline = pd.read_csv(BASELINE_CSV)
    proposed = pd.read_csv(PROPOSED_CSV)

    # Keep only the columns required for comparison.
    keys = ["corruption", "severity"]

    baseline = baseline[keys + ["accuracy"]].rename(
        columns={"accuracy": "baseline_accuracy"}
    )
    proposed = proposed[keys + ["accuracy"]].rename(
        columns={"accuracy": "proposed_accuracy"}
    )

    # Match every corruption type and severity level.
    comparison = baseline.merge(
        proposed,
        on=keys,
        how="inner",
        validate="one_to_one",
    )

    if len(comparison) != 70:
        raise ValueError(
            f"Expected 70 matched conditions, found {len(comparison)}. "
            "Check duplicate or missing corruption/severity rows."
        )

    comparison["improvement_percentage_points"] = (
        comparison["proposed_accuracy"]
        - comparison["baseline_accuracy"]
    ) * 100

    comparison.to_csv(
        OUTPUT_DIR / "baseline_vs_proposed_by_condition.csv",
        index=False,
    )

    summary = (
        comparison.groupby("corruption", as_index=False)
        .agg(
            baseline_accuracy=("baseline_accuracy", "mean"),
            proposed_accuracy=("proposed_accuracy", "mean"),
            improvement_pp=("improvement_percentage_points", "mean"),
        )
        .sort_values("improvement_pp")
    )

    summary.to_csv(
        OUTPUT_DIR / "baseline_vs_proposed_summary.csv",
        index=False,
    )

    print("Matched conditions:", len(comparison))
    print(
        f"Baseline mean corrupted accuracy: "
        f"{comparison['baseline_accuracy'].mean():.4%}"
    )
    print(
        f"Proposed mean corrupted accuracy: "
        f"{comparison['proposed_accuracy'].mean():.4%}"
    )
    print(
        f"Mean improvement: "
        f"{comparison['improvement_percentage_points'].mean():+.2f} "
        f"percentage points"
    )

    print("\nLargest improvements:")
    print(summary.tail(5).sort_values(
        "improvement_pp", ascending=False
    ).to_string(index=False))

    print("\nLargest regressions:")
    print(summary.head(5).to_string(index=False))

    print("\nSaved comparison files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
