
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "corruption_evaluation"

INPUT_FILE = RESULTS_DIR / "corruption_metrics.csv"
CLEAN_FILE = RESULTS_DIR / "corruption_metrics_clean.csv"
SUMMARY_FILE = RESULTS_DIR / "corruption_summary.csv"


def main():
    df = pd.read_csv(INPUT_FILE)

    keys = ["corruption", "severity"]
    duplicate_mask = df.duplicated(subset=keys, keep=False)

    print(f"Rows in original file: {len(df)}")
    print(f"Rows involved in duplicates: {duplicate_mask.sum()}")

    if duplicate_mask.any():
        print("\nDuplicate experiment rows:")
        print(
            df.loc[duplicate_mask, keys + ["accuracy", "macro_f1"]]
            .sort_values(keys)
            .to_string(index=False)
        )

    # Keep the first recorded result for each experiment.
    clean_df = df.drop_duplicates(subset=keys, keep="first").copy()
    clean_df = clean_df.sort_values(keys).reset_index(drop=True)

    expected = 14 * 5
    if len(clean_df) != expected:
        raise ValueError(
            f"Expected {expected} unique experiments, "
            f"but found {len(clean_df)}."
        )

    clean_df.to_csv(CLEAN_FILE, index=False)

    # Summarize average accuracy and macro F1 across five severities.
    summary = (
        clean_df.groupby("corruption")
        .agg(
            mean_accuracy=("accuracy", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            severity_1_accuracy=("accuracy", "first"),
            severity_5_accuracy=("accuracy", "last"),
        )
        .sort_values("mean_accuracy")
        .reset_index()
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    print(f"\nClean results saved to: {CLEAN_FILE}")
    print(f"Summary saved to: {SUMMARY_FILE}")
    print("\nCorruptions ranked by mean accuracy (lowest first):")
    print(summary.to_string(index=False))

    # Plot accuracy against severity for all corruption types.
    plt.figure(figsize=(13, 7))

    for corruption, group in clean_df.groupby("corruption"):
        plt.plot(
            group["severity"],
            group["accuracy"] * 100,
            marker="o",
            label=corruption,
        )

    plt.xlabel("Corruption severity")
    plt.ylabel("Accuracy (%)")
    plt.title("Swin Transformer Accuracy Under Image Corruptions")
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "accuracy_by_severity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Compare average accuracy across corruption types.
    summary_sorted = summary.sort_values("mean_accuracy")

    plt.figure(figsize=(11, 7))
    plt.barh(
        summary_sorted["corruption"],
        summary_sorted["mean_accuracy"] * 100,
    )
    plt.xlabel("Mean accuracy across five severities (%)")
    plt.ylabel("Corruption type")
    plt.title("Average Robustness by Corruption Type")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / "mean_accuracy_by_corruption.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print("\nPlots saved:")
    print(RESULTS_DIR / "accuracy_by_severity.png")
    print(RESULTS_DIR / "mean_accuracy_by_corruption.png")


if __name__ == "__main__":
    main()
