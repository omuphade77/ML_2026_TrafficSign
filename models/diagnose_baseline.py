
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np


# --------------------------------------------------
# 1. Define file paths
# --------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "GTSRB"
TRAIN_CSV = DATA_DIR / "Train.csv"
TEST_CSV = DATA_DIR / "Test.csv"

RESULTS_DIR = PROJECT_ROOT / "results"
PREDICTIONS_CSV = RESULTS_DIR / "baseline_predictions.csv"
CONFUSION_CSV = RESULTS_DIR / "baseline_confusion_matrix.csv"


# --------------------------------------------------
# 2. Load CSV files
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("========== DATASET SIZES ==========")
print("Training rows:", len(train_df))
print("Test rows:", len(test_df))

print("\nTraining CSV columns:", train_df.columns.tolist())
print("Test CSV columns:", test_df.columns.tolist())


# --------------------------------------------------
# 3. Check class distributions
# --------------------------------------------------

expected_classes = set(range(43))

train_classes = set(train_df["ClassId"].unique())
test_classes = set(test_df["ClassId"].unique())

print("\n========== CLASS COVERAGE ==========")
print("Training classes:", len(train_classes))
print("Test classes:", len(test_classes))
print("Missing from training:", sorted(expected_classes - train_classes))
print("Missing from test:", sorted(expected_classes - test_classes))
print("Unexpected training IDs:", sorted(train_classes - expected_classes))
print("Unexpected test IDs:", sorted(test_classes - expected_classes))

train_counts = train_df["ClassId"].value_counts().reindex(
    range(43), fill_value=0
)
test_counts = test_df["ClassId"].value_counts().reindex(
    range(43), fill_value=0
)

distribution = pd.DataFrame({
    "class_id": range(43),
    "training_images": train_counts.values,
    "test_images": test_counts.values,
})

print("\n========== CLASS COUNTS ==========")
print(distribution.to_string(index=False))

distribution.to_csv(
    RESULTS_DIR / "class_distribution.csv",
    index=False,
)


# --------------------------------------------------
# 4. Check image paths
# --------------------------------------------------

def check_paths(df, split_name):
    paths = df["Path"].astype(str)
    missing = [
        path for path in paths
        if not (DATA_DIR / path).is_file()
    ]

    print(f"\n{split_name} missing image files:", len(missing))

    if missing:
        print("First missing paths:", missing[:10])


print("\n========== PATH CHECK ==========")
check_paths(train_df, "Training")
check_paths(test_df, "Test")


# --------------------------------------------------
# 5. Check prediction and confusion matrix files
# --------------------------------------------------

print("\n========== PREDICTION FILE ==========")

if PREDICTIONS_CSV.exists():
    predictions_df = pd.read_csv(PREDICTIONS_CSV)

    y_true = predictions_df["true_label"].to_numpy()
    y_pred = predictions_df["predicted_label"].to_numpy()

    print("Prediction rows:", len(predictions_df))
    print("True labels in valid range:",
          bool(np.all((y_true >= 0) & (y_true < 43))))
    print("Predicted labels in valid range:",
          bool(np.all((y_pred >= 0) & (y_pred < 43))))

    if len(y_true) == len(test_df):
        print("Prediction count matches test CSV: Yes")
    else:
        print("Prediction count matches test CSV: No")

    error_mask = y_true != y_pred
    errors = pd.DataFrame({
        "true_label": y_true[error_mask],
        "predicted_label": y_pred[error_mask],
    })

    print("Incorrect predictions:", int(error_mask.sum()))

    print("\nMost common true classes among errors:")
    print(
        errors["true_label"]
        .value_counts()
        .head(10)
        .to_string()
    )

    print("\nMost common wrong predictions:")
    print(
        errors["predicted_label"]
        .value_counts()
        .head(10)
        .to_string()
    )

else:
    print("Prediction file not found:", PREDICTIONS_CSV)


print("\n========== CONFUSION MATRIX ==========")

if CONFUSION_CSV.exists():
    matrix = np.loadtxt(
        CONFUSION_CSV,
        delimiter=",",
        dtype=int,
    )

    print("Matrix shape:", matrix.shape)
    print("Total examples represented:", int(matrix.sum()))

    if matrix.shape == (43, 43):
        off_diagonal = matrix.copy()
        np.fill_diagonal(off_diagonal, 0)

        # Row = actual class, column = predicted class.
        pairs = []

        for true_class in range(43):
            for predicted_class in range(43):
                count = int(off_diagonal[true_class, predicted_class])

                if count > 0:
                    pairs.append(
                        (count, true_class, predicted_class)
                    )

        pairs.sort(reverse=True)

        print("\nMost common class confusions:")
        for count, true_class, predicted_class in pairs[:15]:
            print(
                f"Actual {true_class} predicted as "
                f"{predicted_class}: {count} images"
            )
else:
    print("Confusion matrix not found:", CONFUSION_CSV)

print("\n========== LABEL ALIGNMENT CHECK ==========")

predictions_df = pd.read_csv(PREDICTIONS_CSV)

csv_labels = test_df["ClassId"].to_numpy()
saved_labels = predictions_df["true_label"].to_numpy()

print("Test CSV labels match saved true labels:",
      bool(np.array_equal(csv_labels, saved_labels)))

print("First 10 test CSV labels:", csv_labels[:10].tolist())
print("First 10 saved true labels:", saved_labels[:10].tolist())
print("\nDiagnostic checks complete.")
