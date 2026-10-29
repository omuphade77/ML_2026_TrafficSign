
import sys
from pathlib import Path

# Make the project root available for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from datasets.gtsrb_dataset import GTSRBDataset
from models.swin_baseline import SwinBaseline


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BATCH_SIZE = 16

CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "swin_baseline_best.pth"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# --------------------------------------------------
# 2. Load the official test dataset
# --------------------------------------------------

test_dataset = GTSRBDataset("data/GTSRB/Test.csv")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print("Test images:", len(test_dataset))


# --------------------------------------------------
# 3. Load the trained model
# --------------------------------------------------

model = SwinBaseline(
    num_classes=43,
    pretrained=False,
).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch:", checkpoint["epoch"])
print(
    "Best recorded validation accuracy:",
    f"{checkpoint['val_accuracy']:.4%}",
)


# --------------------------------------------------
# 4. Predict test labels
# --------------------------------------------------

all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        logits = model(images)
        predictions = logits.argmax(dim=1)

        all_predictions.extend(
            predictions.cpu().numpy()
        )
        all_labels.extend(labels.numpy())

y_true = np.asarray(all_labels)
y_pred = np.asarray(all_predictions)


# --------------------------------------------------
# 5. Calculate metrics
# --------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

precision_macro = precision_score(
    y_true,
    y_pred,
    labels=list(range(43)),
    average="macro",
    zero_division=0,
)

recall_macro = recall_score(
    y_true,
    y_pred,
    labels=list(range(43)),
    average="macro",
    zero_division=0,
)

f1_macro = f1_score(
    y_true,
    y_pred,
    labels=list(range(43)),
    average="macro",
    zero_division=0,
)

print("\n========== TEST RESULTS ==========")
print(f"Test accuracy:          {accuracy:.4%}")
print(f"Macro precision:        {precision_macro:.4%}")
print(f"Macro recall:           {recall_macro:.4%}")
print(f"Macro F1-score:          {f1_macro:.4%}")


# --------------------------------------------------
# 6. Per-class report and confusion matrix
# --------------------------------------------------

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(43)),
    digits=4,
    zero_division=0,
)

print("\n========== PER-CLASS REPORT ==========")
print(report)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(43)),
)

np.savetxt(
    RESULTS_DIR / "baseline_confusion_matrix.csv",
    matrix,
    delimiter=",",
    fmt="%d",
)

np.savetxt(
    RESULTS_DIR / "baseline_predictions.csv",
    np.column_stack((y_true, y_pred)),
    delimiter=",",
    fmt="%d",
    header="true_label,predicted_label",
    comments="",
)

with open(
    RESULTS_DIR / "baseline_metrics.txt",
    "w",
    encoding="utf-8",
) as file:
    file.write(f"Test accuracy: {accuracy:.6f}\n")
    file.write(f"Macro precision: {precision_macro:.6f}\n")
    file.write(f"Macro recall: {recall_macro:.6f}\n")
    file.write(f"Macro F1-score: {f1_macro:.6f}\n\n")
    file.write(report)

print("\nSaved results to:", RESULTS_DIR)
print("Evaluation complete.")
