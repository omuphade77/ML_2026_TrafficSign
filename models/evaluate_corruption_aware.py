
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.gtsrb_dataset import GTSRBDataset
from models.corruption_aware_swin import CorruptionAwareSwin


CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "corruption_aware_best.pth"
)
TEST_CSV = PROJECT_ROOT / "data" / "GTSRB" / "Test.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "corruption_aware"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    # Your dataset returns (image, label), not a dictionary.
    dataset = GTSRBDataset(csv_path=TEST_CSV)

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = CorruptionAwareSwin(
        num_classes=43,
        num_corruptions=15,
        num_severities=5,
        pretrained=False,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            predictions = outputs["class_logits"].argmax(dim=1)

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.cpu().tolist())

    metrics = {
        "accuracy": accuracy_score(
            all_labels, all_predictions
        ),
        "macro_precision": precision_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = RESULTS_DIR / "clean_test_metrics.csv"
    pd.DataFrame([metrics]).to_csv(output_path, index=False)

    print("\nCorruption-aware Swin: clean official test results")
    print("Test images:", len(dataset))

    for name, value in metrics.items():
        print(f"{name}: {value:.4%}")

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
