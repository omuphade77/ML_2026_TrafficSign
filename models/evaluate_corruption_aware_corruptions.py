
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from imagecorruptions import corrupt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.gtsrb_dataset import GTSRBDataset
from datasets.corruption_aware_dataset import CORRUPTIONS
from models.corruption_aware_swin import CorruptionAwareSwin


TEST_CSV = PROJECT_ROOT / "data" / "GTSRB" / "Test.csv"
CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "corruption_aware_best.pth"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "corruption_aware" / "corruptions"

BATCH_SIZE = 16
NUM_WORKERS = 0
SEVERITIES = [1, 2, 3, 4, 5]


class CorruptedTestDataset(Dataset):
    def __init__(self, csv_path, corruption_name, severity):
        self.base_dataset = GTSRBDataset(csv_path=csv_path)
        self.corruption_name = corruption_name
        self.severity = severity

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # Get the original RGB image and label from the CSV.
        row = self.base_dataset.df.iloc[index]
        image_path = self.base_dataset.data_dir / str(row["Path"])

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        width, height = image.size
        if width < 32 or height < 32:
            image = image.resize(
                (max(width, 32), max(height, 32)),
                Image.Resampling.BILINEAR,
            )

        image_array = np.array(image)

        corrupted_array = corrupt(
            image_array,
            corruption_name=self.corruption_name,
            severity=self.severity,
        )

        corrupted_image = Image.fromarray(corrupted_array)

        # Use the exact same resize and normalization as clean evaluation.
        image_tensor = self.base_dataset.transform(corrupted_image)
        label = int(row["ClassId"])

        return image_tensor, label


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = RESULTS_DIR / "corruption_metrics.csv"

    results = []

    for corruption_name in CORRUPTIONS:
        for severity in SEVERITIES:
            dataset = CorruptedTestDataset(
                TEST_CSV,
                corruption_name,
                severity,
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=(device.type == "cuda"),
            )

            labels_all = []
            predictions_all = []

            with torch.inference_mode():
                for images, labels in tqdm(
                    loader,
                    desc=f"{corruption_name}, severity {severity}",
                    leave=False,
                ):
                    images = images.to(device, non_blocking=True)

                    outputs = model(images)
                    predictions = outputs["class_logits"].argmax(dim=1)

                    labels_all.extend(labels.tolist())
                    predictions_all.extend(predictions.cpu().tolist())

            accuracy = accuracy_score(labels_all, predictions_all)

            row = {
                "corruption": corruption_name,
                "severity": severity,
                "accuracy": accuracy,
                "num_images": len(dataset),
            }
            results.append(row)

            # Save progress after every experiment in case execution stops.
            pd.DataFrame(results).to_csv(output_csv, index=False)

            print(
                f"{corruption_name:20s} severity={severity} "
                f"accuracy={accuracy:.4%}"
            )

    results_df = pd.DataFrame(results)
    summary = (
        results_df.groupby("corruption", as_index=False)["accuracy"]
        .mean()
        .sort_values("accuracy")
    )

    summary_path = RESULTS_DIR / "corruption_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nOverall mean corrupted accuracy:",
          f"{results_df['accuracy'].mean():.4%}")
    print("\nLowest-accuracy corruption types:")
    print(summary.head(5).to_string(index=False))

    print("\nDetailed results:", output_csv)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
