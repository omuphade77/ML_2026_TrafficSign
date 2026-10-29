
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from imagecorruptions import corrupt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets.gtsrb_dataset import GTSRBDataset
from datasets.corruption_aware_dataset import CORRUPTIONS
from models.corruption_aware_swin import CorruptionAwareSwin
from models.swin_baseline import SwinBaseline


TEST_CSV = ROOT / "data" / "GTSRB" / "Test.csv"
BASELINE_CHECKPOINT = ROOT / "checkpoints" / "swin_finetuned_best.pth"
PROPOSED_CHECKPOINT = ROOT / "checkpoints" / "corruption_aware_best.pth"
OUTPUT_DIR = ROOT / "results" / "controlled_comparison"

SEVERITIES = [1, 2, 3, 4, 5]
BATCH_SIZE = 16
CORRUPTION_SEED = 12345


class SharedCorruptedDataset(Dataset):
    def __init__(self, csv_path, corruption_name, severity):
        self.base = GTSRBDataset(csv_path=csv_path)
        self.corruption_name = corruption_name
        self.severity = severity

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.base.df.iloc[index]
        path = self.base.data_dir / str(row["Path"])

        with Image.open(path) as image_file:
            image = image_file.convert("RGB")

        width, height = image.size
        if width < 32 or height < 32:
            image = image.resize(
                (max(width, 32), max(height, 32)),
                Image.Resampling.BILINEAR,
            )

        # Deterministic per-image corruption for repeatable comparisons.
        seed = CORRUPTION_SEED + index * 100 + self.severity
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        array = np.array(image)
        corrupted = corrupt(
            array,
            corruption_name=self.corruption_name,
            severity=self.severity,
        )

        image = Image.fromarray(corrupted)
        tensor = self.base.transform(image)
        return tensor, int(row["ClassId"])


def load_model(model_class, checkpoint_path, device):
    model = model_class(
        num_classes=43,
        pretrained=False,
    ) if model_class is SwinBaseline else model_class(
        num_classes=43,
        num_corruptions=15,
        num_severities=5,
        pretrained=False,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model



def evaluate_pair(baseline, proposed, loader, device):
    baseline_labels, baseline_preds = [], []
    proposed_labels, proposed_preds = [], []

    with torch.inference_mode():
        for images, labels in loader:
            # Generate/load each batch only once.
            images = images.to(device)

            baseline_logits = baseline(images)
            proposed_outputs = proposed(images)

            if isinstance(proposed_outputs, dict):
                proposed_logits = proposed_outputs["class_logits"]
            else:
                proposed_logits = proposed_outputs

            baseline_predictions = baseline_logits.argmax(dim=1)
            proposed_predictions = proposed_logits.argmax(dim=1)

            baseline_labels.extend(labels.tolist())
            proposed_labels.extend(labels.tolist())
            baseline_preds.extend(baseline_predictions.cpu().tolist())
            proposed_preds.extend(proposed_predictions.cpu().tolist())

    return {
        "baseline_accuracy": accuracy_score(
            baseline_labels, baseline_preds
        ),
        "proposed_accuracy": accuracy_score(
            proposed_labels, proposed_preds
        ),
        "baseline_macro_f1": f1_score(
            baseline_labels, baseline_preds,
            average="macro", zero_division=0
        ),
        "proposed_macro_f1": f1_score(
            proposed_labels, proposed_preds,
            average="macro", zero_division=0
        ),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    baseline = load_model(
        SwinBaseline,
        BASELINE_CHECKPOINT,
        device,
    )
    proposed = load_model(
        CorruptionAwareSwin,
        PROPOSED_CHECKPOINT,
        device,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for corruption_name in CORRUPTIONS:
        for severity in SEVERITIES:
            dataset = SharedCorruptedDataset(
                TEST_CSV,
                corruption_name,
                severity,
            )
            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
            )

            metrics = evaluate_pair(baseline, proposed, loader, device)
            row = {
                "corruption": corruption_name,
                "severity": severity,
                **metrics,
                "accuracy_improvement_pp": (
                    metrics["proposed_accuracy"]
                    - metrics["baseline_accuracy"]
                ) * 100,
                "macro_f1_improvement_pp": (
                    metrics["proposed_macro_f1"]
                    - metrics["baseline_macro_f1"]
                ) * 100,
            }
            results.append(row)

            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "controlled_results.csv",
                index=False,
            )

            print(
                f"{corruption_name:20s} severity={severity} "
                f"baseline={metrics['baseline_accuracy']:.4%} "
                f"proposed={metrics['proposed_accuracy']:.4%} "
                f"gain={row['accuracy_improvement_pp']:+.2f} pp"
            )

    df = pd.DataFrame(results)
    print("\nControlled baseline mean accuracy:",
          f"{df['baseline_accuracy'].mean():.4%}")
    print("Controlled proposed mean accuracy:",
          f"{df['proposed_accuracy'].mean():.4%}")
    print("Mean improvement:",
          f"{df['accuracy_improvement_pp'].mean():+.2f} percentage points")
    print("Saved:", OUTPUT_DIR / "controlled_results.csv")


if __name__ == "__main__":
    main()
