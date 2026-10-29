
from pyexpat import model
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from imagecorruptions import corrupt
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.swin_baseline import SwinBaseline

DATA_DIR = PROJECT_ROOT / "data" / "GTSRB"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "swin_finetuned_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results" / "corruption_evaluation"

BATCH_SIZE = 16
NUM_WORKERS = 0

CORRUPTIONS = [
    "gaussian_noise",
    "impulse_noise",
    "shot_noise",
    "snow",
    "fog",
    "frost",
    "motion_blur",
    "zoom_blur",
    "defocus_blur",
    "brightness",
    "contrast",
    "pixelate",
    "elastic_transform",
    "jpeg_compression",
]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


class CorruptedTestDataset(Dataset):
    def __init__(self, csv_path, corruption_name, severity):
        self.df = pd.read_csv(csv_path)
        self.corruption_name = corruption_name
        self.severity = severity

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = DATA_DIR / str(row["Path"])

        
        with Image.open(image_path) as img:
            image = img.convert("RGB")

            # Ensure dimensions satisfy imagecorruptions requirements.
            width, height = image.size
            if width < 32 or height < 32:
                new_width = max(width, 32)
                new_height = max(height, 32)
                image = image.resize(
                    (new_width, new_height),
                    Image.Resampling.BILINEAR,
                )

            image_array = np.array(image)


        # Apply corruption before resizing and normalizing.
        corrupted_array = corrupt(
            image_array,
            corruption_name=self.corruption_name,
            severity=self.severity,
        )

        corrupted_image = Image.fromarray(corrupted_array)
        tensor = IMAGE_TRANSFORM(corrupted_image)
        label = int(row["ClassId"])

        return tensor, label


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    test_csv = DATA_DIR / "Test.csv"

    model = SwinBaseline(num_classes=43, pretrained=False).to(device)
    
    checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    
    partial_path = RESULTS_DIR / "corruption_metrics_partial.csv"

    if partial_path.exists():
        previous_df = pd.read_csv(partial_path)
        results = previous_df.to_dict("records")
        completed = {
            (row["corruption"], int(row["severity"]))
            for row in results
        }
        print(f"Resuming: {len(completed)} experiments already completed.")
    else:
        results = []
        completed = set()


    # Each corruption/severity pair is evaluated separately.
    for corruption_name in CORRUPTIONS:
        for severity in range(1, 6):
            print(
                f"\nEvaluating {corruption_name}, "
                f"severity {severity}"
            )

            dataset = CorruptedTestDataset(
                csv_path=test_csv,
                corruption_name=corruption_name,
                severity=severity,
            )

            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=(device.type == "cuda"),
            )

            all_predictions = []
            all_labels = []

            with torch.inference_mode():
                for images, labels in tqdm(loader):
                    images = images.to(device)
                    logits = model(images)
                    predictions = logits.argmax(dim=1).cpu().numpy()

                    all_predictions.extend(predictions.tolist())
                    all_labels.extend(labels.numpy().tolist())

            accuracy = accuracy_score(all_labels, all_predictions)
            macro_f1 = f1_score(
                all_labels,
                all_predictions,
                average="macro",
                zero_division=0,
            )

            results.append({
                "corruption": corruption_name,
                "severity": severity,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "num_images": len(all_labels),
            })

            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(f"Macro F1: {macro_f1 * 100:.2f}%")

            # Save progress after every experiment.
            pd.DataFrame(results).to_csv(
                RESULTS_DIR / "corruption_metrics_partial.csv",
                index=False,
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        RESULTS_DIR / "corruption_metrics.csv",
        index=False,
    )

    print("\nAll corruption experiments completed.")
    print(f"Results saved to: {RESULTS_DIR / 'corruption_metrics.csv'}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
