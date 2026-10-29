import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from imagecorruptions import corrupt
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "GTSRB"

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

# 0 means clean; 1–14 represent the corruption types above.
CORRUPTION_TO_ID = {
    name: index + 1 for index, name in enumerate(CORRUPTIONS)
}

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


class CorruptionAwareDataset(Dataset):
    def __init__(
        self,
        csv_path=None,
        clean_probability=0.2,
        seed=42,
        training=True,
    ):
        if csv_path is None:
            csv_path = DATA_DIR / "Train.csv"

        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(self.csv_path).reset_index(drop=True)

        self.clean_probability = clean_probability
        self.seed = seed
        self.training = training

        if not 0.0 <= clean_probability <= 1.0:
            raise ValueError("clean_probability must be between 0 and 1.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Training gets fresh random corruption choices.
        # Validation gets the same choices for each index on every pass.
        if self.training:
            rng = random.Random()
        else:
            rng = random.Random(self.seed + int(index))

        row = self.df.iloc[index]
        image_path = DATA_DIR / str(row["Path"])

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        # imagecorruptions requires sufficiently large input images.
        width, height = image.size
        if width < 32 or height < 32:
            image = image.resize(
                (max(width, 32), max(height, 32)),
                Image.Resampling.BILINEAR,
            )

        # Label 0 = clean; severity -1 = clean image, so ignore its loss.
        corruption_label = 0
        severity_label = -1

        if rng.random() >= self.clean_probability:
            corruption_name = rng.choice(CORRUPTIONS)
            severity = rng.randint(1, 5)

            image_array = np.array(image)
            corrupted_array = corrupt(
                image_array,
                corruption_name=corruption_name,
                severity=severity,
            )

            image = Image.fromarray(corrupted_array)

            corruption_label = CORRUPTION_TO_ID[corruption_name]
            severity_label = severity - 1  # Severity labels: 0–4.

        image_tensor = IMAGE_TRANSFORM(image)

        return {
            "image": image_tensor,
            "class_label": int(row["ClassId"]),
            "corruption_label": corruption_label,
            "severity_label": severity_label,
        }