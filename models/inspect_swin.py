
from pathlib import Path
import sys

import torch
import timm
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "datasets"))

from gtsrb_dataset import GTSRBDataset


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    dataset = GTSRBDataset(
        PROJECT_ROOT / "data" / "GTSRB" / "Train.csv"
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(loader))
    images = images.to(device)

    model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=True,
        num_classes=0,
    ).to(device)

    model.eval()

    with torch.no_grad():
        features = model(images)

    print("Input shape:", images.shape)
    print("Feature shape:", features.shape)
    print("Labels:", labels.tolist())
    print("Feature device:", features.device)
    print("First 10 feature values:")
    print(features[0, :10].cpu())


if __name__ == "__main__":
    main()
