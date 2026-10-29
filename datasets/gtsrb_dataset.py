
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "GTSRB"


# Preprocessing expected by our Swin-Tiny model.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ),
])


class GTSRBDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.csv_path = Path(csv_path)
        self.data_dir = DATA_DIR
        self.df = pd.read_csv(self.csv_path)
        self.transform = transform or IMAGE_TRANSFORM

        required_columns = {"Path", "ClassId"}
        missing = required_columns - set(self.df.columns)

        if missing:
            raise ValueError(
                f"Missing CSV columns: {missing}"
            )

    def __len__(self):
        # Number of images in this dataset split.
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # Locate the image using its CSV path.
        image_path = self.data_dir / str(row["Path"])

        # Read image and ensure it has three RGB channels.
        image = Image.open(image_path).convert("RGB")

        # Convert image into a normalized tensor.
        if self.transform is not None:
            image = self.transform(image)

        # GTSRB class IDs are 0 through 42.
        label = int(row["ClassId"])
        label = torch.tensor(label, dtype=torch.long)

        return image, label


if __name__ == "__main__":
    train_dataset = GTSRBDataset(
        DATA_DIR / "Train.csv"
    )

    test_dataset = GTSRBDataset(
        DATA_DIR / "Test.csv"
    )

    print("Training images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    image, label = train_dataset[0]

    print("\nOne training sample:")
    print("Image tensor shape:", image.shape)
    print("Image tensor dtype:", image.dtype)
    print("Class label:", label.item())

    # Confirm that tensors contain valid numbers.
    print("All values finite:", torch.isfinite(image).all().item())
