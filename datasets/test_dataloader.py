
import torch
from torch.utils.data import DataLoader

from gtsrb_dataset import GTSRBDataset


def main():
    train_dataset = GTSRBDataset(
        r"data/GTSRB/Train.csv"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("First 16 labels:", labels.tolist())
    print("GPU available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        images = images.to("cuda")
        labels = labels.to("cuda")

        print("Images device:", images.device)
        print("Labels device:", labels.device)


if __name__ == "__main__":
    main()
