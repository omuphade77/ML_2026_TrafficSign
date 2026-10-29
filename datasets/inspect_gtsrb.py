
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# GTSRB dataset directory
DATA_DIR = PROJECT_ROOT / "data" / "GTSRB"

TRAIN_CSV = DATA_DIR / "Train.csv"
TEST_CSV = DATA_DIR / "Test.csv"


def main():
    print("Dataset directory:", DATA_DIR)
    print("Directory exists:", DATA_DIR.exists())

    # Load CSV files
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    print("\nTRAINING DATA")
    print("Number of rows:", len(train_df))
    print("Columns:", train_df.columns.tolist())
    print(train_df.head())

    print("\nTEST DATA")
    print("Number of rows:", len(test_df))
    print("Columns:", test_df.columns.tolist())
    print(test_df.head())

    # Show class distribution
    print("\nNumber of training images:", len(train_df))
    print("Number of test images:", len(test_df))

    # Inspect the first training image
    row = train_df.iloc[0]

    print("\nFirst training record:")
    print(row.to_dict())

    # Find the image path using the CSV's Path column
    image_path = DATA_DIR / str(row["Path"])
    print("\nImage path:", image_path)
    print("Image exists:", image_path.exists())

    if not image_path.exists():
        raise FileNotFoundError(
            f"Could not find image: {image_path}"
        )

    # Load and display the image
    image = Image.open(image_path).convert("RGB")

    # GTSRB CSVs commonly use ClassId for the label.
    label_column = "ClassId"
    label = int(row[label_column])

    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    plt.title(f"Traffic sign class: {label}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
