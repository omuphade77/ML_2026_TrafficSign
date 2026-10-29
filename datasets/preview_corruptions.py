
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from imagecorruptions import corrupt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "GTSRB"
OUTPUT_DIR = PROJECT_ROOT / "results" / "corruption_previews"

TEST_CSV = DATA_DIR / "Test.csv"


def main():
    df = pd.read_csv(TEST_CSV)

    # Choose one test image for visual inspection.
    row = df.iloc[0]
    image_path = DATA_DIR / row["Path"]

    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save the original image.
    image.save(OUTPUT_DIR / "original.png")

    # Generate Gaussian noise at severities 1 through 5.
    for severity in range(1, 6):
        corrupted_array = corrupt(
            image_array,
            corruption_name="gaussian_noise",
            severity=severity,
        )

        output_path = OUTPUT_DIR / f"gaussian_noise_severity_{severity}.png"
        Image.fromarray(corrupted_array).save(output_path)

        print(f"Saved: {output_path}")

    print(f"\nOriginal class label: {int(row['ClassId'])}")
    print(f"Previews saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
