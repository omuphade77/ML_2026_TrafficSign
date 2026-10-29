
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import random
import pandas as pd
import matplotlib.pyplot as plt

from datasets.gtsrb_dataset import GTSRBDataset


# Load the test dataset and saved predictions.
test_dataset = GTSRBDataset("data/GTSRB/Test.csv")

predictions_path = (
    PROJECT_ROOT / "results" / "baseline_predictions.csv"
)
predictions_df = pd.read_csv(predictions_path)

# Find indices where the prediction was incorrect.
errors = predictions_df.index[
    predictions_df["true_label"]
    != predictions_df["predicted_label"]
].tolist()

# Pick up to 12 examples to inspect.
random.seed(42)
selected = random.sample(errors, min(12, len(errors)))

# ImageNet normalization values used by our dataset class.
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

fig, axes = plt.subplots(3, 4, figsize=(14, 10))
axes = axes.flatten()

for ax, index in zip(axes, selected):
    image_tensor, dataset_label = test_dataset[index]

    # Convert normalized tensor back to displayable RGB values.
    image = image_tensor.permute(1, 2, 0).numpy()
    image = image * std + mean
    image = image.clip(0, 1)

    actual = int(predictions_df.iloc[index]["true_label"])
    predicted = int(predictions_df.iloc[index]["predicted_label"])

    ax.imshow(image)
    ax.set_title(f"Actual: {actual} | Predicted: {predicted}")
    ax.axis("off")

    # Check that the prediction CSV's label matches the dataset.
    if actual != int(dataset_label):
        print(
            f"WARNING: label mismatch at index {index}: "
            f"CSV={actual}, dataset={dataset_label}"
        )

for ax in axes[len(selected):]:
    ax.axis("off")

plt.tight_layout()
plt.show()
