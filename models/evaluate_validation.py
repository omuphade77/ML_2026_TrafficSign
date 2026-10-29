
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score

from datasets.gtsrb_dataset import GTSRBDataset
from models.swin_baseline import SwinBaseline


SEED = 42
BATCH_SIZE = 16
CHECKPOINT_PATH = (
    PROJECT_ROOT / "checkpoints" / "swin_baseline_best.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Recreate the exact split used by the training script.
dataset = GTSRBDataset("data/GTSRB/Train.csv")

validation_size = int(len(dataset) * 0.10)
training_size = len(dataset) - validation_size

_, val_dataset = random_split(
    dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(SEED),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

model = SwinBaseline(num_classes=43, pretrained=False).to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

all_true = []
all_pred = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        predictions = model(images).argmax(dim=1)

        all_true.extend(labels.numpy().tolist())
        all_pred.extend(predictions.cpu().numpy().tolist())

accuracy = accuracy_score(all_true, all_pred)
macro_f1 = f1_score(
    all_true,
    all_pred,
    labels=list(range(43)),
    average="macro",
    zero_division=0,
)

print("Validation images:", len(val_dataset))
print(f"Reproduced validation accuracy: {accuracy:.4%}")
print(f"Validation macro F1: {macro_f1:.4%}")
print("Checkpoint validation accuracy:", f"{checkpoint['val_accuracy']:.4%}")
