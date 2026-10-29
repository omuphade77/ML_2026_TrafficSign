
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from datasets.gtsrb_dataset import GTSRBDataset
from models.swin_baseline import SwinBaseline


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
VALIDATION_FRACTION = 0.10

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------------------------------------------
# 2. Load the dataset and create a validation split
# --------------------------------------------------

full_dataset = GTSRBDataset("data/GTSRB/Train.csv")

validation_size = int(
    len(full_dataset) * VALIDATION_FRACTION
)
training_size = len(full_dataset) - validation_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [training_size, validation_size],
    generator=torch.Generator().manual_seed(SEED),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# --------------------------------------------------
# 3. Create the model and freeze the Swin backbone
# --------------------------------------------------

model = SwinBaseline(
    num_classes=43,
    pretrained=True,
).to(device)

for parameter in model.backbone.parameters():
    parameter.requires_grad = False

# Only classifier parameters will be updated.
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.classifier.parameters(),
    lr=LEARNING_RATE,
)

print(
    "Trainable parameters:",
    sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    ),
)


# --------------------------------------------------
# 4. Training and validation
# --------------------------------------------------

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    # Training phase
    model.train()
    model.backbone.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
    )

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.no_grad():
            features = model.backbone(images)

        logits = model.classifier(features)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    train_loss = running_loss / total
    train_accuracy = correct / total

    # Validation phase
    model.eval()

    val_correct = 0
    val_total = 0
    val_loss_sum = 0.0

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
        ):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            val_loss_sum += loss.item() * labels.size(0)
            predictions = logits.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train loss:      {train_loss:.4f}")
    print(f"Train accuracy:  {train_accuracy:.4%}")
    print(f"Val loss:        {val_loss:.4f}")
    print(f"Val accuracy:    {val_accuracy:.4%}")

    # Save the model with the best validation accuracy.
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        checkpoint_path = (
            CHECKPOINT_DIR / "swin_baseline_best.pth"
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "seed": SEED,
            },
            checkpoint_path,
        )

        print("Saved best model to:", checkpoint_path)

print("\nTraining complete.")
print(f"Best validation accuracy: {best_val_accuracy:.4%}")
