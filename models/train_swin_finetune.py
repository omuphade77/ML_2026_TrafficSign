
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
BATCH_SIZE = 8
EPOCHS = 5

BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "swin_finetuned_best.pth"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------------------------------------------
# 2. Load the same training and validation split
# --------------------------------------------------

dataset = GTSRBDataset("data/GTSRB/Train.csv")

validation_size = int(len(dataset) * 0.10)
training_size = len(dataset) - validation_size

train_dataset, val_dataset = random_split(
    dataset,
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
# 3. Load pretrained Swin and enable fine-tuning
# --------------------------------------------------

model = SwinBaseline(
    num_classes=43,
    pretrained=True,
).to(device)

# Unlike the frozen baseline, all model parameters can now learn.
for parameter in model.parameters():
    parameter.requires_grad = True

criterion = nn.CrossEntropyLoss()

# Use a smaller learning rate for pretrained Swin.
optimizer = torch.optim.AdamW(
    [
        {
            "params": model.backbone.parameters(),
            "lr": BACKBONE_LR,
        },
        {
            "params": model.classifier.parameters(),
            "lr": CLASSIFIER_LR,
        },
    ],
    weight_decay=0.01,
)

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)


# --------------------------------------------------
# 4. Train and validate
# --------------------------------------------------

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    train_loss_sum = 0.0
    train_correct = 0
    train_total = 0

    progress = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]",
    )

    for images, labels in progress:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_count = labels.size(0)
        train_loss_sum += loss.item() * batch_count
        train_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        train_total += batch_count

        progress.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = train_loss_sum / train_total
    train_accuracy = train_correct / train_total

    # Validation: no parameter updates.
    model.eval()

    val_loss_sum = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(
            val_loader,
            desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]",
        ):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_count = labels.size(0)
            val_loss_sum += loss.item() * batch_count
            val_correct += (
                logits.argmax(dim=1) == labels
            ).sum().item()
            val_total += batch_count

    val_loss = val_loss_sum / val_total
    val_accuracy = val_correct / val_total

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print(f"Train loss:     {train_loss:.4f}")
    print(f"Train accuracy: {train_accuracy:.4%}")
    print(f"Val loss:       {val_loss:.4f}")
    print(f"Val accuracy:   {val_accuracy:.4%}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "seed": SEED,
                "backbone_lr": BACKBONE_LR,
                "classifier_lr": CLASSIFIER_LR,
            },
            CHECKPOINT_PATH,
        )

        print("Saved best fine-tuned model:", CHECKPOINT_PATH)

print("\nFine-tuning complete.")
print(f"Best validation accuracy: {best_val_accuracy:.4%}")
print("Best checkpoint:", CHECKPOINT_PATH)
