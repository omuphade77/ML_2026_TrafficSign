import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.corruption_aware_dataset import CorruptionAwareDataset
from models.corruption_aware_swin import CorruptionAwareSwin


# ---------------- Configuration ----------------
SEED = 42
BATCH_SIZE = 8
EPOCHS = 5
NUM_WORKERS = 0

BACKBONE_LR = 1e-5
HEADS_LR = 1e-4
WEIGHT_DECAY = 0.01

LAMBDA_CORRUPTION = 0.5
LAMBDA_SEVERITY = 0.5

# Smoke test: process only a few batches to check that everything works.
# Set both values to None for full training.
MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None

CHECKPOINT_IN = PROJECT_ROOT / "checkpoints" / "swin_finetuned_best.pth"
CHECKPOINT_OUT = PROJECT_ROOT / "checkpoints" / "corruption_aware_best.pth"
RESULTS_DIR = PROJECT_ROOT / "results" / "corruption_aware"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_losses(outputs, class_labels, corruption_labels,
                     severity_labels, class_criterion,
                     corruption_criterion, severity_criterion, device):
    sign_loss = class_criterion(
        outputs["class_logits"], class_labels
    )

    corruption_loss = corruption_criterion(
        outputs["corruption_logits"], corruption_labels
    )

    # Clean samples have severity -1, so exclude them from severity loss.
    corrupted_mask = severity_labels >= 0

    if corrupted_mask.any():
        severity_loss = severity_criterion(
            outputs["severity_logits"][corrupted_mask],
            severity_labels[corrupted_mask],
        )
    else:
        severity_loss = torch.zeros((), device=device)

    total_loss = (
        sign_loss
        + LAMBDA_CORRUPTION * corruption_loss
        + LAMBDA_SEVERITY * severity_loss
    )

    return total_loss


def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_OUT.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Use separate dataset instances:
    # training augmentation changes; validation augmentation stays fixed.
    train_dataset = CorruptionAwareDataset(
        seed=SEED,
        clean_probability=0.2,
        training=True,
    )

    val_dataset = CorruptionAwareDataset(
        seed=SEED + 1000,
        clean_probability=0.2,
        training=False,
    )

    # Both datasets have the same image order and labels.
    indices = np.arange(len(train_dataset))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.10,
        random_state=SEED,
        shuffle=True,
        stratify=train_dataset.df["ClassId"].to_numpy(),
    )

    train_ds = Subset(train_dataset, train_indices)
    val_ds = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Training images: {len(train_ds)}")
    print(f"Validation images: {len(val_ds)}")
    print(f"Train batches per epoch: {len(train_loader)}")
    print(f"Validation batches per epoch: {len(val_loader)}")

    model = CorruptionAwareSwin(
        num_classes=43,
        num_corruptions=15,
        num_severities=5,
        pretrained=False,
    )

    # Load the fine-tuned Swin checkpoint into compatible layers.
    # New condition heads and router are initialized for this task.
    old_checkpoint = torch.load(
        CHECKPOINT_IN,
        map_location="cpu",
        weights_only=True,
    )

    old_state = old_checkpoint["model_state_dict"]

    missing, unexpected = model.load_state_dict(
        old_state,
        strict=False,
    )

    print("Newly initialized parameters:", missing)

    if unexpected:
        print("Unused checkpoint keys:", unexpected)

    model = model.to(device)

    backbone_params = list(model.backbone.parameters())
    other_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.")
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": BACKBONE_LR},
            {"params": other_params, "lr": HEADS_LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    class_criterion = nn.CrossEntropyLoss()
    corruption_criterion = nn.CrossEntropyLoss()
    severity_criterion = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    history = []

    for epoch in range(EPOCHS):
        # ---------------- Training ----------------
        model.train()
        train_loss_sum = 0.0
        train_batches_seen = 0
        train_sign_preds = []
        train_sign_labels = []

        train_progress = tqdm(
            enumerate(train_loader),
            total=min(len(train_loader), MAX_TRAIN_BATCHES)
            if MAX_TRAIN_BATCHES is not None else len(train_loader),
            desc=f"Epoch {epoch + 1}/{EPOCHS} train",
        )

        for batch_idx, batch in train_progress:
            if (
                MAX_TRAIN_BATCHES is not None
                and batch_idx >= MAX_TRAIN_BATCHES
            ):
                break

            images = batch["image"].to(device)
            class_labels = batch["class_label"].to(device)
            corruption_labels = batch["corruption_label"].to(device)
            severity_labels = batch["severity_label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)

            loss = calculate_losses(
                outputs,
                class_labels,
                corruption_labels,
                severity_labels,
                class_criterion,
                corruption_criterion,
                severity_criterion,
                device,
            )

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches_seen += 1

            train_sign_preds.extend(
                outputs["class_logits"].argmax(dim=1).detach().cpu().tolist()
            )
            train_sign_labels.extend(class_labels.detach().cpu().tolist())

        if train_batches_seen == 0:
            raise RuntimeError("No training batches were processed.")

        # ---------------- Validation ----------------
        model.eval()
        val_loss_sum = 0.0
        val_batches_seen = 0
        val_sign_preds = []
        val_sign_labels = []

        with torch.inference_mode():
            val_progress = tqdm(
                enumerate(val_loader),
                total=min(len(val_loader), MAX_VAL_BATCHES)
                if MAX_VAL_BATCHES is not None else len(val_loader),
                desc=f"Epoch {epoch + 1}/{EPOCHS} val",
            )

            for batch_idx, batch in val_progress:
                if (
                    MAX_VAL_BATCHES is not None
                    and batch_idx >= MAX_VAL_BATCHES
                ):
                    break

                images = batch["image"].to(device)
                class_labels = batch["class_label"].to(device)
                corruption_labels = batch["corruption_label"].to(device)
                severity_labels = batch["severity_label"].to(device)

                outputs = model(images)

                loss = calculate_losses(
                    outputs,
                    class_labels,
                    corruption_labels,
                    severity_labels,
                    class_criterion,
                    corruption_criterion,
                    severity_criterion,
                    device,
                )

                val_loss_sum += loss.item()
                val_batches_seen += 1

                val_sign_preds.extend(
                    outputs["class_logits"].argmax(dim=1).cpu().tolist()
                )
                val_sign_labels.extend(class_labels.cpu().tolist())

        if val_batches_seen == 0:
            raise RuntimeError("No validation batches were processed.")

        train_acc = accuracy_score(train_sign_labels, train_sign_preds)
        train_f1 = f1_score(
            train_sign_labels,
            train_sign_preds,
            average="macro",
            zero_division=0,
        )

        val_acc = accuracy_score(val_sign_labels, val_sign_preds)
        val_f1 = f1_score(
            val_sign_labels,
            val_sign_preds,
            average="macro",
            zero_division=0,
        )

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss_sum / train_batches_seen,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss_sum / val_batches_seen,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "train_batches": train_batches_seen,
            "val_batches": val_batches_seen,
        }

        history.append(row)
        print(row)

        history_path = RESULTS_DIR / "training_history.csv"
        pd.DataFrame(history).to_csv(history_path, index=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_f1,
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "epochs": EPOCHS,
                        "backbone_lr": BACKBONE_LR,
                        "heads_lr": HEADS_LR,
                        "lambda_corruption": LAMBDA_CORRUPTION,
                        "lambda_severity": LAMBDA_SEVERITY,
                        "clean_probability": 0.2,
                        "seed": SEED,
                    },
                },
                CHECKPOINT_OUT,
            )

            print(f"Saved best checkpoint: {CHECKPOINT_OUT}")

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print("Training history:", RESULTS_DIR / "training_history.csv")


if __name__ == "__main__":
    main()