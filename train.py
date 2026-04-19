"""
train.py — LeafScan AI Training Script
========================================
Trains one model at a time using Transfer Learning.
Supports: EfficientNetB0, MobileNetV2, ResNet50

Usage (Google Colab or local):
    # Train Stage 2 (dryness) for 10 epochs
    python train.py --stage dryness --epochs 10 --data_dir data/dryness

    # Train Stage 3 (disease) for 20 epochs with ResNet50
    python train.py --stage disease --model resnet50 --epochs 20 --data_dir data/disease

Dataset folder structure expected:
    data/
    └── dryness/
        ├── train/
        │   ├── healthy/       ← subfolder name = class label
        │   ├── mild_stress/
        │   └── severe_stress/
        └── val/
            ├── healthy/
            ├── mild_stress/
            └── severe_stress/
"""

import argparse
import os
import time
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# ── Argument Parsing ───────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser(description="LeafScan AI — Training Script")
    parser.add_argument("--stage",     type=str, default="disease",
                        choices=["quality", "dryness", "disease", "pest"],
                        help="Which stage to train")
    parser.add_argument("--model",     type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "mobilenet_v2", "resnet50"],
                        help="Model architecture to use")
    parser.add_argument("--data_dir",  type=str, default="data/disease",
                        help="Path to dataset folder (with train/ and val/)")
    parser.add_argument("--output_dir",type=str, default="models",
                        help="Where to save trained model weights")
    parser.add_argument("--epochs",    type=int, default=15)
    parser.add_argument("--batch_size",type=int, default=32)
    parser.add_argument("--lr",        type=float, default=1e-4,
                        help="Learning rate for fine-tuning layers")
    parser.add_argument("--lr_head",   type=float, default=1e-3,
                        help="Learning rate for the new classification head")
    parser.add_argument("--freeze_epochs", type=int, default=5,
                        help="Freeze backbone for first N epochs, then unfreeze")
    parser.add_argument("--img_size",  type=int, default=224)
    parser.add_argument("--workers",   type=int, default=2,
                        help="DataLoader worker processes")
    return parser.parse_args()


# ── Data Transforms ────────────────────────────────────────────────────────
def get_transforms(img_size: int):
    """
    Training: aggressive augmentation to prevent overfitting on small datasets.
    Validation: only resize + normalize (no augmentation).
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),                  # Random crop
        transforms.RandomHorizontalFlip(p=0.5),           # Mirror leaves
        transforms.RandomVerticalFlip(p=0.3),             # Less common but valid
        transforms.RandomRotation(degrees=30),            # Rotation invariance
        transforms.ColorJitter(
            brightness=0.3,                               # Lighting variation
            contrast=0.3,
            saturation=0.3,
            hue=0.05,
        ),
        transforms.RandomGrayscale(p=0.05),               # Occasionally grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],       # ImageNet stats
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


# ── Dataset Loader ─────────────────────────────────────────────────────────
def load_datasets(data_dir: str, train_tf, val_tf, batch_size: int, workers: int):
    """
    Loads train and val datasets using ImageFolder convention.
    Each subfolder name becomes a class label automatically.
    """
    train_path = os.path.join(data_dir, "train")
    val_path   = os.path.join(data_dir, "val")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training data not found at: {train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation data not found at: {val_path}")

    train_dataset = datasets.ImageFolder(train_path, transform=train_tf)
    val_dataset   = datasets.ImageFolder(val_path,   transform=val_tf)

    print(f"\n📂 Dataset loaded from: {data_dir}")
    print(f"   Classes ({len(train_dataset.classes)}): {train_dataset.classes}")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val   samples: {len(val_dataset)}\n")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes


# ── Model Builder ──────────────────────────────────────────────────────────
def build_model(architecture: str, num_classes: int):
    """
    Loads a pretrained model and replaces the final layer.
    Transfer learning strategy:
      - Keep pretrained ImageNet features
      - Replace classifier head with our task-specific linear layer
    """
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)
        head_params = model.classifier.parameters()
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]

    elif architecture == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)
        head_params = model.classifier.parameters()
        backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]

    elif architecture == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
        head_params = model.fc.parameters()
        backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]

    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    return model, head_params, backbone_params


def freeze_backbone(model, architecture: str):
    """Freeze all layers except the classification head."""
    for name, param in model.named_parameters():
        if architecture in ("efficientnet_b0", "mobilenet_v2") and "classifier" in name:
            param.requires_grad = True
        elif architecture == "resnet50" and "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def unfreeze_backbone(model):
    """Unfreeze all layers for full fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True


# ── Training Loop ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return total_loss / total, correct / total, all_preds, all_labels


# ── Main Training Function ─────────────────────────────────────────────────
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────
    train_tf, val_tf = get_transforms(args.img_size)
    train_loader, val_loader, class_names = load_datasets(
        args.data_dir, train_tf, val_tf, args.batch_size, args.workers
    )
    num_classes = len(class_names)

    # ── Model ─────────────────────────────────────────────────────────────
    model, head_params, backbone_params = build_model(args.model, num_classes)
    model = model.to(device)

    # Start with backbone frozen
    freeze_backbone(model, args.model)
    print(f"🧊 Backbone frozen for first {args.freeze_epochs} epochs.")

    # Two-group optimizer: head gets higher LR, backbone gets lower LR
    optimizer = optim.Adam([
        {"params": head_params,     "lr": args.lr_head},
        {"params": backbone_params, "lr": args.lr},
    ])

    # LR scheduler: reduce on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Class-weighted loss for imbalanced datasets
    criterion = nn.CrossEntropyLoss()

    # ── Training Loop ──────────────────────────────────────────────────────
    best_val_acc   = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n🚀 Training Stage: {args.stage} | Model: {args.model} | Epochs: {args.epochs}\n")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7}")
    print("-" * 55)

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after freeze_epochs
        if epoch == args.freeze_epochs + 1:
            unfreeze_backbone(model)
            print(f"\n🔓 Epoch {epoch}: Backbone unfrozen for full fine-tuning.\n")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
              f"{val_loss:>8.4f} | {val_acc:>6.2%}  ({elapsed:.0f}s)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc   = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    # ── Save Best Weights ──────────────────────────────────────────────────
    save_path = os.path.join(args.output_dir, f"stage{_stage_num(args.stage)}_{args.stage}.pth")
    torch.save(best_model_wts, save_path)
    print(f"\n✅ Best model saved to: {save_path}  (Val Acc: {best_val_acc:.2%})")

    # ── Final Classification Report ────────────────────────────────────────
    model.load_state_dict(best_model_wts)
    _, _, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
    print("\n📊 Classification Report (Validation Set):")
    print(classification_report(val_labels, val_preds, target_names=class_names))


def _stage_num(stage_name: str) -> int:
    return {"quality": 1, "dryness": 2, "disease": 3, "pest": 4}[stage_name]


# ── Google Colab Helper ────────────────────────────────────────────────────
def train_in_colab(
    stage: str = "disease",
    data_dir: str = "data/disease",
    epochs: int = 15,
    model: str = "efficientnet_b0",
):
    """
    Call this function directly in a Colab notebook instead of the CLI.
    Example:
        from train import train_in_colab
        train_in_colab(stage="disease", data_dir="/content/plantvillage", epochs=20)
    """
    import sys
    sys.argv = [
        "train.py",
        "--stage", stage,
        "--data_dir", data_dir,
        "--epochs", str(epochs),
        "--model", model,
    ]
    main()


if __name__ == "__main__":
    main()
