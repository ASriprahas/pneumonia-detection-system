"""
Standalone Model Training Script for Pneumonia Detection System
Directly connects to local Kaggle chest X-ray dataset at:
C:\\Kaggle\\Chest_Xray\\chest_xray

Usage examples:
    python train_models.py --model pneumonianet --epochs 5 --batch-size 32
    python train_models.py --model resnet50 --epochs 10 --batch-size 16
    python train_models.py --model vgg16 --epochs 10 --batch-size 16
    python train_models.py --model all --epochs 10
"""

import os
import sys
import argparse
import time
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights, ResNet50_Weights

# Add parent directory to sys.path to import custom architectures
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

from app.models.pneumonianet import PneumoniaNetArchitecture
from app.core.config import settings

# Default dataset path
DEFAULT_DATASET_DIR = r"C:\Kaggle\Chest_Xray\chest_xray"
WEIGHTS_DIR = current_dir / "app" / "weights"
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomRotation(12),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

eval_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])


def build_model(model_name: str, device: torch.device) -> nn.Module:
    model_name = model_name.lower().strip()
    print(f"\n--> Building architecture for '{model_name}'...")

    if model_name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        # Freeze initial layers for transfer learning
        for param in model.features[:20].parameters():
            param.requires_grad = False
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 2)
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Freeze initial residual stages
        for param in list(model.parameters())[:-15]:
            param.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 2)
        )
    elif model_name == "pneumonianet":
        model = PneumoniaNetArchitecture(num_classes=2)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device)


def get_checkpoint_path(model_name: str) -> Path:
    if model_name == "pneumonianet":
        return WEIGHTS_DIR / "pneumonianet_v1.pth"
    return WEIGHTS_DIR / f"{model_name}_chest_xray.pth"


def train_single_model(
    model_name: str,
    dataset_dir: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device
):
    print("=" * 70)
    print(f"Starting Training: {model_name.upper()} on device {device}")
    print(f"Dataset root: {dataset_dir}")
    print("=" * 70)

    dataset_path = Path(dataset_dir)
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    test_dir = dataset_path / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found at: {train_dir}")

    # Load datasets
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_transforms)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=eval_transforms)
    test_dataset = datasets.ImageFolder(str(test_dir), transform=eval_transforms) if test_dir.exists() else None

    print(f"Found {len(train_dataset)} training images across classes: {train_dataset.classes}")
    print(f"Found {len(val_dataset)} validation images")
    if test_dataset:
        print(f"Found {len(test_dataset)} test images")

    # Handle class imbalance (e.g. ~3875 Pneumonia vs ~1341 Normal)
    targets = [sample[1] for sample in train_dataset.samples]
    class_counts = Counter(targets)
    print(f"Class distribution in train: {dict(class_counts)}")
    
    class_weights = [len(targets) / class_counts[i] for i in range(len(class_counts))]
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2) if test_dataset else None

    model = build_model(model_name, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_acc = 0.0
    checkpoint_file = get_checkpoint_path(model_name)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels).item()
            total_train += labels.size(0)

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels).item()
                total_val += labels.size(0)

        val_epoch_loss = val_loss / total_val if total_val > 0 else 0.0
        val_epoch_acc = correct_val / total_val if total_val > 0 else 0.0
        scheduler.step(val_epoch_loss)

        elapsed = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) - "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc*100:.2f}%")

        # Save best model
        if val_epoch_acc >= best_val_acc:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), checkpoint_file)
            print(f"  --> Saved new best checkpoint to: {checkpoint_file.name} (Val Acc: {val_epoch_acc*100:.2f}%)")

    # Optional Evaluation on test set
    if test_loader and checkpoint_file.exists():
        print(f"\nEvaluating best {model_name} on independent Test Set...")
        model.load_state_dict(torch.load(checkpoint_file, map_location=device))
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                correct_test += torch.sum(preds == labels).item()
                total_test += labels.size(0)
        test_acc = correct_test / total_test if total_test > 0 else 0.0
        print(f"Final Test Accuracy for {model_name}: {test_acc*100:.2f}% ({correct_test}/{total_test})")


def main():
    parser = argparse.ArgumentParser(description="Train Pneumonia Detection Models on Chest X-Ray Dataset")
    parser.add_argument("--model", type=str, default="pneumonianet", choices=["vgg16", "resnet50", "pneumonianet", "all"],
                        help="Model to train: vgg16, resnet50, pneumonianet, or all")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR,
                        help=f"Path to Kaggle chest X-ray root (default: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch Version: {torch.__version__} | Selected Device: {device}")

    models_to_train = ["pneumonianet", "resnet50", "vgg16"] if args.model == "all" else [args.model]
    for m in models_to_train:
        train_single_model(
            model_name=m,
            dataset_dir=args.dataset_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device
        )


if __name__ == "__main__":
    main()
