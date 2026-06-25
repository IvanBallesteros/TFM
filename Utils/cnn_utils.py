import importlib
import subprocess
import sys
import json
import random
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
import PIL
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")


def ensure_package(package_name, import_name=None):
    module_name = import_name or package_name
    try:
        importlib.import_module(module_name)
        print(f"OK: {package_name} already installed")
    except ModuleNotFoundError:
        print(f"Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


# Build a recursive image index using only the top-level folder as the class label.

def collect_class_samples(root_dir, image_extensions):
    root_path = Path(root_dir)
    class_names = sorted([item.name for item in root_path.iterdir() if item.is_dir()])
    samples = []
    image_counts = {}

    for class_idx, class_name in enumerate(class_names):
        class_dir = root_path / class_name
        class_images = sorted(
            path for path in class_dir.rglob('*')
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        image_counts[class_name] = len(class_images)
        samples.extend((str(path), class_idx) for path in class_images)

    return class_names, samples, image_counts


class PathImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, minority_labels=None, minority_transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform
        self.minority_labels = set(minority_labels or [])
        self.minority_transform = minority_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        import PIL
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = PIL.Image.open(image_path).convert('RGB')

        if self.minority_transform is not None and label in self.minority_labels:
            image = self.minority_transform(image)
        elif self.transform is not None:
            image = self.transform(image)

        return image, label
    
# Define a custom CNN architecture trained from scratch
class CustomCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 -> 112
            nn.Dropout(0.25),
        )
        
        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 -> 56
            nn.Dropout(0.25),
        )
        
        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 -> 28
            nn.Dropout(0.25),
        )
        
        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 -> 14
            nn.Dropout(0.25),
        )
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_model_pipeline(
    model_dir,
    class_names_path,
    model_candidates,
    model_class,
    device
):
    model_dir = Path(model_dir)
    class_names_path = Path(class_names_path)

    # -------------------------
    # Validate class names
    # -------------------------
    if not class_names_path.exists():
        raise FileNotFoundError(
            f"class_names.json not found at: {class_names_path}"
        )

    # -------------------------
    # Resolve model path
    # -------------------------
    resolved_model_path = None

    for name in model_candidates:
        candidate = model_dir / name
        if candidate.exists():
            resolved_model_path = candidate
            break

    if resolved_model_path is None:
        raise FileNotFoundError(
            f"No model checkpoint found in {model_dir}. Checked: {model_candidates}"
        )

    # -------------------------
    # Load class names
    # -------------------------
    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # -------------------------
    # Load model
    # -------------------------
    model = model_class(num_classes=len(class_names)).to(device)

    state_dict = torch.load(resolved_model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()

    return model, class_names, resolved_model_path


def predict_image_proba(
    image_path,
    model,
    class_names,
    inference_transform,
    device
):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at: {image_path}")

    # Load image
    image = Image.open(image_path).convert("RGB")

    # Preprocess
    input_tensor = inference_transform(image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    # Build output
    results = [
        {
            "class": class_names[i],
            "confidence": float(probs[i].item())
        }
        for i in range(len(class_names))
    ]

    # Sort by confidence (optional but useful)
    results = sorted(results, key=lambda x: x["confidence"], reverse=True)

    return results

from torchvision import transforms

def safe_mobile_border_crop(img, border_crop):
    crop_margin = min(border_crop, max((img.height - 1) // 2, 0))
    return img.crop((0, crop_margin, img.width, img.height - crop_margin))

def get_transforms(image_size, border_crop):
    
    base_transform = transforms.Compose([
        transforms.Lambda(lambda img: safe_mobile_border_crop(img, border_crop)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.10,
                contrast=0.10,
                saturation=0.05,
                hue=0.02,
            ),
        ], p=0.35),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                shear=3,
                fill=(255, 255, 255),
            ),
        ], p=0.35),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    minority_transform = transforms.Compose([
        transforms.Lambda(lambda img: safe_mobile_border_crop(img, border_crop)),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.08,
                hue=0.03,
            ),
        ], p=0.5),
        transforms.RandomApply([
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                shear=4,
                fill=(255, 255, 255),
            ),
        ], p=0.5),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(lambda img: safe_mobile_border_crop(img, border_crop)),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return base_transform, minority_transform, val_transform