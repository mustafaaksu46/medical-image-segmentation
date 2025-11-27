import os
import cv2
import csv
import gc
import torch
import time
import random
import pandas as pd
import numpy as np
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast

from model import create_model, FocalBCELoss, CombinedLoss

""" Global parameters """
H = 256
W = 256
NUMEPOCS = 50
THRESHOLD = 0.40
MODEL = "DeepLabV3Plus"

class MedicalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask'].unsqueeze(0)

        return image, mask, self.image_paths[idx]

def calculate_metrics(pred, mask, threshold=THRESHOLD):
    smooth = 1e-6
    pred = (torch.sigmoid(pred) > threshold).float()
    TP = torch.sum(pred * mask).item()
    FP = torch.sum(pred * (1 - mask)).item()
    FN = torch.sum((1 - pred) * mask).item()
    TN = torch.sum((1 - pred) * (1 - mask)).item()

    precision = TP / (TP + FP + smooth)
    recall = TP / (TP + FN + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    iou = TP / (TP + FP + FN + smooth)
    specificity = TN / (TN + FP + smooth)
    accuracy = (TP + TN) / (TP + TN + FP + FN + smooth)

    return precision, recall, f1, iou, specificity, accuracy

def get_images_and_masks(root_dir):
    image_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if "mask" not in f])
    mask_paths = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if "mask" in f])
    combined = list(zip(image_paths, mask_paths))
    random.shuffle(combined)
    return zip(*combined)

def train_model(dataset_path, model_name=MODEL, num_epochs=NUMEPOCS, batch_size=12,
                learning_rate=2e-5, weight_decay=1e-4, patience=6):

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Used: {device}")

    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()
        print("CUDA cache cleared")

    # Create log files
    train_log_file = f"{num_epochs}_Pancreas_training_log_{model_name}.csv"
    with open(train_log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "Train Loss", "Val Loss", "Train Accuracy", "Val Accuracy"])

    # Load dataset
    start_time = time.time()
    image_paths, mask_paths = get_images_and_masks(dataset_path)

    # Define augmentations
    train_transform = A.Compose([
        A.Resize(H, W),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.05, 0.05), rotate=(-7, 7), shear=(0, 0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(p=0.2),
        A.RandomCrop(height=int(H * 0.9), width=int(W * 0.9), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Create datasets
    dataset = MedicalDataset(image_paths, mask_paths, transform=train_transform)
    train_len = int(0.80 * len(dataset))
    val_len = int(0.10 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    print(f"Dataset split - Train: {train_len}, Val: {val_len}, Test: {test_len}, Total: {len(dataset)}")

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    val_set.dataset.transform = val_transform
    test_set.dataset.transform = val_transform

    # Create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=batch_size, pin_memory=True)

    # Create model
    model = create_model(model_name=model_name, encoder_name="resnet50", use_dropblock=True,
                         block_size=7, drop_prob=0.10)
    model.to(device)

    # Setup training
    criterion = FocalBCELoss(alpha=0.90, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    scaler = GradScaler()

    torch.backends.cudnn.benchmark = True

    # Training loop
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        for images, masks, _ in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

            for i in range(images.shape[0]):
                _, _, _, _, _, acc = calculate_metrics(outputs[i], masks[i])
                train_accuracy += acc

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss += loss.item()

                for i in range(images.shape[0]):
                    _, _, _, _, _, acc = calculate_metrics(outputs[i], masks[i])
                    val_accuracy += acc

        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader.dataset)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        # Log results
        with open(train_log_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, train_accuracy, val_accuracy])

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{num_epochs}_best_model_{model_name}.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping applied")
                break

    # Plot training curves
    log_df = pd.read_csv(train_log_file)

    # Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(log_df["Epoch"], log_df["Train Loss"], label="Training Loss")
    plt.plot(log_df["Epoch"], log_df["Val Loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{num_epochs}_loss_curve_{model_name}.png")
    plt.show()

    # Accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(log_df["Epoch"], log_df["Train Accuracy"], label="Training Accuracy")
    plt.plot(log_df["Epoch"], log_df["Val Accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{num_epochs}_accuracy_curve_{model_name}.png")
    plt.show()

    end_time = time.time()
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    return model, device, train_log_file


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.freeze_support()

    # “Path to the prepared dataset”
    dataset_path = "   Local dataset path   "  

    model, device, log_file = train_model(
        dataset_path=dataset_path,
        model_name=MODEL,
        num_epochs=NUMEPOCS,
        batch_size=12,
        learning_rate=2e-5,
        weight_decay=1e-4,
        patience=6
    )


    print("Training completed!")

