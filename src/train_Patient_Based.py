import os
import cv2
import csv
import gc
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from model_Patient_Based import (MedicalDataset, build_model, calculate_metrics, FocalBCELoss,
                   FocalBCEDiceLoss, DiceLoss, CombinedLoss, get_images_and_masks)

""" Global parameters """
H = 256
W = 256
NUMEPOCS = 5
BATCH_SIZE = 12

MODEL = "DeepLabV3Plus"     #Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus, Segformer
ENCODER_NAME = "resnet50"      # "resnet34", "efficientnet-b4", "mit_b2"

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    train_transform = A.Compose([
        A.Resize(H, W),
        A.HorizontalFlip(p=0.5),  # Anatomik olarak her zaman geçerli
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.4),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.OneOf([A.GaussNoise(std_range=(0.01, 0.05), p=1.0), A.GaussianBlur(blur_limit=3, p=1.0),], p=0.2),
        # A.RandomResizedCrop(size=(H, W), scale=(0.8, 1.0), p=0.5),
        A.CoarseDropout(num_holes_range=(1, 2), hole_height_range=(16, 24), hole_width_range=(16, 24), fill=0, fill_mask=0, p=0.15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Validation augmentations
    val_transform = A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Used: {device}")

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()

    scaler = GradScaler()
    torch.backends.cudnn.benchmark = True

    dataset_path = "Path to the local dataset"

    train_imgs, train_masks = get_images_and_masks(dataset_path, "train")
    val_imgs, val_masks = get_images_and_masks(dataset_path, "val")
    test_imgs, test_masks = get_images_and_masks(dataset_path, "test")

    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    train_set = MedicalDataset(train_imgs, train_masks, transform=train_transform)
    val_set   = MedicalDataset(val_imgs, val_masks, transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(MODEL, ENCODER_NAME)
    model.to(device)

    print("MODEL:", model.__class__.__name__)
    print("ENCODER:", ENCODER_NAME)

    # Loss and Optimizer

    # criterion = FocalBCELoss(alpha=0.85, gamma=1.5)
    criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.BCELoss()
    # criterion = DiceLoss()
    # criterion = DiceBCELoss(dice_weight=0.7, bce_weight=0.3)
    # criterion = FocalBCEDiceLoss(alpha=0.85, gamma=1.5)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=3e-4)  # 50
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=NUMEPOCS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)


    # Training log file
    train_log_file = str(NUMEPOCS) + "_Pancreas_training_log_" + MODEL + ".csv"
    with open(train_log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "Train Loss", "Val Loss", "Train Accuracy", "Val Accuracy", "Train IoU", "Val IoU"])

    # ------------Training---------------
    best_val_iou = 0.0
    patience = 4
    counter = 0

    start_time = time.time()

    for epoch in range(NUMEPOCS):

        model.train()
        train_loss = 0.0
        train_accuracy_list = []
        train_iou_list = []

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
                _, _, _, iou, _, acc = calculate_metrics(outputs[i], masks[i], use_logits=True)
                train_accuracy_list.append(acc)
                train_iou_list.append(iou)

        train_loss /= len(train_loader)
        train_accuracy = np.mean(train_accuracy_list)
        train_iou = np.mean(train_iou_list)


        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_accuracy_list = []
        val_iou_list = []

        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                val_loss += loss.item()

                for i in range(images.shape[0]):
                    _, _, _, iou, _, acc = calculate_metrics(outputs[i], masks[i], use_logits=True)
                    val_accuracy_list.append(acc)
                    val_iou_list.append(iou)

        val_loss /= len(val_loader)
        val_accuracy = np.mean(val_accuracy_list)
        val_iou = np.mean(val_iou_list)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{NUMEPOCS}, "f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}, "f"Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")

        with open(train_log_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, train_accuracy, val_accuracy, train_iou, val_iou])

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), f"{NUMEPOCS}_best_model_{MODEL}.pth")
            print(f"Best model saved! Val IoU: {val_iou:.4f}")
            counter = 0
        else:
            counter += 1
            if counter > patience:
                print("Early stop applied.")
                break

    print(f"Training finished in {(time.time()-start_time)/60:.2f} minutes")

    # Plot training curves
    log_df = pd.read_csv(train_log_file)
    # Loss Curve
    plt.figure(figsize=(8, 6))
    plt.plot(log_df["Epoch"], log_df["Train Loss"], label="Training Loss")
    plt.plot(log_df["Epoch"], log_df["Val Loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(NUMEPOCS) + "_loss_curve_" + MODEL + ".png")
    plt.show()

    # Accuracy Curve
    plt.figure(figsize=(8, 6))
    plt.plot(log_df["Epoch"], log_df["Train Accuracy"], label="Training Accuracy")
    plt.plot(log_df["Epoch"], log_df["Val Accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(NUMEPOCS) + "_accuracy_curve_" + MODEL + ".png")
    plt.show()

    # IoU Curve
    plt.figure(figsize=(8, 6))
    plt.plot(log_df["Epoch"], log_df["Train IoU"], label="Train IoU")
    plt.plot(log_df["Epoch"], log_df["Val IoU"], label="Val IoU")
    plt.title("IoU Curve")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(str(NUMEPOCS) + "_IoU_curve_" + MODEL + ".png")
    plt.show()