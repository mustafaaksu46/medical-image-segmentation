import os
import cv2
import csv
import gc
import torch
import time
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import segmentation_models_pytorch as smp
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, random_split
from dropblock import DropBlock2D
import matplotlib.pyplot as plt
from torch.amp import GradScaler, autocast

""" Global parameters """
H = 256
W = 256

NUMEPOCS=50
THRESHOLD=0.40

MODEL="DeepLabV3Plus"     #  Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus, Segformer

class DropBlockDeepLab(nn.Module):
    def __init__(self, base_model, block_size=7, drop_prob=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

    def forward(self, x):
        x = self.base_model(x)
        if self.training:
            x = self.dropblock(x)
        return x
"""
class DropBlockUNetPP(nn.Module):
    def __init__(self, base_model, block_size=7, drop_prob=0.1):
        super().__init__()
        self.base_model = base_model
        self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

    def forward(self, x):
        x = self.base_model(x)
        if self.training:
            x = self.dropblock(x)
        return x
"""

"""
class DropBlockSegFormer(nn.Module):
    def __init__(self, base_model, block_size=7, drop_prob=0.2):
        super().__init__()
        self.base_model = base_model
        self.dropblock = DropBlock2D(drop_prob=drop_prob, block_size=block_size)

    def forward(self, x):
        x = self.base_model(x)
        if self.training:
            x = self.dropblock(x)
        return x
"""

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

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.90, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class CombinedLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, alpha=0.9, gamma=2.0):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalBCELoss(alpha=alpha, gamma=gamma)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        return self.dice_weight * dice + self.focal_weight * focal

"""
**************************************************************************
                          Main Program Block
**************************************************************************
"""

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    # Log and result Files
    train_log_file = str(NUMEPOCS) + "_Pancreas_training_log_" + MODEL + ".csv"
    test_log_file = str(NUMEPOCS) + "_Pancreas_test_results_" + MODEL + ".csv"

    with open(train_log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Epoch", "Train Loss", "Val Loss", "Train Accuracy", "Val Accuracy"])

    with open(test_log_file, "w", newline="") as f:
        csv.writer(f).writerow(
            ["Image_Path", "Precision", "Recall", "F1_Score", "IoU_Score", "Specificity", "Accuracy"])

    torch.backends.cudnn.benchmark = True
    scaler = GradScaler()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device Used: {device}")
    if device.type == 'cuda':
        gc.collect()
        torch.cuda.empty_cache()
        print("CUDA cache cleared (or an attempt to clear it was made).")

    # “Path to the prepared dataset”
    dataset_path = "Local dataset path"    
    
    start_time = time.time()

    image_paths, mask_paths = get_images_and_masks(dataset_path)

    # Training augmentations
    train_transform = A.Compose([
        A.Resize(H, W),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(scale=(0.95, 1.05),translate_percent=(0.05, 0.05),rotate=(-7,7),shear=(0, 0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.GaussNoise(p=0.2),
        A.RandomCrop(height=int(H*0.9), width=int(W*0.9), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Validation augmentations
    val_transform = A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    dataset = MedicalDataset(image_paths, mask_paths, transform=train_transform)
    train_len = int(0.80 * len(dataset))
    val_len = int(0.10 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    print(train_len, val_len, test_len, len(dataset))

    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    val_set.dataset.transform = val_transform
    test_set.dataset.transform = val_transform

    train_loader = DataLoader(train_set, batch_size=12, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=12, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=12, pin_memory=True)

    # model name
    base_model = getattr(smp, MODEL)(
        encoder_name="resnet50",   #  resnet34, resnet101, timm-efficientnet-b4
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        encoder_depth=5,
        decoder_aspp_dropout=0.5  #  Used only for DeepLabv3Plus and DeepLabv3
    )

    model = DropBlockDeepLab(base_model, block_size=7, drop_prob=0.10)
    # model=base_model
    criterion = FocalBCELoss(alpha=0.90, gamma=2.0)
    # criterion = CombinedLoss(dice_weight=0.5, focal_weight=0.5)
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)    # 50
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)  # 30
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=NUMEPOCS)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    model.to(device)

    num_epochs = NUMEPOCS   # Epochs
    best_val_loss = float('inf')
    patience = 6
    counter = 0

    for epoch in range(num_epochs):
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

        # scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        with open(train_log_file, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, train_accuracy, val_accuracy])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(NUMEPOCS)+"_best_model_"+MODEL+".pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stop applied.")
                break
    model.load_state_dict(torch.load(str(NUMEPOCS) + "_best_model_" + MODEL + ".pth", weights_only=True,map_location=device))
    model.eval()

    precision_list, recall_list, f1_list, iou_list, specificity_list = [], [], [], [], []

    with torch.no_grad():
        for images, masks, paths in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            for i in range(images.shape[0]):
                p, r, f, iou, s, acc = calculate_metrics(outputs[i], masks[i])
                precision_list.append(p)
                recall_list.append(r)
                f1_list.append(f)
                iou_list.append(iou)
                specificity_list.append(s)
                with open(test_log_file, "a", newline="") as f:
                    csv.writer(f).writerow([paths[i], p, r, f, iou, s, acc])

    # Load the training log file
    log_df = pd.read_csv(train_log_file)

    end_time = time.time()

    print(f"Running Time: {end_time - start_time:.2f} seconds")

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

    print("\nTest Results")
    print(f"IoU: {np.mean(iou_list):.4f}")
    print(f"Precision: {np.mean(precision_list):.4f}")
    print(f"Recall: {np.mean(recall_list):.4f}")
    print(f"Specificity: {np.mean(specificity_list):.4f}")
    print(f"F1-Score: {np.mean(f1_list):.4f}")
    print(f"==============================")

    os.makedirs(str(NUMEPOCS) + "_visual_results_" + MODEL, exist_ok=True)
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    with torch.no_grad():

        save_probability = 0.25

        for images, masks, paths in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            for i in range(images.shape[0]):
                if np.random.rand() > save_probability:
                    continue

                base_filename = os.path.basename(paths[i])
                filename = os.path.splitext(base_filename)[0]
                output_path = f"{str(NUMEPOCS)}_visual_results_{MODEL}/{filename}.png"

                original_img = cv2.imread(paths[i])
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                pred_mask = torch.sigmoid(outputs[i]).cpu().squeeze().numpy()
                binary_mask = (pred_mask > THRESHOLD).astype(np.uint8)

                resized_mask = cv2.resize(binary_mask,
                                          (original_img.shape[1], original_img.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                # Load the ground truth mask and resize it to the original size
                gt_mask = masks[i].cpu().squeeze().numpy()
                gt_binary_mask = (gt_mask > THRESHOLD).astype(np.uint8)
                resized_gt_mask = cv2.resize(gt_binary_mask,
                                             (original_img.shape[1], original_img.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)

                # Detect contours (for overlay visualization)
                gt_contours, _ = cv2.findContours(resized_gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pred_contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # For overlay: make a copy of the original image
                overlay_img = original_img.copy()
                overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)

                # Draw ground truth (green contour) and predicted (red contour) on the overlay
                cv2.drawContours(overlay_bgr, gt_contours, -1, (0, 255, 0), thickness=4)  # Green contour
                cv2.drawContours(overlay_bgr, pred_contours, -1, (0, 0, 255), thickness=4)  # Red contour
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

                # Ground Truth Image: white mask + green contour
                gt_mask_gray = (resized_gt_mask * 255).astype(np.uint8)
                gt_mask_rgb = cv2.cvtColor(gt_mask_gray, cv2.COLOR_GRAY2RGB)
                gt_mask_bgr = cv2.cvtColor(gt_mask_rgb, cv2.COLOR_RGB2BGR)
                cv2.drawContours(gt_mask_bgr, gt_contours, -1, (0, 255, 0), thickness=4)
                gt_rgb = cv2.cvtColor(gt_mask_bgr, cv2.COLOR_BGR2RGB)

                # Predicted Mask Image: white mask + red contour
                pred_mask_gray = (resized_mask * 255).astype(np.uint8)
                pred_mask_rgb = cv2.cvtColor(pred_mask_gray, cv2.COLOR_GRAY2RGB)
                pred_mask_bgr = cv2.cvtColor(pred_mask_rgb, cv2.COLOR_RGB2BGR)
                cv2.drawContours(pred_mask_bgr, pred_contours, -1, (0, 0, 255), thickness=4)
                pred_rgb = cv2.cvtColor(pred_mask_bgr, cv2.COLOR_BGR2RGB)

                # Use matplotlib to display the images side by side
                plt.figure(figsize=(12, 4))

                # Original + GT + Prediction (Overlay image)
                plt.subplot(1, 3, 1)
                plt.imshow(overlay_rgb)
                plt.title("Original image (Overlay Image)")
                plt.axis("off")

                # Ground truth image (white mask + green contour)
                plt.subplot(1, 3, 2)
                plt.imshow(gt_rgb)
                plt.title("Ground Truth Mask")
                plt.axis("off")

                # Predicted image (white mask + red contour)
                plt.subplot(1, 3, 3)
                plt.imshow(pred_rgb)
                plt.title("Predicted Mask")
                plt.axis("off")

                # Save the visualization
                plt.tight_layout()
                plt.savefig(output_path, format='png', bbox_inches='tight', dpi=200)
                plt.close()

    print(f"Random 25% of images were saved")
