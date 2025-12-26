import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset


class MedicalDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask / 255.0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask, self.image_paths[idx]


def build_model(MODEL, ENCODER_NAME):

    if MODEL == "DeepLabV3Plus":
        return smp.DeepLabV3Plus(
            encoder_name=ENCODER_NAME,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            decoder_dropout=0.3,
            decoder_aspp_dropout=0.5
        )

    elif MODEL == "UnetPlusPlus":
        return smp.UnetPlusPlus(
            encoder_name=ENCODER_NAME,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            decoder_dropout=0.3
        )

    elif MODEL == "Segformer":
        return smp.Segformer(
            encoder_name=ENCODER_NAME,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

    else:
        raise ValueError(f"Unknown MODEL type: {MODEL}")


def calculate_metrics(pred, mask, threshold=0.4, use_logits=False):
    smooth = 1e-6

    if use_logits:
        pred = torch.sigmoid(pred)

    pred_binary = (pred > threshold).float()

    TP = torch.sum(pred_binary * mask).item()
    FP = torch.sum(pred_binary * (1 - mask)).item()
    FN = torch.sum((1 - pred_binary) * mask).item()
    TN = torch.sum((1 - pred_binary) * (1 - mask)).item()

    precision = TP / (TP + FP + smooth)
    recall = TP / (TP + FN + smooth)
    f1 = 2 * (precision * recall) / (precision + recall + smooth)
    iou = TP / (TP + FP + FN + smooth)
    specificity = TN / (TN + FP + smooth)
    accuracy = (TP + TN) / (TP + TN + FP + FN + smooth)

    return precision, recall, f1, iou, specificity, accuracy


def get_images_and_masks(dataset_path, split):

    images_dir = os.path.join(dataset_path, "images", split)
    masks_dir  = os.path.join(dataset_path, "masks", split)

    images = sorted(os.listdir(str(images_dir)))
    masks = sorted(os.listdir(str(masks_dir)))

    assert len(images) == len(masks), f"{split} image-mask count mismatch"

    image_paths = [os.path.join(str(images_dir), f) for f in images]
    mask_paths  = [os.path.join(str(masks_dir), f) for f in masks]

    return image_paths, mask_paths

class FocalBCELoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=1.5):
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


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, inputs, targets):
        smooth = 1e-5

        bce = F.binary_cross_entropy_with_logits(inputs, targets)

        probs = torch.sigmoid(inputs)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = 1 - (2. * intersection + smooth) / (probs.sum() + targets.sum() + smooth)

        return self.dice_weight * dice + self.bce_weight * bce

class FocalBCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.85, gamma=1.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Focal Loss kısmı
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce_loss).mean()
        inputs_sigmoid = torch.sigmoid(inputs)
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (
                    inputs_flat.sum() + targets_flat.sum() + self.smooth)

        return focal_loss + dice_loss

