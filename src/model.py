import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from dropblock import DropBlock2D

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

#  Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus, Segformer
def create_model(model_name="DeepLabV3Plus",
                 encoder_name="resnet50",
                 use_dropblock=True,
                 block_size=7,
                 drop_prob=0.10):

    if model_name == "DeepLabV3Plus":
        base_model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
            encoder_depth=5,
            decoder_aspp_dropout=0.5
        )
    else:
        base_model = getattr(smp, model_name)(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )

    if use_dropblock:
        model = DropBlockDeepLab(base_model, block_size=block_size, drop_prob=drop_prob)
    else:
        model = base_model

    return model