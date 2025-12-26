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
from torch.utils.data import DataLoader

from model_Patient_Based import (MedicalDataset, build_model, calculate_metrics, FocalBCELoss,
                   FocalBCEDiceLoss, DiceLoss, CombinedLoss, get_images_and_masks)

""" Global parameters """
H = 256
W = 256
NUMEPOCS = 5
BATCH_SIZE = 12
THRESHOLD = 0.40

MODEL = "DeepLabV3Plus"     #Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus, Segformer
ENCODER_NAME = "resnet50"      # "resnet34", "efficientnet-b4", "mit_b2"

if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

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

    test_log_file = str(NUMEPOCS) + "_Pancreas_test_results_" + MODEL + ".csv"

    with open(test_log_file, "w", newline="") as f:
        csv.writer(f).writerow(
            ["Image_Path", "Precision", "Recall", "F1_Score", "IoU_Score", "Specificity", "Accuracy"])

    dataset_path = "Path to the local dataset"

    train_imgs, train_masks = get_images_and_masks(dataset_path, "train")
    val_imgs, val_masks = get_images_and_masks(dataset_path, "val")
    test_imgs, test_masks = get_images_and_masks(dataset_path, "test")

    print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    test_set  = MedicalDataset(test_imgs, test_masks, transform=val_transform)

    test_loader  = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(MODEL, ENCODER_NAME)
    model.to(device)

    print("MODEL:", model.__class__.__name__)
    print("ENCODER:", ENCODER_NAME)

    # ===== TEST =====
    start_time = time.time()
    model.load_state_dict(torch.load(str(NUMEPOCS) + "_best_model_" + MODEL + ".pth", weights_only=True, map_location=device))
    model.eval()

    precision_list, recall_list, f1_list, iou_list, specificity_list = [], [], [], [], []

    with torch.no_grad():
        for images, masks, paths in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            for i in range(images.shape[0]):
                p, r, f1, iou, s, acc = calculate_metrics(outputs[i], masks[i], use_logits=True)
                precision_list.append(p)
                recall_list.append(r)
                f1_list.append(f1)
                iou_list.append(iou)
                specificity_list.append(s)
                with open(test_log_file, "a", newline="") as f:
                    csv.writer(f).writerow([paths[i], p, r, f1, iou, s, acc])

    end_time = time.time()

    print(f"Running Time: {end_time - start_time:.2f} seconds")

    print("\n====== Test Results ======")
    print(f"IoU:         {np.mean(iou_list):.4f}")
    print(f"Precision:   {np.mean(precision_list):.4f}")
    print(f"Recall:      {np.mean(recall_list):.4f}")
    print(f"Specificity: {np.mean(specificity_list):.4f}")
    print(f"F1-Score:    {np.mean(f1_list):.4f}")
    print(f"=========================")

    with open(test_log_file, "a", newline="") as f:
        csv.writer(f).writerow(["MEAN_RESULTS",  np.mean(precision_list),  np.mean(recall_list), np.mean(f1_list), np.mean(iou_list), np.mean(specificity_list), "-"])

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

