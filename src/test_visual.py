import os
import cv2
import csv
import torch
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model import create_model
from train import MedicalDataset, calculate_metrics, get_images_and_masks

""" Global parameters """
H = 256
W = 256
NUMEPOCS = 50
THRESHOLD = 0.40
MODEL = "DeepLabV3Plus"     #  Unet, UnetPlusPlus, DeepLabV3, DeepLabV3Plus, Segformer


def test_and_visualize(dataset_path, model_path, model_name=MODEL, num_epochs=NUMEPOCS,
                       batch_size=12, save_probability=0.25):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device Used: {device}")

    # Create test log file
    test_log_file = f"{num_epochs}_Pancreas_test_results_{model_name}.csv"
    with open(test_log_file, "w", newline="") as f:
        csv.writer(f).writerow(
            ["Image_Path", "Precision", "Recall", "F1_Score", "IoU_Score", "Specificity", "Accuracy"])

    # Load dataset
    image_paths, mask_paths = get_images_and_masks(dataset_path)

    # Define validation transform
    val_transform = A.Compose([
        A.Resize(H, W),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Create dataset and split
    dataset = MedicalDataset(image_paths, mask_paths, transform=val_transform)
    train_len = int(0.80 * len(dataset))
    val_len = int(0.10 * len(dataset))
    test_len = len(dataset) - train_len - val_len

    _, _, test_set = random_split(dataset, [train_len, val_len, test_len])
    test_loader = DataLoader(test_set, batch_size=batch_size, pin_memory=True)

    print(f"Test set size: {test_len}")

    # Load model
    model = create_model(model_name=model_name, encoder_name="resnet50", use_dropblock=True,
                         block_size=7, drop_prob=0.10)
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    # Test model
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

                with open(test_log_file, "a", newline="") as file:
                    csv.writer(file).writerow([paths[i], p, r, f, iou, s, acc])

    # Test results
    print("\n" + "=" * 50)
    print("Test Results")
    print("=" * 50)
    print(f"IoU:         {np.mean(iou_list):.4f}")
    print(f"Precision:   {np.mean(precision_list):.4f}")
    print(f"Recall:      {np.mean(recall_list):.4f}")
    print(f"Specificity: {np.mean(specificity_list):.4f}")
    print(f"F1-Score:    {np.mean(f1_list):.4f}")
    print("=" * 50)

    # Create visualizations
    output_dir = f"{num_epochs}_visual_results_{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for images, masks, paths in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            for i in range(images.shape[0]):
                # Random sampling for visualization
                if np.random.rand() > save_probability:
                    continue

                # Prepare file paths
                base_filename = os.path.basename(paths[i])
                filename = os.path.splitext(base_filename)[0]
                output_path = f"{output_dir}/{filename}.png"

                # Load original image
                original_img = cv2.imread(paths[i])
                original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

                # Process predicted mask
                pred_mask = torch.sigmoid(outputs[i]).cpu().squeeze().numpy()
                binary_mask = (pred_mask > THRESHOLD).astype(np.uint8)
                resized_mask = cv2.resize(binary_mask,
                                          (original_img.shape[1], original_img.shape[0]),
                                          interpolation=cv2.INTER_NEAREST)

                # Process ground truth mask
                gt_mask = masks[i].cpu().squeeze().numpy()
                gt_binary_mask = (gt_mask > THRESHOLD).astype(np.uint8)
                resized_gt_mask = cv2.resize(gt_binary_mask,
                                             (original_img.shape[1], original_img.shape[0]),
                                             interpolation=cv2.INTER_NEAREST)

                # Find contours
                gt_contours, _ = cv2.findContours(resized_gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                pred_contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Create overlay image (original + GT green + Pred red)
                overlay_img = original_img.copy()
                overlay_bgr = cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR)
                cv2.drawContours(overlay_bgr, gt_contours, -1, (0, 255, 0), thickness=4)  # Green GT
                cv2.drawContours(overlay_bgr, pred_contours, -1, (0, 0, 255), thickness=4)  # Red Pred
                overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

                # Create GT mask image (white mask + green contour)
                gt_mask_gray = (resized_gt_mask * 255).astype(np.uint8)
                gt_mask_rgb = cv2.cvtColor(gt_mask_gray, cv2.COLOR_GRAY2RGB)
                gt_mask_bgr = cv2.cvtColor(gt_mask_rgb, cv2.COLOR_RGB2BGR)
                cv2.drawContours(gt_mask_bgr, gt_contours, -1, (0, 255, 0), thickness=4)
                gt_rgb = cv2.cvtColor(gt_mask_bgr, cv2.COLOR_BGR2RGB)

                # Create predicted mask image (white mask + red contour)
                pred_mask_gray = (resized_mask * 255).astype(np.uint8)
                pred_mask_rgb = cv2.cvtColor(pred_mask_gray, cv2.COLOR_GRAY2RGB)
                pred_mask_bgr = cv2.cvtColor(pred_mask_rgb, cv2.COLOR_RGB2BGR)
                cv2.drawContours(pred_mask_bgr, pred_contours, -1, (0, 0, 255), thickness=4)
                pred_rgb = cv2.cvtColor(pred_mask_bgr, cv2.COLOR_BGR2RGB)

                # Create visualization
                plt.figure(figsize=(12, 4))

                # Original image with overlay
                plt.subplot(1, 3, 1)
                plt.imshow(overlay_rgb)
                plt.title("Original Image (Overlay)")
                plt.axis("off")

                # Ground truth mask
                plt.subplot(1, 3, 2)
                plt.imshow(gt_rgb)
                plt.title("Ground Truth Mask")
                plt.axis("off")

                # Predicted mask
                plt.subplot(1, 3, 3)
                plt.imshow(pred_rgb)
                plt.title("Predicted Mask")
                plt.axis("off")

                # Save visualization
                plt.tight_layout()
                plt.savefig(output_path, format='png', bbox_inches='tight', dpi=200)
                plt.close()

    print(f"\nVisualizations saved to {output_dir}/")
    print(f"Random {int(save_probability * 100)}% of test images were visualized")


if __name__ == "__main__":
    import torch.multiprocessing

    torch.multiprocessing.freeze_support()
    
  # “Path to the prepared dataset”
    dataset_path = "   Local dataset path   "  
    model_path = f"{NUMEPOCS}_best_model_{MODEL}.pth"

    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using train.py")
    else:
        test_and_visualize(
            dataset_path=dataset_path,
            model_path=model_path,
            model_name=MODEL,
            num_epochs=NUMEPOCS,
            batch_size=12,
            save_probability=0.25

        )

