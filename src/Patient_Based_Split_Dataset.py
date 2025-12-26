import os
import shutil
import random
from glob import glob
from tqdm import tqdm

SOURCE_DIR = r"-----Source dataset path----"
OUTPUT_DIR = r"-----Output dataset path----"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
RANDOM_SEED = 42

IMAGE_KEY = "image"
MASK_KEY = "mask"

random.seed(RANDOM_SEED)

# CREATE OUTPUT STRUCTURE
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "masks", split), exist_ok=True)

# COLLECT PATIENTS
patients = sorted(
    d for d in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, d))
)

random.shuffle(patients)

n_train = int(len(patients) * TRAIN_RATIO)
n_val = int(len(patients) * VAL_RATIO)

splits = {
    "train": patients[:n_train],
    "val": patients[n_train:n_train + n_val],
    "test": patients[n_train + n_val:]
}

print(
    f"Patients → Train: {len(splits['train'])}, "
    f"Val: {len(splits['val'])}, "
    f"Test: {len(splits['test'])}"
)

global_counter = 0

for split, patient_list in splits.items():
    print(f"\nProcessing {split.upper()} set...")

    out_img_dir = os.path.join(OUTPUT_DIR, "images", split)
    out_mask_dir = os.path.join(OUTPUT_DIR, "masks", split)

    for pid in patient_list:
        patient_dir = os.path.join(SOURCE_DIR, pid)

        image_paths = sorted(
            glob(os.path.join(patient_dir, f"*{IMAGE_KEY}*.png"))
        )

        for img_path in tqdm(image_paths, leave=False):
            mask_path = img_path.replace(IMAGE_KEY, MASK_KEY)

            if not os.path.exists(mask_path):
                continue

            img_filename = os.path.basename(img_path)
            name_no_ext, _ = os.path.splitext(img_filename)

            # Remove image keyword safely
            name_no_ext = name_no_ext.replace(IMAGE_KEY, "").strip("_")

            out_name = f"{pid}_{global_counter}.png"

            shutil.copy2(img_path, os.path.join(out_img_dir, out_name))
            shutil.copy2(mask_path, os.path.join(out_mask_dir, out_name))

            global_counter += 1

print("\nDONE")
print("📁 Dataset ready for U-Net++, SegFormer, DeepLab")
print("📁 Output path:", OUTPUT_DIR)