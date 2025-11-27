import os
import shutil

# Source and target folders
source_dir = "ureter"           # Existing dataset (with subfolders)
target_dir = "ureter_new"       # New folder (will contain images/ and masks/ subfolders)
images_dir = os.path.join(target_dir, "images")
masks_dir = os.path.join(target_dir, "masks")

def organize_dataset():
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # Traverse subfolders
    for subdir, _, files in sorted(os.walk(source_dir)):
        folder_name = os.path.basename(subdir)
        if folder_name == source_dir:
            continue   # Exclude the root folder

        images = sorted([f for f in files if "image" in f])
        masks = sorted([f for f in files if "mask" in f])

        # Match images and masks
        for idx, (img, msk) in enumerate(zip(images, masks)):
            img_path = os.path.join(subdir, img)
            msk_path = os.path.join(subdir, msk)

            new_img_name = f"{folder_name}_{idx}.png"
            new_msk_name = f"{folder_name}_{idx}.png"

            # Copy files to the appropriate subfolders
            shutil.copy(img_path, os.path.join(images_dir, new_img_name))
            shutil.copy(msk_path, os.path.join(masks_dir, new_msk_name))

    print(f"All images have been copied to '{images_dir}' and all masks to '{masks_dir}' folders.")

if __name__ == "__main__":

    organize_dataset()

