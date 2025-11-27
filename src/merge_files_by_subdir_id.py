import os
import shutil

# Source and target folders
source_dir = "ureter"        # Existing dataset (with subfolders)
target_dir = "ureter_new"    # Images and masks in the same folder

def organize_dataset():
    os.makedirs(target_dir, exist_ok=True)
    # Traverse the subfolders
    for subdir, _, files in sorted(os.walk(source_dir)):
        folder_name = os.path.basename(subdir)
        if folder_name == source_dir:
            continue  # Ana klasörün kendisini atla

        images = sorted([f for f in files if "image" in f])
        masks = sorted([f for f in files if "mask" in f])

        # Match the images and masks
        for idx, (img, msk) in enumerate(zip(images, masks)):
            img_path = os.path.join(subdir, img)
            msk_path = os.path.join(subdir, msk)

            new_img_name = f"{folder_name}_{idx}_image.png"
            new_msk_name = f"{folder_name}_{idx}_mask.png"

            # Copy the files to a new folder
            shutil.copy(img_path, os.path.join(target_dir, new_img_name))
            shutil.copy(msk_path, os.path.join(target_dir, new_msk_name))

    print(f"All images and masks have been copied to the {target_dir} folder along with their subfolder names.")

if __name__ == "__main__":

    organize_dataset()

