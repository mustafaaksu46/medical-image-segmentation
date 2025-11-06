import os
import shutil

# 🔹 Kaynak ve hedef klasörler
source_dir = "ureter"        # Mevcut veri seti (alt klasörler var)
target_dir = "ureter_yeni"       # Yeni klasör (altında images/ ve masks/ olacak)
images_dir = os.path.join(target_dir, "images")
masks_dir = os.path.join(target_dir, "masks")

def organize_dataset():
    # 🔹 Hedef klasörleri oluştur (eğer yoksa)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # 🔹 Alt klasörleri dolaş
    for subdir, _, files in sorted(os.walk(source_dir)):
        folder_name = os.path.basename(subdir)
        if folder_name == source_dir:
            continue  # Ana klasörün kendisini atla

        images = sorted([f for f in files if "image" in f])
        masks = sorted([f for f in files if "mask" in f])

        # 🔹 Görüntü ve maskeleri eşleştir
        for idx, (img, msk) in enumerate(zip(images, masks)):
            img_path = os.path.join(subdir, img)
            msk_path = os.path.join(subdir, msk)

            new_img_name = f"{folder_name}_{idx}.png"
            new_msk_name = f"{folder_name}_{idx}.png"

            # 🔹 Dosyaları uygun alt klasörlere kopyala
            shutil.copy(img_path, os.path.join(images_dir, new_img_name))
            shutil.copy(msk_path, os.path.join(masks_dir, new_msk_name))

    print(f"✅ Tüm görüntüler '{images_dir}' ve maskeler '{masks_dir}' klasörüne kopyalandı.")

if __name__ == "__main__":
    organize_dataset()