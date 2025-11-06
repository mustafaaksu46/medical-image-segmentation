import os
import shutil

# 🔹 Kaynak ve hedef klasörler
source_dir = "ureter"        # Mevcut veri seti (alt klasörler var)
target_dir = "ureter_yeni"             # image ve mask aynı klasörde

def organize_dataset():
    # 🔹 Hedef klasörü oluştur (eğer yoksa)
    os.makedirs(target_dir, exist_ok=True)

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

            new_img_name = f"{folder_name}_{idx}_image.png"
            new_msk_name = f"{folder_name}_{idx}_mask.png"

            # 🔹 Dosyaları yeni klasöre kopyala
            shutil.copy(img_path, os.path.join(target_dir, new_img_name))
            shutil.copy(msk_path, os.path.join(target_dir, new_msk_name))

    print(f"✅ Tüm görüntü ve maskeler alt klasör isimleriyle birlikte '{target_dir}' klasörüne kopyalandı.")

if __name__ == "__main__":
    organize_dataset()