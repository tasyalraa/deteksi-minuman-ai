import os
import shutil
import cv2

DATASET_DIR = "dataset_tambahan"
CLEAN_DIR = "dataset_clean"

os.makedirs(CLEAN_DIR, exist_ok=True)

# threshold blur (semakin kecil = semakin strict)
BLUR_THRESHOLD = 100

def is_blurry(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return True

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    return variance < BLUR_THRESHOLD

def clean_dataset():
    for class_name in os.listdir(DATASET_DIR):
        class_path = os.path.join(DATASET_DIR, class_name)

        if not os.path.isdir(class_path):
            continue

        clean_class_path = os.path.join(CLEAN_DIR, class_name)
        os.makedirs(clean_class_path, exist_ok=True)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # skip junk file
            if img_name.startswith("."):
                continue

            # check blur
            if is_blurry(img_path):
                print(f"❌ SKIP (blur): {img_path}")
                continue

            # copy clean image
            shutil.copy(img_path, clean_class_path)
            print(f"✔ KEEP: {img_path}")

clean_dataset()
print("DONE CLEANING DATASET")