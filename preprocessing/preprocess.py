import os
import cv2
import pandas as pd
import numpy as np

# =======================
# PATHS
# =======================
SOURCE_ROOT = 'merged_dataset'
OUTPUT_ROOT = 'merged_dataset_preprocessed'

IMG_FOLDER = os.path.join(SOURCE_ROOT, 'all_images')
OUT_IMG_FOLDER = os.path.join(OUTPUT_ROOT, 'all_images')

os.makedirs(OUT_IMG_FOLDER, exist_ok=True)

# =======================
# PREPROCESS FUNCTIONS
# =======================
def crop_black_spaces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y:y+h, x:x+w]
    return img


def preprocess_image_swin(img):
    # 1. Crop black borders
    img = crop_black_spaces(img)

    # 2. Resize (224 for Swin-S default)
    img = cv2.resize(img, (224, 224))

    # 3. Convert to LAB color space (better than raw RGB)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 4. Apply CLAHE on L channel ONLY (preserve colors)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    # 5. Merge back
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 6. Light denoising
    img = cv2.GaussianBlur(img, (3, 3), 0)

    return img


# =======================
# PROCESS ALL IMAGES
# =======================
print("🚀 Processing all_images...")

files = [f for f in os.listdir(IMG_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]

for i, filename in enumerate(files):
    src_path = os.path.join(IMG_FOLDER, filename)
    dst_path = os.path.join(OUT_IMG_FOLDER, filename)

    try:
        img = cv2.imread(src_path)
        if img is None:
            print("⚠️ Skipped:", filename)
            continue

        processed = preprocess_image_swin(img)

        cv2.imwrite(dst_path, processed, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    except Exception as e:
        print(f"❌ Error on {filename}: {e}")

    if i % 1000 == 0:
        print(f"Processed {i}/{len(files)}")

# =======================
# COPY CSV FILES (NO CHANGE)
# =======================
print("📄 Copying CSV files...")

for csv_name in ['train.csv', 'val.csv', 'test.csv']:
    src_csv = os.path.join(SOURCE_ROOT, csv_name)
    dst_csv = os.path.join(OUTPUT_ROOT, csv_name)

    df = pd.read_csv(src_csv)
    df.to_csv(dst_csv, index=False)

print("✅ Done! Preprocessed dataset ready at:", OUTPUT_ROOT)
