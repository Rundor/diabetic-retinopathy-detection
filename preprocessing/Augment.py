import os
import shutil
import pandas as pd
from PIL import Image
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance


# =======================
# PATHS
# =======================
DATASET_PATH = 'merged_dataset'          # your merged dataset folder
IMG_FOLDER = os.path.join(DATASET_PATH, 'all_images')

TRAIN_CSV = os.path.join(DATASET_PATH, 'train.csv')
VAL_CSV   = os.path.join(DATASET_PATH, 'val.csv')
TEST_CSV  = os.path.join(DATASET_PATH, 'test.csv')

AUGMENTED_PREFIX = 'aug_'   # prefix for augmented images


def circular_mask(img):
    # Convert to numpy
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    
    # Create mask
    y, x = np.ogrid[:h, :w]
    center = (h/2, w/2)
    radius = min(center)  # use smallest dimension
    mask = (x - center[1])**2 + (y - center[0])**2 <= radius**2
    
    # Apply mask
    masked_img = np.zeros_like(img_np)
    masked_img[mask] = img_np[mask]
    
    return Image.fromarray(masked_img)
    
# =======================
# AUGMENTATION FUNCTION
# =======================
import numpy as np
from PIL import Image, ImageEnhance
import random


def safe_vit_augment(img, brightness_range=(0.9,1.1), contrast_range=(0.9,1.1), seed=None):
    """
    Safe augmentation for fundus images for ViT:
    - Horizontal flip
    - Brightness & contrast jitter
    """
    if seed is not None:
        random.seed(seed)

    # Horizontal flip with 50% probability
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # Brightness jitter
    factor = random.uniform(*brightness_range)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast jitter
    factor = random.uniform(*contrast_range)
    img = ImageEnhance.Contrast(img).enhance(factor)

    return img

# =======================
# LOAD TRAIN CSV
# =======================
train_df = pd.read_csv(TRAIN_CSV)

# =======================
# Identify minority classes (stages 3 & 4)
# =======================
minority_classes = [3, 4]
class_counts = train_df['diagnosis'].value_counts()
max_count = train_df['diagnosis'].value_counts().max()

print("Class counts before augmentation:\n", class_counts)

# =======================
# AUGMENTATION LOOP
# =======================
augmented_rows = []

for cls in minority_classes:
    cls_df = train_df[train_df['diagnosis'] == cls]
    n_to_generate = max_count - len(cls_df)   # how many extra needed

    print(f"Augmenting class {cls}: {n_to_generate} new images")

    # repeat images if needed
    img_list = cls_df['id_code'].tolist()
    idx = 0
    for i in range(n_to_generate):
        orig_name = img_list[idx % len(img_list)]
        idx += 1

        # open image
        img_path = os.path.join(IMG_FOLDER, orig_name)
        if not os.path.exists(img_path):
            print("Missing image:", img_path)
            continue
        img = Image.open(img_path).convert('RGB')

        # augment
        img_aug = safe_vit_augment(img, seed=i)

        # save with new name
        new_name = f"{AUGMENTED_PREFIX}{orig_name.replace('.jpg','')}_{i}.jpg"
        img_aug.save(os.path.join(IMG_FOLDER, new_name))

        # add new row to CSV
        augmented_rows.append({'id_code': new_name, 'diagnosis': cls})

# =======================
# UPDATE TRAIN CSV
# =======================
if augmented_rows:
    aug_df = pd.DataFrame(augmented_rows)
    train_aug_df = pd.concat([train_df, aug_df], ignore_index=True)
else:
    train_aug_df = train_df.copy()  # no new images, still save CSV to be safe

train_aug_df.to_csv(TRAIN_CSV, index=False)

# =======================
# SHOW FINAL COUNTS
# =======================
final_counts = train_aug_df['diagnosis'].value_counts()
print("Class counts after augmentation:\n", final_counts)

print("✅ Augmentation complete! Images and CSV updated.")
