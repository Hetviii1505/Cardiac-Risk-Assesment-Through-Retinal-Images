import os
import shutil
import random

# PATHS 
SOURCE_DIR = r"C:\Users\Asys\OneDrive\Desktop\edunet-project\original_dataset\gaussian_filtered_images"
TARGET_DIR = r"C:\Users\Asys\OneDrive\Desktop\edunet-project\Dataset"

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

def split_and_copy(images, target_base):
    random.shuffle(images)
    n = len(images)

    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

    splits = {
        "Train": images[:train_end],
        "Val": images[train_end:val_end],
        "Test": images[val_end:]
    }

    for split, imgs in splits.items():
        for img in imgs:
            src = img
            dst = os.path.join(TARGET_DIR, split, target_base, os.path.basename(img))
            shutil.copy(src, dst)

# ---------------- LOW RISK ----------------
no_dr_path = os.path.join(SOURCE_DIR, "No_DR")
no_dr_images = [
    os.path.join(no_dr_path, f)
    for f in os.listdir(no_dr_path)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
]
split_and_copy(no_dr_images, "Low_Risk")

# ---------------- HIGH RISK ----------------
high_risk_classes = ["Mild", "Moderate", "Severe", "Proliferate_DR"]

for cls in high_risk_classes:
    cls_path = os.path.join(SOURCE_DIR, cls)
    images = [
        os.path.join(cls_path, f)
        for f in os.listdir(cls_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    split_and_copy(images, "High_Risk")

print("✅ Dataset successfully split into train / val / test")
