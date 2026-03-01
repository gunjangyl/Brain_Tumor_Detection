"""
Download & Organize the 4-class Brain Tumor MRI Dataset for Research.
Dependencies: opendatasets, kaggle

Dataset: masoudnickparvar/brain-tumor-mri-dataset
Classes: glioma, meningioma, notumor, pituitary
Split:   The raw dataset provides 'Training' and 'Testing' folders.
         We will merge them into a single 'data_multi/' folder to allow
         custom Stratified Train/Val/Test splits during training.
"""

import os
import sys
import shutil

# Fix Windows console encoding issues
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=== Downloading Research-Grade 4-Class Brain MRI Dataset ===")

DATASET_URL = "https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset"
RAW_DIR     = "brain-tumor-mri-dataset"
TARGET_DIR  = "data_multi"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

# Check if data already exists
if os.path.exists(TARGET_DIR):
    counts = {}
    valid = True
    for c in CLASSES:
        p = os.path.join(TARGET_DIR, c)
        if os.path.isdir(p):
            counts[c] = len([f for f in os.listdir(p) if f.lower().endswith(('.jpg','.jpeg','.png'))])
        else:
            valid = False
            break
    if valid and sum(counts.values()) > 6000:
        print(f"Dataset already present and populated: {counts}")
        print("Total images:", sum(counts.values()))
        print("No download needed.")
        sys.exit(0)

# Try kaggle CLI first since credentials should be in ~/.kaggle/kaggle.json
print("\nAttempting download via Kaggle CLI...")
ret = os.system(
    "kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip"
)

if ret != 0:
    print("\n[ERROR] Kaggle CLI failed. Falling back to opendatasets...")
    try:
        import opendatasets as od
        od.download(DATASET_URL, data_dir=".")
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)

# The dataset extracts into 'Training' and 'Testing' folders containing the 4 class subfolders.
# We will merge them into our TARGET_DIR for a uniform split later.
print("\nReorganizing into unified directory structure...")
os.makedirs(TARGET_DIR, exist_ok=True)
for c in CLASSES:
    os.makedirs(os.path.join(TARGET_DIR, c), exist_ok=True)

raw_bases = []
if os.path.exists("Training"): raw_bases.append("Training")
if os.path.exists("Testing"): raw_bases.append("Testing")
if os.path.exists(os.path.join(RAW_DIR, "Training")): raw_bases.append(os.path.join(RAW_DIR, "Training"))
if os.path.exists(os.path.join(RAW_DIR, "Testing")): raw_bases.append(os.path.join(RAW_DIR, "Testing"))

moved_count = 0
for base in raw_bases:
    for c in CLASSES:
        src_dir = os.path.join(base, c)
        if not os.path.isdir(src_dir):
            src_dir = os.path.join(base, c.capitalize()) # Handle case sensitivity

        if os.path.isdir(src_dir):
            for fname in os.listdir(src_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_file = os.path.join(src_dir, fname)
                    # Use a unique name to avoid collisions between Training/Testing sets
                    unique_name = f"{base.replace(os.sep, '_')}_{fname}"
                    dst_file = os.path.join(TARGET_DIR, c, unique_name)
                    if not os.path.exists(dst_file):
                        shutil.move(src_file, dst_file)
                        moved_count += 1

print(f"\nMoved {moved_count} images into '{TARGET_DIR}/'")

# Clean up raw folders
for base in raw_bases:
    try:
        shutil.rmtree(base)
    except Exception:
        pass
try:
    if os.path.exists(RAW_DIR): shutil.rmtree(RAW_DIR)
except Exception:
    pass

# Verify
print("\nDataset contents:")
total = 0
for c in CLASSES:
    p = os.path.join(TARGET_DIR, c)
    if os.path.isdir(p):
        imgs = [f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {c:<12}: {len(imgs)} images")
        total += len(imgs)
    else:
        print(f"  [WARNING] {p} not found!")
print(f"==> Total Images: {total}")
