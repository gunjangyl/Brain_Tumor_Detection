"""
Download the Brain MRI dataset from Kaggle.
Run once: python download_dataset.py

You will be prompted for:
  - Kaggle username  (from kaggle.com -> Account -> Settings)
  - Kaggle API key   (from kaggle.com -> Account -> Create New Token -> kaggle.json)
"""
import os
import sys
import json
import shutil

# Fix Windows console encoding issues
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=== Downloading Brain MRI Dataset from Kaggle ===")
print("You will be prompted for your Kaggle username and API key.\n")
print("Find them at: https://www.kaggle.com/settings -> API -> Create New Token\n")

DATASET_URL = "https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection"
RAW_DIR     = "brain-mri-images-for-brain-tumor-detection"
TARGET_DIR  = "data"

# Check if data already exists
if (os.path.isdir(os.path.join(TARGET_DIR, "yes")) and
        os.path.isdir(os.path.join(TARGET_DIR, "no"))):
    yes_c = len(os.listdir(os.path.join(TARGET_DIR, "yes")))
    no_c  = len(os.listdir(os.path.join(TARGET_DIR, "no")))
    print(f"Dataset already present: yes={yes_c}, no={no_c}")
    print("No download needed. Run: python train_model.py")
    sys.exit(0)

# Try to use stored kaggle.json first
kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
if os.path.exists(kaggle_json):
    print(f"Found existing credentials at {kaggle_json}")
else:
    # Ask user for credentials and persist them
    print("Enter your Kaggle credentials (one-time setup):")
    username = input("Kaggle username: ").strip()
    api_key  = input("Kaggle API key : ").strip()

    kaggle_dir = os.path.join(os.path.expanduser("~"), ".kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    with open(kaggle_json, "w") as f:
        json.dump({"username": username, "key": api_key}, f)
    # Restrict permissions on Windows — best effort
    try:
        import stat
        os.chmod(kaggle_json, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    print(f"Credentials saved to {kaggle_json}\n")

# Now download via opendatasets (uses the saved kaggle.json)
try:
    import opendatasets as od
    od.download(DATASET_URL, data_dir=".")
except Exception as e:
    print(f"\nopendatasets error: {e}")
    print("Trying kaggle CLI instead...")
    ret = os.system(
        "kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection "
        "--unzip -p data/"
    )
    if ret != 0:
        print("Download failed. Please download manually from:")
        print("  https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection")
        sys.exit(1)

# Move to data/ if downloaded into the named folder
if os.path.exists(RAW_DIR) and not os.path.exists(TARGET_DIR):
    shutil.move(RAW_DIR, TARGET_DIR)
    print(f"\nMoved dataset folder to '{TARGET_DIR}/'")
elif os.path.exists(RAW_DIR) and os.path.exists(TARGET_DIR):
    for cls in ("yes", "no"):
        src = os.path.join(RAW_DIR, cls)
        dst = os.path.join(TARGET_DIR, cls)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
    try:
        shutil.rmtree(RAW_DIR)
    except Exception:
        pass
    print(f"\nMerged dataset into '{TARGET_DIR}/'")

# Verify
print("\nDataset contents:")
for cls in ("yes", "no"):
    p = os.path.join(TARGET_DIR, cls)
    if os.path.isdir(p):
        imgs = [f for f in os.listdir(p)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  {cls}: {len(imgs)} images")
    else:
        print(f"  WARNING: {p} not found!")

print("\nDataset ready. Now run: python train_model.py")
