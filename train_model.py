"""
Brain Tumor Detection - VGG16 Transfer Learning Trainer
=========================================================
Dataset : data/yes/  (tumor MRI scans)
          data/no/   (healthy MRI scans)

Output  : model/VGG_model.h5
          Compatible with predictor.py:
              output[0][0] == 1  -> Tumor detected
              output[0][0] == 0  -> No Tumor

Usage   : python train_model.py
"""

import os
import sys
import numpy as np

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array
)

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "VGG_model.h5")

IMG_SIZE   = (224, 224)
BATCH_SIZE = 16
SEED       = 42

IMG_EXTS   = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# ── Checks ──────────────────────────────────────────────────────────────────
for cls in ("yes", "no"):
    if not os.path.isdir(os.path.join(DATA_DIR, cls)):
        print(f"[ERROR] Missing: data/{cls}/")
        print("Run: kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection --unzip -p data/")
        sys.exit(1)

os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load image paths + labels manually (avoids picking up nested folders) ──
def load_paths_labels(data_dir):
    paths, labels = [], []
    for label, cls in enumerate(("no", "yes")):   # no=0, yes=1  (matches predictor.py)
        folder = os.path.join(data_dir, cls)
        for fname in os.listdir(folder):
            if fname.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(folder, fname))
                labels.append(label)
    return np.array(paths), np.array(labels)

all_paths, all_labels = load_paths_labels(DATA_DIR)
yes_c = np.sum(all_labels == 1)
no_c  = np.sum(all_labels == 0)
total = len(all_labels)

print(f"\nDataset summary:")
print(f"  Tumor   (yes=1): {yes_c}")
print(f"  Healthy (no=0) : {no_c}")
print(f"  Total          : {total}\n")

# ── Train/val split ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    all_paths, all_labels,
    test_size=0.20,
    stratify=all_labels,
    random_state=SEED
)
print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

# ── Class weights ────────────────────────────────────────────────────────────
class_weights = {
    0: total / (2.0 * no_c),    # no-tumor
    1: total / (2.0 * yes_c),   # tumor
}
print(f"Class weights: no={class_weights[0]:.3f}, yes={class_weights[1]:.3f}\n")

# ── Custom data generator ────────────────────────────────────────────────────
def make_generator(paths, labels, augment=False, batch_size=BATCH_SIZE, shuffle=True):
    """Yields (batch_images, batch_labels) indefinitely."""
    aug = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        fill_mode="nearest",
    ) if augment else ImageDataGenerator()

    n = len(paths)
    indices = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_x, batch_y = [], []
            for i in batch_idx:
                img = load_img(paths[i], target_size=IMG_SIZE)
                arr = img_to_array(img) / 255.0
                batch_x.append(arr)
                batch_y.append(labels[i])
            batch_x = np.array(batch_x)
            batch_y = np.array(batch_y, dtype=np.float32)
            if augment:
                for j in range(len(batch_x)):
                    batch_x[j] = aug.random_transform(batch_x[j])
            yield batch_x, batch_y

train_steps = int(np.ceil(len(X_train) / BATCH_SIZE))
val_steps   = int(np.ceil(len(X_val)   / BATCH_SIZE))

train_gen = make_generator(X_train, y_train, augment=True,  shuffle=True)
val_gen   = make_generator(X_val,   y_val,   augment=False, shuffle=False)

# ── Model: VGG16 + custom head ───────────────────────────────────────────────
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# ── Phase 1: Train head only ─────────────────────────────────────────────────
print("\n=== Phase 1: Training top layers (VGG base frozen) ===\n")
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

cb1 = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
]

h1 = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=cb1,
    class_weight=class_weights,
)

# ── Phase 2: Fine-tune last VGG block ────────────────────────────────────────
print("\n=== Phase 2: Fine-tuning VGG16 block5 ===\n")
for layer in base_model.layers:
    layer.trainable = layer.name.startswith("block5")

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

cb2 = [
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-8, verbose=1),
]

h2 = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    epochs=30,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=cb2,
    class_weight=class_weights,
)

# ── Save training curves ─────────────────────────────────────────────────────
def merge(h1, h2, key):
    return h1.history.get(key, []) + h2.history.get(key, [])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(merge(h1, h2, "accuracy"),     label="train")
axes[0].plot(merge(h1, h2, "val_accuracy"), label="val")
axes[0].set_title("Accuracy"); axes[0].legend()

axes[1].plot(merge(h1, h2, "loss"),     label="train")
axes[1].plot(merge(h1, h2, "val_loss"), label="val")
axes[1].set_title("Loss"); axes[1].legend()

plt.tight_layout()
plot_path = os.path.join(MODEL_DIR, "training_curves.png")
plt.savefig(plot_path)
print(f"\nTraining curves saved to {plot_path}")

# ── Final evaluation ─────────────────────────────────────────────────────────
print("\n=== Final Evaluation on Validation Set ===")
val_gen_eval = make_generator(X_val, y_val, augment=False, shuffle=False)
loss, acc = model.evaluate(val_gen_eval, steps=val_steps, verbose=1)
print(f"  Val Loss     : {loss:.4f}")
print(f"  Val Accuracy : {acc:.4f}  ({acc*100:.1f}%)")

# ── Done ─────────────────────────────────────────────────────────────────────
if os.path.exists(MODEL_PATH):
    size_mb = os.path.getsize(MODEL_PATH) / 1_048_576
    print(f"\n[OK] Model saved to {MODEL_PATH}  ({size_mb:.1f} MB)")
    print("     Run: python app.py")
else:
    print(f"\n[FAIL] Model file not found at {MODEL_PATH}")
    sys.exit(1)
