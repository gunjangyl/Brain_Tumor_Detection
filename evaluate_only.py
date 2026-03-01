"""
Model Evaluation Script
=======================
Extracts ROC-AUC, classification metrics, and Confusion Matrices using the 
successfully trained `EfficientNet_model.h5` weights.
"""
import os
import sys
import numpy as np

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR    = "data_multi"
MODEL_DIR   = "model"
MODEL_PATH  = os.path.join(MODEL_DIR, "EfficientNet_model.h5")
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
SEED        = 42
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)

print("--- Initializing Evaluation Subsystem ---")
paths, labels = [], []
for idx, cls_name in enumerate(CLASSES):
    cls_folder = os.path.join(DATA_DIR, cls_name)
    files = [f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    for f in files:
        paths.append(os.path.join(cls_folder, f))
        labels.append(idx)
        
paths = np.array(paths)
labels = np.array(labels)
X_temp, X_test, y_temp, y_test = train_test_split(paths, labels, test_size=0.10, stratify=labels, random_state=SEED)

test_df  = pd.DataFrame({'path': X_test,  'label': [CLASSES[y] for y in y_test]})
val_aug = ImageDataGenerator()
test_gen = val_aug.flow_from_dataframe(
    test_df, x_col='path', y_col='label', target_size=IMG_SIZE,
    class_mode='categorical', batch_size=BATCH_SIZE, shuffle=False
)
test_steps = len(test_gen)

print("--- Building Architecture & Loading Weights ---")
base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(*IMG_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)
model.load_weights(MODEL_PATH)

print("\n--- Running Strict Evaluation on Unseen Test Set ---")
y_true = test_gen.classes
y_pred_probs = model.predict(test_gen, steps=test_steps, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

# A. Classification Report
report = classification_report(y_true, y_pred, target_names=CLASSES)
with open(os.path.join(MODEL_DIR, "metrics_report.txt"), "w") as f:
    f.write("=== Brain Tumor Multi-Class Inference Report ===\n\n")
    f.write(report)
print("\n" + report)

# B. Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.title("Confusion Matrix")
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))

# C. Multi-class ROC Curve
plt.figure(figsize=(8, 6))
y_true_bin = label_binarize(y_true, classes=[0,1,2,3])
colors = ['blue', 'red', 'green', 'orange']

for i, color in zip(range(NUM_CLASSES), colors):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{CLASSES[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-Class Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "roc_auc_curve.png"))

print(f"\n✅ Evaluation complete! All artifacts saved in {MODEL_DIR}/")
