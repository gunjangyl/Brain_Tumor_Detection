import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

# Globals
MODEL_PATH = "model/EfficientNet_model.h5"
CLASSES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Load model lazily
_model = None

def get_model():
    global _model
    if _model is None:
        if os.path.exists(MODEL_PATH):
            try:
                base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
                x = base_model.output
                x = GlobalAveragePooling2D()(x)
                x = BatchNormalization()(x)
                x = Dense(512, activation="relu")(x)
                x = Dropout(0.5)(x)
                x = Dense(256, activation="relu")(x)
                x = Dropout(0.4)(x)
                output = Dense(4, activation="softmax")(x)
                _model = Model(inputs=base_model.input, outputs=output)
                _model.load_weights(MODEL_PATH)
                print(f"[INFO] Successfully loaded model weights from {MODEL_PATH}")
            except Exception as e:
                print(f"[FATAL ERROR] Failed to load model weights: {e}")
                return None
        else:
            print(f"[WARNING] Model weights not found at {MODEL_PATH}.")
            return None
    return _model

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    superimposed_img.save(cam_path)
    return cam_path

def check(img_path):
    """
    Predicts the tumor class and generates a Grad-CAM heatmap.
    Args:
        img_path (str): Absolute path to the uploaded image.
    """
    model = get_model()
    if model is None:
        return {
            "predicted_class": "Model Not Found",
            "confidence": "0%",
            "probabilities": {"Glioma": 25, "Meningioma": 25, "No Tumor": 25, "Pituitary": 25},
            "heatmap_path": "static/heatmaps/error.jpg"
        }

    # 1. Preprocess
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # 2. Predict
    preds = model.predict(img_array)[0]
    pred_idx = np.argmax(preds)
    pred_class = CLASSES[pred_idx]
    confidence = float(preds[pred_idx] * 100)
    
    # Probabilities dict for React
    probs_dict = {CLASSES[i]: round(float(preds[i]) * 100, 2) for i in range(len(CLASSES))}

    # 3. Grad-CAM
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break
            
    filename = os.path.basename(img_path)
    # Ensure static/heatmaps exists
    os.makedirs(os.path.join("static", "heatmaps"), exist_ok=True)
    heatmap_rel_path = os.path.join("static", "heatmaps", "cam_" + filename)
    heatmap_abs_path = os.path.abspath(heatmap_rel_path)

    if last_conv_layer_name:
        try:
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
            save_and_display_gradcam(img_path, heatmap, cam_path=heatmap_abs_path, alpha=0.5)
            heatmap_output = heatmap_rel_path.replace("\\", "/") # For URL consistency
        except Exception as e:
            print(f"[Grad-CAM Error] {e}")
            heatmap_output = img_path # fallback
    else:
        heatmap_output = img_path

    return {
        "predicted_class": pred_class,
        "confidence": f"{round(confidence, 2)}%",
        "probabilities": probs_dict,
        "heatmap_path": heatmap_output
    }
