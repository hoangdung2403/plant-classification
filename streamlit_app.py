import os
import json
import io
from typing import List, Tuple

import numpy as np
from PIL import Image
import streamlit as st

# Optional: use TensorFlow/Keras if your Kaggle model was saved via tf.keras
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
except Exception as e:  # pragma: no cover
    tf = None
    keras_load_model = None

############################
# ⚙️ App Config
############################
st.set_page_config(
    page_title="CNN Image Classifier",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 CNN Image Classifier — Streamlit UI")


############################
# 🔧 Sidebar Controls
############################
model_path = st.sidebar.text_input(
    "Đường dẫn mô hình (model.h5 / model_fixed.keras)",
    value=os.environ.get("MODEL_PATH", "model_fixed.keras"),
    help="Ví dụ: model.h5, models/cnn-v1.keras, ...",
)
labels_path = st.sidebar.text_input(
    "Đường dẫn labels.json (tùy chọn)",
    value=os.environ.get("LABELS_PATH", "labels.json"),
    help="Nội dung dạng danh sách tên lớp hoặc dict {class_index: class_name}",
)
conf_threshold = st.sidebar.slider("Ngưỡng hiển thị (confidence)", 0.0, 1.0, 0.0, 0.01)
show_top_k = st.sidebar.number_input("Hiển thị Top‑K", min_value=1, max_value=20, value=5, step=1)
use_gradcam = st.sidebar.checkbox("Hiển thị Grad‑CAM (thử nghiệm)", value=False)

st.sidebar.caption(
    ""
)

############################
# 📦 Utilities
############################

def load_class_names(path: str, num_classes: int | None = None) -> List[str]:
    """Load class names from a JSON file if available.
    Accepts list ["cat", "dog", ...] or dict {"0": "cat", "1": "dog"}.
    Fallback to ["class_0", ...].
    """
    if path and os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return [str(x) for x in data]
            if isinstance(data, dict):
                # sort by numeric key if possible
                items = sorted(data.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0])
                return [str(v) for _, v in items]
        except Exception as e:  # pragma: no cover
            st.warning(f"Không đọc được labels từ {path}: {e}")
    # fallback
    if num_classes is None:
        num_classes = 2
    return [f"class_{i}" for i in range(num_classes)]


@st.cache_resource(show_spinner=True)
def load_keras_model(path: str):
    if keras_load_model is None:
        raise RuntimeError("TensorFlow/Keras chưa được cài. Hãy cài tensorflow trước.")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy mô hình: {path}")
    model = keras_load_model(path, compile=False)
    # Try to infer input shape (H, W, C)
    try:
        input_shape = model.input_shape
        # handle nested input shapes
        if isinstance(input_shape, (list, tuple)) and isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            # pick the first input if multiple inputs (common in some Keras graphs)
            input_shape = input_shape[0]
        h, w = int(input_shape[1]), int(input_shape[2])
    except Exception:
        h, w = 224, 224
    return model, (h, w)


def preprocess_image(img: Image.Image, target_size: Tuple[int, int]) -> np.ndarray:
    img = img.convert("RGB").resize(target_size)
    x = np.asarray(img).astype("float32") / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def predict_proba(model, x: np.ndarray) -> np.ndarray:
    preds = model.predict(x, verbose=0)
    # if model outputs logits or a single scalar per class
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = np.expand_dims(preds, axis=0)
    # If binary output shape (N,1), convert to 2-class probs
    if preds.shape[-1] == 1:
        p1 = preds.squeeze(-1)
        preds = np.stack([1 - p1, p1], axis=-1)
    # Softmax normalize (safety)
    exps = np.exp(preds - np.max(preds, axis=-1, keepdims=True))
    probs = exps / np.clip(exps.sum(axis=-1, keepdims=True), 1e-8, None)
    return probs[0]


def get_topk(probs: np.ndarray, class_names: List[str], k: int) -> List[Tuple[str, float]]:
    k = min(k, len(probs))
    idx = np.argsort(probs)[::-1][:k]
    return [(class_names[i] if i < len(class_names) else f"class_{i}", float(probs[i])) for i in idx]


############################
# 🧩 Load model
############################
model = None
input_hw = (224, 224)
if model_path:
    try:
        model, input_hw = load_keras_model(model_path)
        st.success(f"Đã tải mô hình từ: {model_path} — input size suy đoán: {input_hw}")
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình: {e}")
        st.stop()

# Try to infer number of classes from the model output
num_classes = None
try:
    if hasattr(model, "output_shape"):
        out_shape = model.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        if isinstance(out_shape, tuple) and len(out_shape) >= 2:
            num_classes = int(out_shape[-1]) if out_shape[-1] and out_shape[-1] > 1 else 2
except Exception:
    num_classes = None

class_names = load_class_names(labels_path, num_classes)

############################
# 📤 Uploader & Camera
############################
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Chọn ảnh…", type=["jpg", "jpeg", "png", "bmp", "webp"])
with col2:
    camera_img = st.camera_input("Hoặc chụp ảnh từ webcam")

img_bytes = None
if uploaded is not None:
    img_bytes = uploaded.read()
elif camera_img is not None:
    img_bytes = camera_img.getvalue()

if img_bytes is None:
    st.info("Hãy tải hoặc chụp một ảnh để dự đoán.")
    st.stop()

# Display image
image = Image.open(io.BytesIO(img_bytes))
st.image(image, caption="Ảnh đã chọn", use_container_width=True)

############################
# 🔮 Inference
############################
with st.spinner("Đang dự đoán…"):
    x = preprocess_image(image, input_hw)
    probs = predict_proba(model, x)
    topk = [(name, p) for name, p in get_topk(probs, class_names, int(show_top_k)) if p >= conf_threshold]

if not topk:
    st.warning("Không có lớp nào vượt ngưỡng hiển thị. Giảm ngưỡng để xem kết quả.")
else:
    st.subheader("Kết quả dự đoán")
    for i, (name, p) in enumerate(topk, start=1):
        st.write(f"**{i}. {name}** — {p*100:.2f}%")

    # Bar chart
    try:
        import pandas as pd
        df = pd.DataFrame({"class": [n for n, _ in topk], "prob": [p for _, p in topk]}).set_index("class")
        st.bar_chart(df)
    except Exception:
        pass

############################
# 🔥 Grad‑CAM (experimental)
############################
if use_gradcam:
    st.markdown("---")
    st.subheader("Grad‑CAM (thử nghiệm)")
    if tf is None:
        st.error("Cần TensorFlow để chạy Grad‑CAM.")
    else:
        try:
            # Try to find the last conv layer automatically
            last_conv = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break
            if last_conv is None:
                raise ValueError("Không tìm thấy lớp Conv2D trong mô hình để vẽ Grad‑CAM.")

            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(last_conv).output, model.output]
            )
            img_tensor = tf.convert_to_tensor(x)
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_tensor)
                pred_index = int(np.argmax(predictions[0].numpy()))
                loss = predictions[:, pred_index]
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            heatmap = np.maximum(heatmap.numpy(), 0)
            if heatmap.max() > 0:
                heatmap /= heatmap.max()

            # Resize heatmap to original image size and overlay
            heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(image.size)
            heatmap_img = heatmap_img.convert("RGBA")
            # Colorize: map grayscale alpha
            # Create an RGBA where heat intensity controls alpha channel
            alpha = np.array(heatmap_img.getchannel("A"))
            alpha = (alpha / 255.0 * 180).astype(np.uint8)  # max ~70% opacity
            heat_rgba = np.zeros((heatmap_img.height, heatmap_img.width, 4), dtype=np.uint8)
            # Use red channel for intensity; keep others 0 for simplicity
            heat_rgba[..., 0] = np.array(heatmap_img)  # R
            heat_rgba[..., 3] = alpha                  # A
            overlay = Image.fromarray(heat_rgba, mode="RGBA")

            blended = image.convert("RGBA").copy()
            blended.alpha_composite(overlay)
            st.image(blended, caption=f"Grad‑CAM tại lớp '{last_conv}' (lớp dự đoán: {class_names[pred_index] if pred_index < len(class_names) else pred_index})", use_container_width=True)
        except Exception as e:
            st.warning(f"Không thể tạo Grad‑CAM: {e}")

st.markdown("---")
st.caption(
    "."
)
