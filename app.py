
import time
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import keras
keras.config.enable_unsafe_deserialization()
import config, data_loader, model as model_builder

st.set_page_config(page_title="Nhận dạng chữ viết tay", page_icon="✍️", layout="wide")


@st.cache_resource
def load_ocr_model():
    train_labels = data_loader.clean_labels(config.TRAIN_LABELS)[1]
    valid_labels = data_loader.clean_labels(config.VALID_LABELS)[1]
    char_to_num, num_to_char = data_loader.build_vocabulary(train_labels + valid_labels)

    model = tf.keras.models.load_model(
        config.CHECKPOINT_PATH,
        custom_objects={"CTCLayer": model_builder.CTCLayer},
        compile=False
    )

    try:
        preds_layer = model.get_layer(name="predictions")
    except Exception:
        preds_layer = model.layers[-2]

    prediction_model = tf.keras.models.Model(
        inputs=model.inputs[0],
        outputs=preds_layer.output
    )
    return prediction_model, num_to_char


def decode_predictions(pred, num_to_char, beam_width=100):
    """Beam Search decode — chính xác hơn greedy."""
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(
        pred,
        input_length=input_len,
        greedy=False,
        beam_width=beam_width
    )[0][0][:, :config.MAX_LABEL_LENGTH]
    texts = []
    for r in results:
        text = tf.strings.reduce_join(num_to_char(r)).numpy().decode("utf-8")
        texts.append(text.replace("[UNK]", "").strip())
    return texts


def process_uploaded_image(image: Image.Image):
    """Áp dụng đúng preprocessing như lúc training."""
    image = image.convert("L")
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = data_loader.distortion_free_resize(img_array, (config.IMG_WIDTH, config.IMG_HEIGHT))
    img_array = tf.cast(img_array, tf.float32) / 255.0
    return tf.expand_dims(img_array, axis=0)


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Cấu hình")
    beam_width = st.slider("Beam width (cao hơn = chính xác hơn, chậm hơn)", 1, 200, 100, step=10)
    st.caption(f"Model: `{config.CHECKPOINT_PATH}`")
    st.caption(f"Input: {config.IMG_WIDTH}×{config.IMG_HEIGHT}px (grayscale)")

# ── Main ───────────────────────────────────────────────────────────────────
st.title("✍️ Nhận dạng chữ viết tay (OCR)")
st.write("Upload ảnh chứa **một dòng chữ viết tay** để AI nhận diện.")

try:
    with st.spinner("Đang khởi động model..."):
        prediction_model, num_to_char = load_ocr_model()
    st.success("Model sẵn sàng!", icon="✅")

    uploaded_file = st.file_uploader("Chọn ảnh (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ảnh gốc")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Ảnh sau tiền xử lý (32px height)")
            preview = image.convert("L").resize(
                (config.IMG_WIDTH, config.IMG_HEIGHT), Image.LANCZOS
            )
            st.image(preview, use_container_width=True, clamp=True)

        if st.button("🔍 Nhận diện chữ", type="primary"):
            with st.spinner("Đang phân tích..."):
                t0 = time.time()
                processed = process_uploaded_image(image)
                preds = prediction_model.predict(processed, verbose=0)
                pred_text = decode_predictions(preds, num_to_char, beam_width=beam_width)[0]
                elapsed = time.time() - t0

            st.markdown("### 📝 Kết quả:")
            st.code(pred_text if pred_text else "(không nhận diện được)", language="text")
            st.caption(f"Thời gian xử lý: {elapsed:.3f}s | Beam width: {beam_width}")

except FileNotFoundError as e:
    st.error(str(e))
    st.info("Hãy chạy `python train.py` trước để tạo file checkpoint.")
except Exception as e:
    st.error(f"Lỗi: {e}")