
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from src import config, data_loader, model as model_builder

# 1. Cấu hình trang Web
st.set_page_config(page_title="Nhận dạng chữ viết tay", page_icon="✍️")
st.title("✍️ Ứng dụng Nhận dạng Chữ viết tay (OCR)")
st.write("Tải ảnh chứa dòng chữ viết tay lên để AI đọc nhé!")

# 2. Hàm Load Model (Dùng cache để không phải load lại mỗi lần f5)
@st.cache_resource
def load_ocr_model():
    # Load lại bộ từ điển
    train_labels = data_loader.clean_labels(config.TRAIN_LABELS)[1]
    valid_labels = data_loader.clean_labels(config.VALID_LABELS)[1]
    char_to_num, num_to_char = data_loader.build_vocabulary(train_labels + valid_labels)
    
    # Load model
    model_path = "checkpoints/best_model.keras"
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={"CTCLayer": model_builder.CTCLayer}
    )
    
    # Tách lấy phần dự đoán (giống predict.py)
    prediction_model = tf.keras.models.Model(
        inputs=model.inputs[0], 
        outputs=model.get_layer(name="dense_1").output
    )
    return prediction_model, num_to_char

# 3. Hàm giải mã kết quả
def decode_batch_predictions(pred, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :config.MAX_LABEL_LENGTH]
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        # Lọc bỏ ký tự lạ
        res = res.replace("[UNK]", "").strip() 
        output_text.append(res)
    return output_text

# 4. Hàm xử lý ảnh từ Upload
def process_uploaded_image(image):
    # Chuyển sang ảnh xám
    image = image.convert("L")
    # Chuyển thành mảng số
    image = tf.keras.preprocessing.image.img_to_array(image)
    # Resize theo đúng chuẩn training (Hàm này lấy từ data_loader)
    image = data_loader.distortion_free_resize(image, (config.IMG_WIDTH, config.IMG_HEIGHT))
    # Chuẩn hóa về 0-1
    image = tf.cast(image, tf.float32) / 255.0
    # Thêm chiều batch (1, 1024, 32, 1)
    image = tf.expand_dims(image, axis=0)
    return image

# --- GIAO DIỆN CHÍNH ---

try:
    # Load model ngay khi vào web
    with st.spinner("Đang khởi động AI... Đợi chút nhé!"):
        model, num_to_char = load_ocr_model()
    st.success("AI đã sẵn sàng!")

    # Nút upload file
    uploaded_file = st.file_uploader("Chọn ảnh (PNG, JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Hiển thị ảnh gốc
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh bạn vừa tải lên", use_container_width=True)

        # Nút bấm Dự đoán
        if st.button("Đọc chữ trong ảnh"):
            with st.spinner("AI đang đọc..."):
                # Xử lý ảnh
                processed_img = process_uploaded_image(image)
                
                # Dự đoán
                preds = model.predict(processed_img)
                pred_text = decode_batch_predictions(preds, num_to_char)[0]
                
                # Hiện kết quả to đẹp
                st.markdown("### 🎯 Kết quả:")
                st.code(pred_text, language="text")

except Exception as e:
    st.error(f"Có lỗi xảy ra: {e}")
    st.warning("Hãy chắc chắn bạn đã chạy train.py xong và có file 'checkpoints/best_model.keras' nhé!")