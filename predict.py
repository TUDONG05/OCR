
# import os
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import config, data_loader, model as model_builder
# import keras
# keras.config.enable_unsafe_deserialization()

# def decode_batch_predictions(pred, num_to_char):
#     input_len = np.ones(pred.shape[0]) * pred.shape[1]
#     # Greedy search: Chọn ký tự có xác suất cao nhất tại mỗi bước
#     results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :config.MAX_LABEL_LENGTH]
    
#     output_text = []
#     for res in results:
#         # Chuyển số về lại ký tự
#         res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
#         res = res.replace("[UNK]", "").strip() 
#         output_text.append(res)
#     return output_text

# def main():
#     # 1. Load lại từ điển (phải khớp với lúc train)
#     print("🔹 Đang tải cấu hình và từ điển...")
#     train_labels = data_loader.clean_labels(config.TRAIN_LABELS)[1]
#     valid_labels = data_loader.clean_labels(config.VALID_LABELS)[1]
#     char_to_num, num_to_char = data_loader.build_vocabulary(train_labels + valid_labels)
    
#     # 2. Load Model
#     model_path = r"C:\Users\khact\Desktop\AI\src\checkpoints\best_model.keras"
    
#     if not os.path.exists(model_path):
#         print("Chưa tìm thấy file model! Hãy đợi train.py chạy xong Epoch 1 đã nhé.")
#         return

#     # Load model với custom layer CTCLayer
#     model = tf.keras.models.load_model(model_path, custom_objects={"CTCLayer": model_builder.CTCLayer})
#     print(f"Đã load model từ: {model_path}")

#     # Tách phần dự đoán ra khỏi phần tính Loss
#     prediction_model = tf.keras.models.Model(
#         inputs=model.inputs[0], 
#         outputs=model.get_layer(name="dense_1").output
#     )
#     # 3. Lấy ngẫu nhiên 5 ảnh trong tập Test để thử
#     test_img_paths, test_labels = data_loader.clean_labels(config.TEST_LABELS)
#     indices = np.random.randint(0, len(test_img_paths), 5)
    
#     print("\n" + "="*40)
#     print("KẾT QUẢ DỰ ĐOÁN")
#     print("="*40)

#     for i in indices:
#         img_path = test_img_paths[i]
#         true_label = test_labels[i]
        
#         # Xử lý ảnh đầu vào
#         img = data_loader.preprocess_image(img_path)
#         img_batch = tf.expand_dims(img, axis=0) # Tạo batch size = 1

#         # Dự đoán
#         preds = prediction_model.predict(img_batch, verbose=0)
#         pred_text = decode_batch_predictions(preds, num_to_char)[0]
        
#         print(f"Ảnh: {os.path.basename(img_path)}")
#         print(f"Nhãn đúng:  {true_label}")
#         print(f"Mô hình đọc là:  {pred_text}")
        
#         # Đánh giá đơn giản
#         if true_label == pred_text:
#             print("CHÍNH XÁC TUYỆT ĐỐI")
#         else:
#             print("Có sai sót nhỏ")
#         print("-" * 30)

# if __name__ == "__main__":
#     main()


import os
import tensorflow as tf
import numpy as np
import config, data_loader, model as model_builder
import keras
keras.config.enable_unsafe_deserialization()


def decode_batch_predictions(pred, num_to_char):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    
    results = tf.keras.backend.ctc_decode(
        pred, 
        input_length=input_len, 
        greedy=True
    )[0][0][:, :config.MAX_LABEL_LENGTH]

    texts = []
    for r in results:
        text = tf.strings.reduce_join(num_to_char(r)).numpy().decode("utf-8")
        text = text.replace("[UNK]", "").strip()
        texts.append(text)
    return texts


def main():
    print("🔹 Đang load từ điển...")
    train_labels = data_loader.clean_labels(config.TRAIN_LABELS)[1]
    valid_labels = data_loader.clean_labels(config.VALID_LABELS)[1]
    char_to_num, num_to_char = data_loader.build_vocabulary(train_labels + valid_labels)

    model_path = r"/home/tudong/src/checkpoints/best_model.keras"

    if not os.path.exists(model_path):
        print(" Model không tồn tại!")
        return

    print(" Load model...")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"CTCLayer": model_builder.CTCLayer}
    )

    prediction_layer = model.get_layer("predictions")
    prediction_model = tf.keras.models.Model(
        inputs=model.inputs[0],
        outputs=prediction_layer.output
    )

    test_img_paths, test_labels = data_loader.clean_labels(config.TEST_LABELS)
    indices = np.random.randint(0, len(test_img_paths), 5)

    print("\n========== KẾT QUẢ DỰ ĐOÁN ==========")

    for i in indices:
        img_path = test_img_paths[i]
        true = test_labels[i]

        img = data_loader.preprocess_image(img_path)
        img_batch = tf.expand_dims(img, axis=0)

        preds = prediction_model.predict(img_batch, verbose=0)
        pred_text = decode_batch_predictions(preds, num_to_char)[0]

        print(f"\nẢnh: {os.path.basename(img_path)}")
        print(f"Nhãn đúng: {true}")
        print(f"Model đọc: {pred_text}")

        print("✔ Chính xác" if pred_text == true else "⚠ Sai")
        print("----------------------------------")


if __name__ == "__main__":
    main()
