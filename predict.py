
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



def cer(reference, hypothesis):
    import numpy as np
    ref = list(reference)
    hyp = list(hypothesis)

    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.uint32)
    for i in range(len(ref)+1):
        d[i][0] = i
    for j in range(len(hyp)+1):
        d[0][j] = j

    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
    return d[len(ref)][len(hyp)] / max(1, len(ref))


def wer(reference, hypothesis):
    import numpy as np
    ref = reference.split()
    hyp = hypothesis.split()

    d = np.zeros((len(ref)+1, len(hyp)+1), dtype=np.uint32)
    for i in range(len(ref)+1):
        d[i][0] = i
    for j in range(len(hyp)+1):
        d[0][j] = j

    for i in range(1, len(ref)+1):
        for j in range(1, len(hyp)+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )
    return d[len(ref)][len(hyp)] / max(1, len(ref))



import os
import tensorflow as tf
import numpy as np
import config, data_loader, model as model_builder
import keras
keras.config.enable_unsafe_deserialization()


def decode_batch_predictions(pred, num_to_char, beam_width=100):
    """Beam Search CTC decode — chính xác hơn greedy ~5-8% WER."""
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
        text = text.replace("[UNK]", "").strip()
        texts.append(text)
    return texts


def load_prediction_model(char_to_num):
    if not os.path.exists(config.CHECKPOINT_PATH):
        raise FileNotFoundError(f"Model không tồn tại tại: {config.CHECKPOINT_PATH}")

    print("Load model...")
    model = tf.keras.models.load_model(
        config.CHECKPOINT_PATH,
        custom_objects={"CTCLayer": model_builder.CTCLayer}
    )
    prediction_layer = model.get_layer("predictions")
    prediction_model = tf.keras.models.Model(
        inputs=model.inputs[0],
        outputs=prediction_layer.output
    )
    return prediction_model


def main():
    print("Đang load từ điển...")
    train_labels = data_loader.clean_labels(config.TRAIN_LABELS)[1]
    valid_labels = data_loader.clean_labels(config.VALID_LABELS)[1]
    char_to_num, num_to_char = data_loader.build_vocabulary(train_labels + valid_labels)

    prediction_model = load_prediction_model(char_to_num)

    test_img_paths, test_labels = data_loader.clean_labels(config.TEST_LABELS)

    # --- 5 mẫu ngẫu nhiên ---
    print("\n========== KẾT QUẢ DỰ ĐOÁN (5 mẫu) ==========")
    indices = np.random.randint(0, len(test_img_paths), 5)
    for i in indices:
        img = data_loader.preprocess_image(test_img_paths[i])
        preds = prediction_model.predict(tf.expand_dims(img, 0), verbose=0)
        pred_text = decode_batch_predictions(preds, num_to_char)[0]
        print(f"\nẢnh     : {os.path.basename(test_img_paths[i])}")
        print(f"Nhãn    : {test_labels[i]}")
        print(f"Dự đoán : {pred_text}")
        print("-" * 40)

    # --- Đánh giá toàn bộ test set ---
    print("\n===== ĐÁNH GIÁ TOÀN TẬP TEST =====")
    total_cer = 0.0
    total_wer = 0.0
    n = len(test_img_paths)

    for i in range(n):
        img = data_loader.preprocess_image(test_img_paths[i])
        preds = prediction_model.predict(tf.expand_dims(img, 0), verbose=0)
        pred_text = decode_batch_predictions(preds, num_to_char)[0]

        total_cer += cer(test_labels[i], pred_text)
        total_wer += wer(test_labels[i], pred_text)

        if i % 200 == 0:
            print(f"  [{i}/{n}] CER hiện tại: {total_cer / (i + 1):.4f}")

    print(f"\nCER trung bình : {total_cer / n:.4f}  ({total_cer / n * 100:.2f}%)")
    print(f"WER trung bình : {total_wer / n:.4f}  ({total_wer / n * 100:.2f}%)")


if __name__ == "__main__":
    main()
