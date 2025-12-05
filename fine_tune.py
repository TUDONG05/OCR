
# import os
# import tensorflow as tf
# import config, data_loader, model as model_builder

# def main():
#     # 1. Setup GPU
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     if gpus:
#         try:
#             for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
#         except RuntimeError: pass

#     print("🔹 Đang tải dữ liệu để học tiếp...")
    
#     # 2. Load lại dữ liệu (như train.py)
#     train_img_paths, train_labels = data_loader.clean_labels(config.TRAIN_LABELS)
#     valid_img_paths, valid_labels = data_loader.clean_labels(config.VALID_LABELS)
#     char_to_num, num_to_char = data_loader.build_vocabulary(train_labels + valid_labels)
    
#     train_ds = data_loader.prepare_dataset(train_img_paths, train_labels, char_to_num)
#     valid_ds = data_loader.prepare_dataset(valid_img_paths, valid_labels, char_to_num)

#     # 3. QUAN TRỌNG: Load lại model cũ (Best Model)
#     model_path = "checkpoints/best_model.keras"
#     if not os.path.exists(model_path):
#         print(" Không tìm thấy model cũ! Bạn phải chạy train.py trước.")
#         return
    
#     print(f" Đang load model từ: {model_path}")
#     # Load model với custom object
#     model = tf.keras.models.load_model(model_path, custom_objects={"CTCLayer": model_builder.CTCLayer})

#     # 4. THIẾT LẬP HỌC CHẬM (Fine-tuning)
#     # Giảm Learning Rate xuống 10 lần (0.001 -> 0.0001) để học chi tiết
#     NEW_LEARNING_RATE = 0.00001
#     print(f" Thiết lập tốc độ học mới: {NEW_LEARNING_RATE}")
    
#     optimizer = tf.keras.optimizers.Adam(learning_rate=NEW_LEARNING_RATE, clipnorm=1.0)
#     model.compile(optimizer=optimizer)

#     # 5. Callbacks
#     checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
#         "checkpoints/best_model.keras", 
#         monitor="val_loss", 
#         save_best_only=True, 
#         mode="min",
#         verbose=1
#     )
#     early_stopping_cb = tf.keras.callbacks.EarlyStopping(
#         monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
#     )

#     # 6. Bắt đầu học tiếp thêm 50 Epochs nữa
#     print("\n Bắt đầu Fine-tuning...")
#     history = model.fit(
#         train_ds,
#         validation_data=valid_ds,
#         epochs=50, # Chạy thêm 50 vòng nữa
#         callbacks=[checkpoint_cb, early_stopping_cb]
#     )
#     print("\n Hoàn tất Fine-tuning!")

# if __name__ == "__main__":
#     main()



import os
import tensorflow as tf
import config, data_loader, model as model_builder


def main():

    # 1. GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    print("🔹 Đang load dữ liệu để Fine-tune...")

    # 2. Load dataset
    train_img_paths, train_labels = data_loader.clean_labels(config.TRAIN_LABELS)
    valid_img_paths, valid_labels = data_loader.clean_labels(config.VALID_LABELS)

    # Dùng lại từ điển CHUẨN
    char_to_num, num_to_char = data_loader.build_vocabulary(
        train_labels + valid_labels
    )

    train_ds = data_loader.prepare_dataset(train_img_paths, train_labels, char_to_num)
    valid_ds = data_loader.prepare_dataset(valid_img_paths, valid_labels, char_to_num)

    # 3. Load model cũ (như predict.py)
    model_path = r"/home/tudong/src/checkpoints/best_model.keras"

    if not os.path.exists(model_path):
        print("❌ Không tìm thấy model cũ!")
        return

    print(f"🔹 Load model từ: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "CTCLayer": model_builder.CTCLayer,
            "conv_block": model_builder.conv_block
        }
    )

    # 4. Fine-tune với LR nhỏ
    NEW_LR = 1e-5
    print(f"🔹 Learning Rate mới: {NEW_LR}")

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=NEW_LR, 
        clipnorm=1.0
    )
    model.compile(optimizer=optimizer)

    # 5. Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    # 6. Train tiếp
    print("\n BẮT ĐẦU FINE-TUNING...\n")

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=10,
        callbacks=[checkpoint_cb, early_stop_cb]
    )

    print("\n Fine-tuning hoàn tất!")


if __name__ == "__main__":
    main()
