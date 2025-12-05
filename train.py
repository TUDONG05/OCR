import os
import tensorflow as tf
import config, data_loader, model as model_builder

def main():
    # Setup GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError: pass

    # Load Data
    print(f" Đang đọc dữ liệu...")
    if not os.path.exists(config.TRAIN_LABELS):
        print(f"Không tìm thấy file nhãn tại: {config.TRAIN_LABELS}")
        return

    train_img_paths, train_labels = data_loader.clean_labels(config.TRAIN_LABELS)
    valid_img_paths, valid_labels = data_loader.clean_labels(config.VALID_LABELS)
    print(f" Train: {len(train_img_paths)} | Val: {len(valid_img_paths)}")

    # Vocab & Dataset
    char_to_num, num_to_char = data_loader.build_vocabulary(train_labels + valid_labels)
    train_ds = data_loader.prepare_dataset(train_img_paths, train_labels, char_to_num)
    valid_ds = data_loader.prepare_dataset(valid_img_paths, valid_labels, char_to_num)

    # Model
    model = model_builder.build_model(len(char_to_num.get_vocabulary()))
    model.summary()

    # Callbacks
    os.makedirs("checkpoints", exist_ok=True)
    ckpt = tf.keras.callbacks.ModelCheckpoint("checkpoints/best_model.keras", monitor="val_loss", save_best_only=True, mode="min")
    early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # Train
    print("\n Bắt đầu training...")
    model.fit(train_ds, validation_data=valid_ds, epochs=config.EPOCHS, callbacks=[ckpt, early])
    print("\n Hoàn tất! Model lưu tại checkpoints/best_model.keras")

if __name__ == "__main__":
    main()