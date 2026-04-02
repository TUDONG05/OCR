import os
import tensorflow as tf
import config, data_loader, model as model_builder


def main():
    # Setup GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    # KHÔNG dùng mixed precision — RTX 3050 Laptop không có AVX-512,
    # oneDNN fallback về CPU làm tốn thêm VRAM thay vì tiết kiệm.

    # Load Data
    print("Đang đọc dữ liệu...")
    if not os.path.exists(config.TRAIN_LABELS):
        print(f"Không tìm thấy file nhãn tại: {config.TRAIN_LABELS}")
        return

    train_img_paths, train_labels = data_loader.clean_labels(config.TRAIN_LABELS)
    valid_img_paths, valid_labels = data_loader.clean_labels(config.VALID_LABELS)
    print(f"Train: {len(train_img_paths)} | Val: {len(valid_img_paths)}")

    # Vocab & Dataset
    char_to_num, num_to_char = data_loader.build_vocabulary(train_labels + valid_labels)
    # training=True bật shuffle + augmentation
    train_ds = data_loader.prepare_dataset(train_img_paths, train_labels, char_to_num, training=True)
    valid_ds = data_loader.prepare_dataset(valid_img_paths, valid_labels, char_to_num, training=False)

    # Model
    model = model_builder.build_model(len(char_to_num.get_vocabulary()))
    model.summary()

    # Callbacks
    os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        config.CHECKPOINT_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )

    # Patience=10 — tránh dừng sớm khi loss vẫn đang cải thiện chậm
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    # Giảm LR khi val_loss không cải thiện sau 4 epochs
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    csv_logger = tf.keras.callbacks.CSVLogger("training_log.csv", append=True)

    # Train
    print("\nBắt đầu training...")
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=config.EPOCHS,
        callbacks=[ckpt, early, reduce_lr, csv_logger]
    )
    print(f"\nHoàn tất! Model tốt nhất lưu tại: {config.CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()