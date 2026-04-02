
import os
import tensorflow as tf
import config, data_loader, model as model_builder


# Số lớp CNN cần freeze: 6 Conv + 3 MaxPool + BN layers = ~20 layers đầu
# Giữ CNN features đã học, chỉ train RNN head với data mới
CNN_FREEZE_UNTIL = "max_pooling2d_2"   # tên layer MaxPool cuối của Block 3


def main():
    # 1. GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    print("Đang load dữ liệu để Fine-tune...")

    # 2. Load dataset (augmentation bật cho train)
    train_img_paths, train_labels = data_loader.clean_labels(config.TRAIN_LABELS)
    valid_img_paths, valid_labels = data_loader.clean_labels(config.VALID_LABELS)
    char_to_num, num_to_char = data_loader.build_vocabulary(train_labels + valid_labels)
    train_ds = data_loader.prepare_dataset(train_img_paths, train_labels, char_to_num, training=True)
    valid_ds = data_loader.prepare_dataset(valid_img_paths, valid_labels, char_to_num, training=False)

    # 3. Load model
    if not os.path.exists(config.CHECKPOINT_PATH):
        print(f"Không tìm thấy model tại: {config.CHECKPOINT_PATH}")
        return

    print(f"Load model từ: {config.CHECKPOINT_PATH}")
    model = tf.keras.models.load_model(
        config.CHECKPOINT_PATH,
        custom_objects={"CTCLayer": model_builder.CTCLayer}
    )

    # 4. Freeze CNN layers — chỉ train Dense + BiLSTM + output head
    freeze = True
    for layer in model.layers:
        if freeze:
            layer.trainable = False
        if layer.name == CNN_FREEZE_UNTIL:
            freeze = False  # các layer sau MaxPool Block3 sẽ được train

    frozen = sum(1 for l in model.layers if not l.trainable)
    trainable = sum(1 for l in model.layers if l.trainable)
    print(f"Frozen layers: {frozen} | Trainable layers: {trainable}")

    # 5. Compile với LR nhỏ
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
    )

    # 6. Callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        config.CHECKPOINT_PATH,
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    )
    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    print("\nBẮT ĐẦU FINE-TUNING...\n")
    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=15,
        callbacks=[checkpoint_cb, early_stop_cb, reduce_lr]
    )
    print("\nFine-tuning hoàn tất!")


if __name__ == "__main__":
    main()
