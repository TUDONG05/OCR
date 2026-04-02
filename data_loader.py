import tensorflow as tf
from tensorflow.keras.layers import StringLookup
import config


def _remap_image_path(raw_path: str) -> str:
    """Chuyển đường dẫn cũ trong label file → đường dẫn thực tế.

    Label files được tạo bởi download_data.py với đường dẫn tuyệt đối
    của máy gốc (vd: /home/tudong/src/iam-dataset/train/train_0.png).
    Hàm này trích xuất 2 thành phần cuối (split/filename) rồi ghép
    với _DATA_DIR hiện tại → hoạt động dù chạy ở đâu.
    """
    import os
    parts = raw_path.replace("\\", "/").split("/")
    # 2 thành phần cuối: "train/train_0.png" hoặc "validation/validation_0.png"
    relative = os.path.join(parts[-2], parts[-1])
    return os.path.join(config._DATA_DIR, relative)


def clean_labels(labels_filepath):
    image_paths = []
    labels = []
    with open(labels_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                image_paths.append(_remap_image_path(parts[0]))
                labels.append(parts[1])
    return image_paths, labels


def build_vocabulary(all_labels):
    characters = set(char for label in all_labels for char in label)
    vocab = sorted(list(characters))
    char_to_num = StringLookup(vocabulary=vocab, mask_token=None)
    num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
    return char_to_num, num_to_char


def distortion_free_resize(image, img_size):
    """Resize giữ aspect ratio, pad đều 2 bên, transpose để width → time steps.
    KHÔNG flip — flip ngang làm mất khả năng generalize với ảnh thật.
    """
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    pad_height = h - tf.shape(image)[0]
    pad_width  = w - tf.shape(image)[1]

    pad_height_top    = pad_height // 2
    pad_height_bottom = pad_height - pad_height_top
    pad_width_left    = pad_width  // 2
    pad_width_right   = pad_width  - pad_width_left

    image = tf.pad(
        image,
        paddings=[[pad_height_top, pad_height_bottom],
                  [pad_width_left, pad_width_right],
                  [0, 0]]
    )
    # Transpose (H,W,C) → (W,H,C) để chiều rộng trở thành time steps cho RNN
    image = tf.transpose(image, perm=[1, 0, 2])
    return image


def augment_image(image):
    """Augmentation chỉ dùng khi training.
    Mô phỏng biến thể thực tế của chữ viết tay:
      - Độ sáng/tương phản thay đổi (ảnh chụp điện thoại)
      - Nhiễu Gaussian nhẹ (scan kém chất lượng)
      - Xoay nhỏ ±3° (chữ hơi nghiêng)
    """
    # Random brightness & contrast
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)

    # Gaussian noise nhẹ
    noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.03)
    image = image + noise

    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def preprocess_image(image_path, img_size=(config.IMG_WIDTH, config.IMG_HEIGHT)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def process_images_labels(image_path, label, char_to_num, training=False):
    image = preprocess_image(image_path)
    if training:
        image = augment_image(image)
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels, char_to_num, training=False):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if training:
        dataset = dataset.shuffle(buffer_size=min(len(image_paths), 5000), seed=42)

    dataset = dataset.map(
        lambda x, y: process_images_labels(x, y, char_to_num, training=training),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.padded_batch(
        config.BATCH_SIZE,
        padded_shapes={
            "image": [config.IMG_WIDTH, config.IMG_HEIGHT, 1],
            "label": [None]
        },
        padding_values={
            "image": 0.0,
            "label": tf.cast(0, tf.int64)
        }
    ).prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset