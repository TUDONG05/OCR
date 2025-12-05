import tensorflow as tf
from tensorflow.keras.layers import StringLookup
import config

def clean_labels(labels_filepath):
    image_paths = []
    labels = []
    with open(labels_filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split("\t")
            if len(parts) == 2:
                image_paths.append(parts[0])
                labels.append(parts[1])
    return image_paths, labels

def build_vocabulary(all_labels):
    characters = set(char for label in all_labels for char in label)
    vocab = sorted(list(characters))
    # mask_token=None means index 0 is reserved for padding, vocab starts at 1
    char_to_num = StringLookup(vocabulary=vocab, mask_token=None)
    num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
    return char_to_num, num_to_char

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]
    
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(image, paddings=[[pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0]])
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def preprocess_image(image_path, img_size=(config.IMG_WIDTH, config.IMG_HEIGHT)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image

def process_images_labels(image_path, label, char_to_num):
    image = preprocess_image(image_path)
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    return {"image": image, "label": label}

def prepare_dataset(image_paths, labels, char_to_num):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: process_images_labels(x, y, char_to_num), num_parallel_calls=tf.data.AUTOTUNE)
    
    # --- PHẦN ĐÃ SỬA ---
    # Sử dụng padded_batch thay vì batch thường
    # padded_shapes: Image cố định kích thước, Label thì để None (tự động co giãn)
    dataset = dataset.padded_batch(
        config.BATCH_SIZE,
        padded_shapes={
            "image": [config.IMG_WIDTH, config.IMG_HEIGHT, 1],
            "label": [None] 
        },
        padding_values={
            "image": 0.0,
            "label": tf.cast(0, tf.int64) # Điền số 0 vào chỗ trống của nhãn
        }
    ).prefetch(buffer_size=tf.data.AUTOTUNE)
    # -------------------
    
    return dataset