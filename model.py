
import tensorflow as tf
from tensorflow.keras import layers, models
import config

def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    return tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)

class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        label_length = tf.math.count_nonzero(y_true, axis=-1, keepdims=True, dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)

        self.add_loss(loss)
        return y_pred

    def get_config(self):
        return super().get_config()


def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def build_model(vocab_size):
    input_img = layers.Input(
        shape=(config.IMG_WIDTH, config.IMG_HEIGHT, 1),
        name="image",
        dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    x = input_img

    # -------------------------------
    #  6 LỚP CNN 
    # -------------------------------

    # Block 1
    x = conv_block(x, 64)
    x = conv_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = conv_block(x, 128)
    x = conv_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = conv_block(x, 256)
    x = conv_block(x, 256)
    x = layers.MaxPooling2D((2, 1))(x)

    # -------------------------------
    #  RESHAPE → TIME STEPS
    # -------------------------------
    shape = x.shape #[B,H,W,C]
    new_w = shape[1]
    new_features = shape[2] * shape[3] 
    x = layers.Reshape((new_w, new_features))(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)

    # -------------------------------
    #  BI-LSTM (256 → 128)
    # -------------------------------
    x = layers.Bidirectional(layers.LSTM(256, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)

    # -------------------------------
    #  OUTPUT
    # -------------------------------
    x = layers.Dense(vocab_size + 1, activation="softmax", name="predictions")(x)

    output = CTCLayer(name="ctc_loss")(labels, x)

    model = models.Model(inputs=[input_img, labels], outputs=output)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.LEARNING_RATE,
        clipnorm=1.0
    )
    model.compile(optimizer=optimizer)

    return model
