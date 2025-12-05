import os

TRAIN_LABELS = r"/home/tudong/src/iam-dataset/train_labels.txt"
VALID_LABELS = r"/home/tudong/src/iam-dataset/validation_labels.txt"
TEST_LABELS = r"/home/tudong/src/iam-dataset/validation"


IMG_WIDTH = 1024
IMG_HEIGHT = 48

# Tham số huấn luyện
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 5e-5
MAX_LABEL_LENGTH = 128