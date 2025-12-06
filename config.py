import os

TRAIN_LABELS = r"/home/tudong/src/iam-dataset/train_labels.txt"
VALID_LABELS = r"/home/tudong/src/iam-dataset/validation_labels.txt"
TEST_LABELS = r"/home/tudong/src/iam-dataset/test_labels.txt"


IMG_WIDTH = 1024
IMG_HEIGHT = 72

# Tham số huấn luyện
BATCH_SIZE = 1
EPOCHS = 20
LEARNING_RATE = 1e-4
MAX_LABEL_LENGTH = 128