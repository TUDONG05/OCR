import os

# Đường dẫn tự động theo vị trí file config.py (hoạt động cả Windows lẫn Linux)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR = os.path.join(_BASE_DIR, "iam-dataset")

TRAIN_LABELS = os.path.join(_DATA_DIR, "train_labels.txt")
VALID_LABELS = os.path.join(_DATA_DIR, "validation_labels.txt")
TEST_LABELS  = os.path.join(_DATA_DIR, "test_labels.txt")

CHECKPOINT_PATH = os.path.join(_BASE_DIR, "checkpoints", "best_model.keras")

IMG_WIDTH = 1024
IMG_HEIGHT = 72

# Tham số huấn luyện
BATCH_SIZE = 32        # Fix: batch=1 phá hỏng BatchNormalization
EPOCHS = 50
LEARNING_RATE = 1e-3   # OneCycle sẽ tự điều chỉnh xuống
MAX_LABEL_LENGTH = 128