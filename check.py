# check_files.py
import os

print("=== KIỂM TRA CÁC FILE CẦN THIẾT ===")

# Kiểm tra file labels
labels = [
    r"/home/tudong/src/iam-dataset/train_labels.txt",
    r"/home/tudong/src/iam-dataset/validation_labels.txt", 
    r"/home/tudong/src/iam-dataset/test_labels.txt"
]

for label_file in labels:
    exists = os.path.exists(label_file)
    print(f"{'' if exists else '❌'} {label_file}")
    if exists:
        print(f"   Kích thước: {os.path.getsize(label_file)} bytes")

print("\n=== KIỂM TRA MODEL ===")
model_path = r"/home/tudong/src/checkpoints/best_model.keras"
if os.path.exists(model_path):
    print(f" Model found: {model_path}")
    print(f"   Size: {os.path.getsize(model_path) / (1024*1024):.2f} MB")
else:
    print(f" Model NOT found: {model_path}")
    
print("\n=== KIỂM TRA THƯ MỤC ẢNH ===")
img_dir = r"/home/tudong/src/iam-dataset/test"
if os.path.exists(img_dir):
    files = os.listdir(img_dir)[:5]  # Lấy 5 file đầu
    print(f"Thư mục test tồn tại, có {len(os.listdir(img_dir))} files")
    print(f"   Ví dụ: {files}")
else:
    print(f"Thư mục test không tồn tại: {img_dir}")