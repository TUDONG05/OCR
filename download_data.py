from datasets import load_dataset
import os
import numpy as np
from PIL import Image


# 1. Load dataset


ds = load_dataset("Teklia/IAM-line")

# 2. Thư mục lưu output

output_root = r"/home/tudong/src/iam-dataset"
os.makedirs(output_root, exist_ok=True)


# 3. Hàm lưu dataset split

def export_split(split_name, split_data):
    split_dir = os.path.join(output_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    label_file = os.path.join(output_root, f"{split_name}_labels.txt")
    f = open(label_file, "w", encoding="utf-8")

    print(f"\n🔹 Exporting split: {split_name} ...")

    for i, item in enumerate(split_data):
        img_pil = item["image"]       # PIL image
        text = item["text"]           # label

        # filename
        fn = os.path.join(split_dir, f"{split_name}_{i}.png")

        # Save PNG
        img_pil.save(fn)

        # Write label
        f.write(f"{fn}\t{text}\n")

        if i % 500 == 0:
            print(f"  → saved {i} images...")

    f.close()
    print(f"✔ Done: {split_name}")

# ============================
# 4. Export từng split
# ============================
export_split("train", ds["train"])
export_split("validation", ds["validation"])
export_split("test", ds["test"])

print("\n🎉 All splits exported successfully!")
print(f" Folder: {output_root}")
