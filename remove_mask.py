import os
import shutil

dataset_path = r"D:\radiology\Dataset_BUSI_with_GT"
clean_path = r"D:\radiology\ultrasound_clean"

classes = ["benign", "malignant", "normal"]

for cls in classes:
    src_folder = os.path.join(dataset_path, cls)
    dst_folder = os.path.join(clean_path, cls)

    os.makedirs(dst_folder, exist_ok=True)

    for file in os.listdir(src_folder):

        if "_mask" not in file:   # ignore mask images
            src = os.path.join(src_folder, file)
            dst = os.path.join(dst_folder, file)
            shutil.copy(src, dst)

print("Masks removed successfully!")