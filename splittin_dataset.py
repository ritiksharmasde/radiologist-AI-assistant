import os
import random
import shutil

dataset_path = r"D:\radiology\ultrasound_clean"
output_path = "dataset3"

classes = ["benign", "malignant", "normal"]

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

for cls in classes:

    source = os.path.join(dataset_path, cls)
    images = os.listdir(source)
    random.shuffle(images)

    train_size = int(len(images) * train_ratio)
    val_size = int(len(images) * val_ratio)

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    for split, split_images in zip(
        ["train", "val", "test"],
        [train_images, val_images, test_images]
    ):

        split_folder = os.path.join(output_path, split, cls)
        os.makedirs(split_folder, exist_ok=True)

        for img in split_images:
            src = os.path.join(source, img)
            dst = os.path.join(split_folder, img)
            shutil.copy(src, dst)

print("Dataset successfully split!")