from PIL import Image
import os

dataset_path = r"D:\radiology\archive2\Bone_Fracture_Dataset"

bad_images = []

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            img.verify()
        except:
            print("Corrupted image:", path)
            bad_images.append(path)

for img in bad_images:
    os.remove(img)

print("Removed", len(bad_images), "corrupted images")