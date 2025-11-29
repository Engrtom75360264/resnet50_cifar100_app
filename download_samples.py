import os
from torchvision import datasets, transforms
from PIL import Image

# -------------------------------
# 1. Create folder
# -------------------------------
save_dir = "cifar100_samples"
os.makedirs(save_dir, exist_ok=True)

# -------------------------------
# 2. Load CIFAR-100 test dataset
# -------------------------------
dataset = datasets.CIFAR100(
    root="./data",
    train=False,
    download=True
)

# -------------------------------
# 3. Choose 10 classes to save
# -------------------------------
selected_classes = [
    "apple", "bear", "bee", "bicycle", "boy",
    "camel", "clock", "lion", "mountain", "tiger"
]

# Map class name → index
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(dataset.classes)}

# -------------------------------
# 4. Save one sample per class
# -------------------------------
saved = 0

for cls_name in selected_classes:
    cls_idx = class_to_idx[cls_name]

    # find the first image belonging to that class
    for i in range(len(dataset)):
        img, label = dataset[i]

        if label == cls_idx:
            # Convert to PIL image (dataset already returns PIL)
            img = img.convert("RGB")

            # Save with filename
            filename = f"{cls_name}.png"
            img.save(os.path.join(save_dir, filename))
            print(f"Saved: {filename}")

            saved += 1
            break

print(f"\nDone! Saved {saved} sample images to {save_dir}/")
