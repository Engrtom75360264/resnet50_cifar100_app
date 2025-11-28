import os
from torchvision import datasets

# 1. Load CIFAR-100 test dataset
dataset = datasets.CIFAR100(root="./data", train=False, download=True)

# 2. Create folder to save all sample images
save_dir = "cifar100_samples"
os.makedirs(save_dir, exist_ok=True)

# 3. Track saved classes
saved = {cls: False for cls in dataset.classes}

# 4. Loop through dataset
for img, label in dataset:
    cls_name = dataset.classes[label]

    # Save only one sample per class
    if not saved[cls_name]:
        img.save(os.path.join(save_dir, f"{cls_name}.png"))
        saved[cls_name] = True

    # Stop when all 100 classes are saved
    if all(saved.values()):
        break

print("Finished saving all CIFAR-100 class samples!")
