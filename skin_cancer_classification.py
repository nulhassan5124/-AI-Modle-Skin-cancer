# Skin Cancer Classification Pipeline
import os
import shutil
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# 1. Dataset preprocessing to 128x128
def preprocess_dataset(root_dir):
    processed_count = 0
    for root, dirs, files_list in os.walk(root_dir):
        for file in files_list:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        img_resized = img.resize((128, 128), Image.Resampling.LANCZOS)
                        target_path = os.path.splitext(file_path)[0] + ".jpg"
                        img_resized.save(target_path, "JPEG", quality=70)
                        if file_path != target_path:
                            os.remove(file_path)
                    processed_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    print(f"Processed {processed_count} images successfully.")

# 2. PyTorch Lightweight Model Architecture
class LightweightCNN(nn.Module):
    def __init__(self):
        super(LightweightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    print("Skin Cancer Classification Module loaded!")
