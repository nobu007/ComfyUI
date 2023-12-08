import numpy as np
import torch
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


# データセットの定義
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


# データセットの読み込み
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image_paths = ["image1.png", "image2.png", "image3.png"]
labels = [0, 1, 0]  # 0: 別画像, 1: 同一画像
dataset = ImageDataset(image_paths, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# VGG16モデルの定義
class VGG16BinaryClassifier(torch.nn.Module):
    def __init__(self):
        super(VGG16BinaryClassifier, self).__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)

    def forward(self, x):
        x = self.vgg16(x)
        return x


# モデルの初期化
model = VGG16BinaryClassifier()

# 損失関数と最適化器の定義
criterion = torch.nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# ファインチューニング
for epoch in range(10):  # 10エポック
    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# モデルの保存
torch.save(model.state_dict(), "vgg16_binary_classifier.pth")
