import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def get_model(model_type):
    if model_type == "custom_cnn":
        return SimpleCNN()
    elif model_type == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
        return m
    elif model_type == "mobilenet_v2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
        return m
    else:
        raise ValueError(f"Неизвестная модель: {model_type}")

def train():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)["train"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = datasets.ImageFolder("data/processed/train", transform=transform)
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)

    model = get_model(cfg["model_type"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    model.train()
    for epoch in range(cfg["epochs"]):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/model.pt")
    print(f"Модель {cfg['model_type']} успешно обучена и сохранена в models/model.pt")

if __name__ == "__main__":
    train()