import json
import yaml
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from train import get_model

def evaluate():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)["train"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_data = datasets.ImageFolder("data/processed/test", transform=transform)
    test_loader = DataLoader(test_data, batch_size=cfg["batch_size"], shuffle=False)

    model = get_model(cfg["model_type"])
    model.load_state_dict(torch.load("models/model.pt"))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision_score(all_labels, all_preds, average="binary", zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, average="binary", zero_division=0)),
        "f1_score": float(f1_score(all_labels, all_preds, average="binary", zero_division=0))
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("Метрики успешно рассчитаны и сохранены в metrics.json:", metrics)

if __name__ == "__main__":
    evaluate()