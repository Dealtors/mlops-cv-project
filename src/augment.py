import os
import shutil
import yaml
from PIL import Image
from torchvision import transforms

def augment_data():
    with open("params.yaml", "r") as f:
        cfg = yaml.safe_load(f)["augment"]

    train_dir = "data/processed/train"
    aug_dir = "data/augmented/train"

    # Удаляем старые данные аугментации, если они были
    if os.path.exists("data/augmented"):
        shutil.rmtree("data/augmented")

    os.makedirs(os.path.join(aug_dir, "food"), exist_ok=True)
    os.makedirs(os.path.join(aug_dir, "non_food"), exist_ok=True)

    # Преобразования аугментации
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=cfg["rotation_degrees"]),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])

    for cls in ["food", "non_food"]:
        src_cls_dir = os.path.join(train_dir, cls)
        dst_cls_dir = os.path.join(aug_dir, cls)

        files = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for fname in files:
            img_path = os.path.join(src_cls_dir, fname)
            img = Image.open(img_path).convert("RGB")

            # 1. Сохраняем исходное изображение
            img.save(os.path.join(dst_cls_dir, fname))

            # 2. Генерируем дополнительные модификации
            name_part, ext = os.path.splitext(fname)
            for i in range(cfg["num_copies"]):
                aug_img = augment_transform(img)
                aug_img.save(os.path.join(dst_cls_dir, f"{name_part}_aug{i}{ext}"))

    total_food = len(os.listdir(os.path.join(aug_dir, "food")))
    total_non_food = len(os.listdir(os.path.join(aug_dir, "non_food")))
    print(f"Аугментация завершена. Итого в обучающей выборке: food={total_food}, non_food={total_non_food}")

if __name__ == "__main__":
    augment_data()