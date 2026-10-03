import os
import shutil
import random
import yaml
from PIL import Image

def prepare_data():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["prepare"]

    raw_dir = "data/raw"
    processed_dir = "data/processed"
    test_size = params["test_size"]
    random_state = params["random_state"]
    random.seed(random_state)

    for split in ["train", "test"]:
        for cls in ["food", "non_food"]:
            os.makedirs(os.path.join(processed_dir, split, cls), exist_ok=True)

    for cls in ["food", "non_food"]:
        src_folder = os.path.join(raw_dir, cls)
        files = [f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(files)

        split_idx = int(len(files) * (1 - test_size))
        train_files = files[:split_idx]
        test_files = files[split_idx:]

        for fname in train_files:
            img = Image.open(os.path.join(src_folder, fname)).convert("RGB").resize((128, 128))
            img.save(os.path.join(processed_dir, "train", cls, fname))

        for fname in test_files:
            img = Image.open(os.path.join(src_folder, fname)).convert("RGB").resize((128, 128))
            img.save(os.path.join(processed_dir, "test", cls, fname))

    print("Предобработка завершена: данные разделены на train/test в data/processed.")

if __name__ == "__main__":
    prepare_data()