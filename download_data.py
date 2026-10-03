import os
import shutil
import kagglehub

DATA_RAW_DIR = "data/raw"
FOOD_DIR = os.path.join(DATA_RAW_DIR, "food")
NON_FOOD_DIR = os.path.join(DATA_RAW_DIR, "non_food")

os.makedirs(FOOD_DIR, exist_ok=True)
os.makedirs(NON_FOOD_DIR, exist_ok=True)

# kagglehub не будет качать заново — файлы уже лежат в кэше
dataset_path = kagglehub.dataset_download("trolukovich/food5k-image-dataset")
print(f"Путь к кэшу датасета: {dataset_path}")

copied_food = 0
copied_non_food = 0
LIMIT_PER_CLASS = 100

for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        full_path = os.path.join(root, file)
        root_lower = root.lower()
        file_lower = file.lower()

        # Определение класса по имени файла или папке
        is_food = (
                file_lower.startswith("1_")
                or "food" in root_lower and "non" not in root_lower
                or "food" in file_lower and "non" not in file_lower
        )
        is_non_food = (
                file_lower.startswith("0_")
                or "non_food" in root_lower
                or "non-food" in root_lower
                or "non_food" in file_lower
                or "nonfood" in root_lower
        )

        if is_food and not is_non_food and copied_food < LIMIT_PER_CLASS:
            shutil.copy2(full_path, os.path.join(FOOD_DIR, f"food_{copied_food:03d}.jpg"))
            copied_food += 1
        elif is_non_food and copied_non_food < LIMIT_PER_CLASS:
            shutil.copy2(full_path, os.path.join(NON_FOOD_DIR, f"non_food_{copied_non_food:03d}.jpg"))
            copied_non_food += 1

        if copied_food >= LIMIT_PER_CLASS and copied_non_food >= LIMIT_PER_CLASS:
            break

print(f"Готово! Скопировано:")
print(f"- Еда (food): {copied_food} шт.")
print(f"- Не еда (non_food): {copied_non_food} шт.")