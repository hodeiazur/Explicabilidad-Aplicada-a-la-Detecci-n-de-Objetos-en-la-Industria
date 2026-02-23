import os
import numpy as np
from PIL import Image

DATASET_ROOT = "."

# -----------------------------
# NUEVA DISTRIBUCIÓN
# -----------------------------
TRAIN_FOLDERS = [f"_out_sdrec_{i:02d}" for i in range(1, 14)]  # 01–13
VAL_FOLDER = "_out_sdrec_14"                                   # 14
TEST_FOLDER = "_out_sdrec_15"                                  # 15 (EXCLUIDA)
# -----------------------------

OUT = "dataset_yolo"

# Crear carpetas
for split in ["train", "val", "test"]:
    os.makedirs(f"{OUT}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{split}", exist_ok=True)


def save_yolo(data, W, H, out_file):
    with open(out_file, "w") as f:
        for obj in data:
            x1, y1 = obj["x_min"], obj["y_min"]
            x2, y2 = obj["x_max"], obj["y_max"]
            cls = int(obj["semanticId"])

            xc = (x1 + x2) / 2 / W
            yc = (y1 + y2) / 2 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            f.write(f"{cls} {xc} {yc} {w} {h}\n")


def convert_folder(folders, split):
    for folder in (folders if isinstance(folders, list) else [folders]):
        folder_path = os.path.join(DATASET_ROOT, folder)
        rgb_files = [f for f in os.listdir(folder_path) if f.startswith("rgb_")]

        for rgb in rgb_files:
            base = rgb.split("_")[-1].split(".")[0]
            box_file = os.path.join(folder_path, f"bounding_box_2d_tight_{base}.npy")

            if not os.path.exists(box_file):
                continue

            img_path = os.path.join(folder_path, rgb)
            img = Image.open(img_path)
            W, H = img.size
            data = np.load(box_file, allow_pickle=True)

            out_img = f"{OUT}/images/{split}/{folder}_{rgb}"
            out_lbl = f"{OUT}/labels/{split}/{folder}_{rgb.replace('.png', '.txt')}"

            img.save(out_img)
            save_yolo(data, W, H, out_lbl)

        print(f"Procesada carpeta {folder} → {split.upper()}")


# Procesar TRAIN/VAL/TEST
print("Convirtiendo TRAIN...")
convert_folder(TRAIN_FOLDERS, "train")

print("Convirtiendo VAL...")
convert_folder(VAL_FOLDER, "val")

print("Convirtiendo TEST (carpeta excluida)...")
convert_folder(TEST_FOLDER, "test")

print("Terminado.")
