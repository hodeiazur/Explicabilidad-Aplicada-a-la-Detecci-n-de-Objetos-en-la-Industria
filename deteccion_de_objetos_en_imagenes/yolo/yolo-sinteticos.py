from ultralytics import YOLO
import itertools
import os
import csv

# Ruta al modelo base
base_model = "yolov8s.pt"

# Archivo de datos
data_yaml = "dataset.yaml"

# Carpeta donde guardar resultados de cada experimento
os.makedirs("experiments", exist_ok=True)

# ──────────────────────────────
# Hiperparámetros a probar
# ──────────────────────────────
batch_sizes = [8, 16]
#batch_sizes = [16]
learning_rates = [0.001, 0.0005]
#learning_rates = [0.0005]
img_sizes = [640, 512]
epochs = [50]  

# ──────────────────────────────
# Generar todas las combinaciones
# ──────────────────────────────
param_combinations = list(itertools.product(batch_sizes, learning_rates, img_sizes, epochs))

# ──────────────────────────────
# Entrenamiento iterativo
# ──────────────────────────────
for i, (batch, lr, imgsz, ep) in enumerate(param_combinations):
    print(f"\n=== Experimento {i+1}/{len(param_combinations)} ===")
    print(f"Batch={batch}, LR={lr}, ImgSize={imgsz}, Epochs={ep}")

    exp_name = f"exp_bs{batch}_lr{lr}_img{imgsz}_ep{ep}"
    exp_path = os.path.join("experiments", exp_name)

    # Entrenar
    model = YOLO(base_model)
    model.train(
        data=data_yaml,
        epochs=ep,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        project="experiments",
        name=exp_name,
        exist_ok=True
    )

    # Evaluar en val y guardar resultados
    metrics = model.val()  # devuelve dict con mAP, precision, recall, etc.
    print(metrics.__dict__)

