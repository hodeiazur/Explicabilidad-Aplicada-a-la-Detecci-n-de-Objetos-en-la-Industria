from ultralytics import YOLO
import itertools
import os
import csv

# ──────────────────────────────
# Configuración base
# ──────────────────────────────
BASE_MODEL = "yolov8s.pt"
DATA_YAML = "/global/home/TRI.LAN/cooperacion-111445/HUMAN_YOLO/human.yaml"


PROJECT_DIR = "experiments_yolo"
os.makedirs(PROJECT_DIR, exist_ok=True)

RESULTS_CSV = os.path.join(PROJECT_DIR, "results.csv")

# ──────────────────────────────
# Hiperparámetros a explorar
# ──────────────────────────────
batch_sizes = [8, 16]
learning_rates = [1e-3, 5e-4]
img_sizes = [640, 512]
epochs_list = [50]

param_combinations = list(itertools.product(
    batch_sizes, learning_rates, img_sizes, epochs_list
))

# ──────────────────────────────
# Preparar CSV
# ──────────────────────────────
if not os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "experiment",
            "batch",
            "lr",
            "imgsz",
            "epochs",
            "mAP50",
            "mAP50-95",
            "precision",
            "recall"
        ])

# ──────────────────────────────
# Grid search
# ──────────────────────────────
for i, (batch, lr, imgsz, epochs) in enumerate(param_combinations, 1):
    exp_name = f"bs{batch}_lr{lr}_img{imgsz}_ep{epochs}"
    print(f"\n Experimento {i}/{len(param_combinations)} → {exp_name}")

    model = YOLO(BASE_MODEL)

    # Entrenamiento
    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr,
        project=PROJECT_DIR,
        name=exp_name,
        exist_ok=True,
        device=0,
        patience=20,       # early stopping
        pretrained=True,
        verbose=False
    )

    # Evaluación
    metrics = model.val(
        data=DATA_YAML,
        imgsz=imgsz,
        batch=batch,
        device=0,
        verbose=False
    )

    # Extraer métricas
    mAP50 = metrics.box.map50
    mAP5095 = metrics.box.map
    precision = metrics.box.mp
    recall = metrics.box.mr

    # Guardar resultados
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            exp_name,
            batch,
            lr,
            imgsz,
            epochs,
            round(mAP50, 4),
            round(mAP5095, 4),
            round(precision, 4),
            round(recall, 4)
        ])

    print(
        f"✅ mAP50={mAP50:.3f} | "
        f"mAP50-95={mAP5095:.3f} | "
        f"P={precision:.3f} | R={recall:.3f}"
    )

print("\n🎉 Grid search finalizado. Resultados en:", RESULTS_CSV)
