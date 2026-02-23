import os
import csv
from ultralytics import YOLO

EXPERIMENTS_DIR = "experiments"
OUTPUT_CSV = "comparativa_validacion_buena.csv"

# Crear lista para filas
rows = []

# Cabecera CSV
header = [
    "experiment",
    "mAP50",
    "mAP50-95",
    "precision",
    "recall",
   # "box_loss",
    # "cls_loss",
    # "dfl_loss"
]

# Recorrer subcarpetas dentro de experiments/
for exp in os.listdir(EXPERIMENTS_DIR):
    exp_path = os.path.join(EXPERIMENTS_DIR, exp)
    weights_path = os.path.join(exp_path, "weights", "best.pt")

    if os.path.isfile(weights_path):
        print(f"\n Validando modelo: {weights_path}")

        model = YOLO(weights_path)
        metrics = model.val()

        row = [
            exp,
            metrics.box.map50,         # mAP50
            metrics.box.map,           # mAP50-95
            metrics.box.mp,            # precision
            metrics.box.mr,            # recall
            # metrics.loss.box_loss,     # box loss
            # metrics.loss.cls_loss,     # cls loss
            # metrics.loss.dfl_loss      # dfl loss
        ]

        rows.append(row)

# Guardar CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\nResultados guardados en: {OUTPUT_CSV}")
