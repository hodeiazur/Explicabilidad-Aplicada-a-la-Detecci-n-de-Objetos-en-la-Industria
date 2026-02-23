import os
import glob
import pandas as pd
from ultralytics import YOLO

# ──────────────────────────────
# Rutas
# ──────────────────────────────
EXPERIMENTS_DIR = "/global/home/TRI.LAN/cooperacion-111445/HUMAN_YOLO/runs/detect/experiments_yolo"
OUTPUT_CSV = os.path.join(EXPERIMENTS_DIR, "resumen_experimentos.csv")
DATA_YAML = "/global/home/TRI.LAN/cooperacion-111445/HUMAN_YOLO/human.yaml"

# ──────────────────────────────
# Lista de resultados
# ──────────────────────────────
results = []

# Buscar todas las carpetas de experimentos
experiment_folders = sorted(glob.glob(os.path.join(EXPERIMENTS_DIR, "*")))

for folder in experiment_folders:
    folder_name = os.path.basename(folder)
    best_model = os.path.join(folder, "weights", "best.pt")
    
    if not os.path.exists(best_model):
        print(f"⚠️  No se encontró best.pt en {folder_name}, saltando...")
        continue
    
    # Extraer hiperparámetros del nombre de la carpeta
    # Formato esperado: bs{batch}_lr{lr}_img{imgsz}_ep{epochs}
    try:
        parts = folder_name.split("_")
        batch = int(parts[0].replace("bs", ""))
        lr = float(parts[1].replace("lr", ""))
        imgsz = int(parts[2].replace("img", ""))
        epochs = int(parts[3].replace("ep", ""))
    except Exception as e:
        print(f"⚠️  No se pudo parsear {folder_name}: {e}")
        continue

    # Cargar modelo
    model = YOLO(best_model)

    # Evaluar sobre el dataset completo
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
    results.append({
        "experiment": folder_name,
        "batch": batch,
        "lr": lr,
        "imgsz": imgsz,
        "epochs": epochs,
        "mAP50": round(mAP50, 4),
        "mAP50-95": round(mAP5095, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4)
    })

# ──────────────────────────────
# Crear DataFrame
# ──────────────────────────────
df = pd.DataFrame(results)

# Marcar el mejor experimento según mAP50
df["best"] = False
if not df.empty:
    best_idx = df["mAP50"].idxmax()
    df.at[best_idx, "best"] = True

# Guardar CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ CSV generado: {OUTPUT_CSV}")
print(df.sort_values("mAP50", ascending=False))
