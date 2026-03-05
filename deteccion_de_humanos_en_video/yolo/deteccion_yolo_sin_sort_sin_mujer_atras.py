import torch
import cv2
import os
import glob
import json
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from ultralytics import YOLO

# ======================
# CONFIGURACIÓN
# ======================
MODEL_PATH = "best.pt" 
IMAGES_DIR = "cam_ta1_ws2/"
OUTPUT_VIDEO = "detecciones_yolo_best_sinSORT_sinMujerAtras.mp4"
CONF_THRESH = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLO_HUMAN_CLASS_ID = 0      # índice en YOLO
COCO_HUMAN_CATEGORY_ID = 1  # id real en el JSON


# ======================
# CARGAR MODELO YOLO
# ======================
model = YOLO(MODEL_PATH)

# ======================
# CARGAR GROUND TRUTH COCO
# ======================
GT_JSON = "ground_truth_sinMujerAtras.json"

with open(GT_JSON, "r") as f:
    coco = json.load(f)

# image_id -> filename
image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}

# filename -> GT boxes
gt_boxes_per_image = defaultdict(list)
for ann in coco["annotations"]:
    if ann["category_id"] != COCO_HUMAN_CATEGORY_ID:
        continue

    file_name = image_id_to_name[ann["image_id"]]
    x, y, w, h = ann["bbox"]
    gt_boxes_per_image[file_name].append([x, y, x+w, y+h])

for k in gt_boxes_per_image:
    gt_boxes_per_image[k] = np.array(gt_boxes_per_image[k])

# ======================
# FUNCIONES
# ======================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

# ======================
# PROCESAR IMÁGENES
# ======================
image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
first_frame = cv2.imread(image_paths[0])
height, width, _ = first_frame.shape

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (width, height)
)

total_TP, total_FP, total_FN = 0, 0, 0

for path in image_paths:
    frame = cv2.imread(path)
    frame_name = os.path.basename(path)

    # ======================
    # PREDICCIONES YOLO
    # ======================
    results = model.predict(frame, conf=CONF_THRESH, device=DEVICE)
    detections = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        for box, score, cls in zip(boxes, scores, classes):
            if int(cls) == YOLO_HUMAN_CLASS_ID:
                detections.append({"bbox": box, "score": score})
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    # ======================
    # GROUND TRUTH
    # ======================
    gt_boxes_frame = gt_boxes_per_image.get(frame_name, np.empty((0,4)))

    # ======================
    # MATCHING
    # ======================
    matched_gt = set()
    TP, FP = 0, 0

    for det in detections:
        ious = [compute_iou(det["bbox"], gt) for gt in gt_boxes_frame]
        if len(ious) == 0 or max(ious) < 0.5:
            FP += 1
        else:
            idx = np.argmax(ious)
            if idx not in matched_gt:
                TP += 1
                matched_gt.add(idx)
            else:
                FP += 1

    FN = len(gt_boxes_frame) - len(matched_gt)

    total_TP += TP
    total_FP += FP
    total_FN += FN

    out.write(frame)

out.release()
print(f"Vídeo guardado en {OUTPUT_VIDEO}")

# ======================
# MÉTRICAS
# ======================
precision = total_TP / (total_TP + total_FP + 1e-6)
recall = total_TP / (total_TP + total_FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("\n===== MATRIZ DE CONFUSIÓN GLOBAL =====")
print(f"TP: {total_TP}, FP: {total_FP}, FN: {total_FN}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# ======================
# MATRIZ DE CONFUSIÓN
# ======================
TN = 0
conf_matrix = np.array([
    [total_TP, total_FN],
    [total_FP, TN]
])

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicho Objeto", "Predicho No Objeto"],
            yticklabels=["GT Objeto", "GT No Objeto"])
plt.title("Matriz de Confusión — YOLO")
plt.xlabel("Predicción")
plt.ylabel("Ground Truth")
plt.tight_layout()
plt.savefig("confusion_matrix_yolo_best.png", dpi=300)
plt.close()
