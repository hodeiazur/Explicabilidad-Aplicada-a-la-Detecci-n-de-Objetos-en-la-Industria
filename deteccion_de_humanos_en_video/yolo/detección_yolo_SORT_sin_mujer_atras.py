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
from sklearn.metrics import roc_curve, auc

from ultralytics import YOLO
from sort import Sort

# ======================
# CONFIGURACIÓN
# ======================
MODEL_PATH = "best.pt"
IMAGES_DIR = "cam_ta1_ws2/"
OUTPUT_VIDEO = "detecciones_yolo_SORT.mp4"
CONF_THRESH = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

YOLO_HUMAN_CLASS_ID = 0
COCO_HUMAN_CATEGORY_ID = 1

# ======================
# MODELO YOLO
# ======================
model = YOLO(MODEL_PATH)

# ======================
# TRACKER SORT
# ======================
tracker = Sort(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
)

# ======================
# GROUND TRUTH COCO
# ======================
GT_JSON = "ground_truth_sinMujerAtras.json"

with open(GT_JSON, "r") as f:
    coco = json.load(f)

image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}

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
# IoU
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
# MATCHING DETECTOR + SORT
# ======================
def match_detections_detector_plus_sort(final_detections, gt_boxes, iou_thresh=0.5):

    matched_gt = set()
    TP, FP = 0, 0

    detector_dets = [d for d in final_detections if d["source"] == "detector"]

    for det in detector_dets:
        ious = [compute_iou(det["bbox"], gt) for gt in gt_boxes]
        if len(ious) == 0 or max(ious) < iou_thresh:
            FP += 1
        else:
            idx = np.argmax(ious)
            if idx not in matched_gt:
                TP += 1
                matched_gt.add(idx)
            else:
                FP += 1

    kalman_dets = [d for d in final_detections if d["source"] == "kalman"]
    for det in kalman_dets:
        ious = [compute_iou(det["bbox"], gt) for gt in gt_boxes]
        if len(ious) > 0 and max(ious) >= iou_thresh:
            idx = np.argmax(ious)
            if idx not in matched_gt:
                TP += 1
                matched_gt.add(idx)

    FN = len(gt_boxes) - len(matched_gt)
    return TP, FP, FN

# ======================
# PROCESAR FRAMES
# ======================
image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
first_frame = cv2.imread(image_paths[0])
h, w, _ = first_frame.shape

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"mp4v"),
    30,
    (w, h)
)

total_TP, total_FP, total_FN = 0, 0, 0
y_true_all, y_score_all = [], []

for path in image_paths:
    frame = cv2.imread(path)
    frame_name = os.path.basename(path)

    # ======================
    # YOLO DETECTIONS
    # ======================
    results = model.predict(frame, conf=CONF_THRESH, device=DEVICE, verbose=False)

    dets = []
    scores = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()

        for box, score, cls in zip(boxes, confs, clss):
            if int(cls) == YOLO_HUMAN_CLASS_ID:
                dets.append(box)
                scores.append(score)

    if len(dets) > 0:
        dets = np.hstack([np.array(dets), np.array(scores)[:, None]])
    else:
        dets = np.empty((0, 5))

    # ======================
    # SORT
    # ======================
    tracks = tracker.update(dets)

    final_detections = []

    for i, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = track
        box = np.array([x1, y1, x2, y2])

        kalman_tracker = tracker.trackers[i]
        if kalman_tracker.time_since_update == 0:
            source = "detector"
            color = (0, 255, 0)
            score_det = scores[i] if i < len(scores) else 1.0
        else:
            source = "kalman"
            color = (0, 0, 255)
            score_det = 1.0

        final_detections.append({
            "bbox": box,
            "source": source,
            "score": score_det
        })

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # ======================
    # METRICAS
    # ======================
    gt_boxes = gt_boxes_per_image.get(frame_name, np.empty((0,4)))
    TP, FP, FN = match_detections_detector_plus_sort(final_detections, gt_boxes)

    total_TP += TP
    total_FP += FP
    total_FN += FN

    # ROC (solo detector)
    for det in final_detections:
        if det["source"] != "detector":
            continue
        matched = any(compute_iou(det["bbox"], gt) >= 0.5 for gt in gt_boxes)
        y_true_all.append(1 if matched else 0)
        y_score_all.append(det["score"])

    out.write(frame)

out.release()

# ======================
# RESULTADOS
# ======================
precision = total_TP / (total_TP + total_FP + 1e-6)
recall = total_TP / (total_TP + total_FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print(f"TP={total_TP}, FP={total_FP}, FN={total_FN}")
print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

# ======================
# MATRIZ CONFUSIÓN
# ======================
conf_matrix = np.array([[total_TP, total_FN],
                        [total_FP, 0]])

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Predicho Objeto", "Predicho No Objeto"],
            yticklabels=["GT Objeto", "GT No Objeto"])
plt.title("Matriz de Confusión YOLO + SORT")
plt.tight_layout()
plt.savefig("confusion_matrix_yolo_SORT.png", dpi=300)
plt.close()

# ======================
# ROC
# ======================
fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],"--",color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC YOLO + SORT")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_yolo_SORT.png", dpi=300)
plt.close()
