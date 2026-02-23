from ultralytics import YOLO
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

# ------------------------------
# CARGA MODELO E IMAGEN
# ------------------------------
model = YOLO("best.pt")
image_path = "00901.jpg"
image = cv2.imread(image_path)

# Obtener nombres de clases del modelo YOLO
class_names = model.names  # dict {id: name}

# ------------------------------
# LEER JSON COCO (GROUND TRUTH)
# ------------------------------
with open("instances_default_00901.json") as f:
    data = json.load(f)

gt_bboxes = []
gt_labels = []

for ann in data["annotations"]:
    gt_bboxes.append(ann["bbox"])          # [x, y, w, h]
    gt_labels.append(ann["category_id"])   # id de clase

# COCO [x,y,w,h] -> [x1,y1,x2,y2]
gt_bboxes_xyxy = [
    [x, y, x + w, y + h] for x, y, w, h in gt_bboxes
]

# ------------------------------
# PREDICCIONES DEL MODELO
# ------------------------------
results = model.predict(image_path, verbose=False)

pred_bboxes = results[0].boxes.xyxy.cpu().numpy()
pred_scores = results[0].boxes.conf.cpu().numpy()
pred_labels = results[0].boxes.cls.cpu().numpy().astype(int)

# ------------------------------
# FUNCIÓN IOU
# ------------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / (boxAArea + boxBArea - interArea + 1e-6)

# ------------------------------
# MATRIZ DE CONFUSIÓN
# ------------------------------
conf_threshold = 0.5
iou_threshold = 0.5

# Filtrar predicciones por confianza
mask = pred_scores > conf_threshold
pred_bboxes = pred_bboxes[mask]
pred_labels = pred_labels[mask]
pred_scores = pred_scores[mask]

y_true_cm = []
y_pred_cm = []
matched_preds = set()

# Match GT -> Pred
for gt_idx, gt_box in enumerate(gt_bboxes_xyxy):
    best_iou = 0
    best_pred_idx = -1

    for pred_idx, pred_box in enumerate(pred_bboxes):
        current_iou = iou(gt_box, pred_box)
        if current_iou > best_iou:
            best_iou = current_iou
            best_pred_idx = pred_idx

    if best_iou >= iou_threshold:
        y_true_cm.append(gt_labels[gt_idx])
        y_pred_cm.append(pred_labels[best_pred_idx])
        matched_preds.add(best_pred_idx)
    else:
        # False Negative
        y_true_cm.append(gt_labels[gt_idx])
        y_pred_cm.append(-1)  # Background

# False Positives
for pred_idx in range(len(pred_bboxes)):
    if pred_idx not in matched_preds:
        y_true_cm.append(-1)               # Background
        y_pred_cm.append(pred_labels[pred_idx])


labels_cm = sorted(set(y_true_cm) | set(y_pred_cm))

label_names = []
for l in labels_cm:
    if l == -1:
        label_names.append("Background")
    else:
        label_names.append(class_names[l])

cm = confusion_matrix(
    y_true_cm,
    y_pred_cm,
    labels=labels_cm
)

plt.figure(figsize=(8,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names
)

plt.xlabel("Predicho")
plt.ylabel("Verdadero")
plt.title("Matriz de Confusión (conf > 0.5)")
plt.tight_layout()
plt.savefig("confusion_matrix_conf_0_5.png")
plt.show()

# -----------
# CURVA ROC 
# -----------
y_true_roc = []
y_score_roc = []
matched_gt = set()

for pred_idx, pred_box in enumerate(pred_bboxes):
    best_iou = 0
    best_gt_idx = -1

    for gt_idx, gt_box in enumerate(gt_bboxes_xyxy):
        current_iou = iou(pred_box, gt_box)
        if current_iou > best_iou:
            best_iou = current_iou
            best_gt_idx = gt_idx

    if best_iou >= iou_threshold:
        y_true_roc.append(1)   # TP
        y_score_roc.append(pred_scores[pred_idx])
        matched_gt.add(best_gt_idx)
    else:
        y_true_roc.append(0)   # FP
        y_score_roc.append(pred_scores[pred_idx])

fpr, tpr, thresholds = roc_curve(y_true_roc, y_score_roc)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Curva ROC (conf > 0.5)")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_conf_0_5.png")
plt.show()
