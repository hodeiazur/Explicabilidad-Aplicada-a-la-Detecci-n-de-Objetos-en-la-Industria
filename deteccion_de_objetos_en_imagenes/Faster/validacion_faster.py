import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# CONFIG
# ------------------------------
image_path = "00901.jpg"
json_path = "instances_default_00901.json"
model_path = "fasterrcnn_trained_4.pth"
iou_thresh = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# CARGAR MODELO
# ------------------------------
model = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 12
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# ------------------------------
# CARGAR IMAGEN
# ------------------------------
image_cv = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
image_tensor = F.to_tensor(image_rgb).to(device)

# ------------------------------
# DETECCIÓN
# ------------------------------
with torch.no_grad():
    outputs = model([image_tensor])

pred_bboxes = outputs[0]['boxes'].cpu().numpy()
pred_scores = outputs[0]['scores'].cpu().numpy()
pred_labels = outputs[0]['labels'].cpu().numpy()

# ------------------------------
# LEER JSON COCO
# ------------------------------
with open(json_path) as f:
    data = json.load(f)

gt_bboxes = []
gt_labels = []
for ann in data['annotations']:
    gt_bboxes.append(ann['bbox'])
    gt_labels.append(ann['category_id'])

# Convertir COCO [x,y,w,h] a [x1,y1,x2,y2]
gt_bboxes_xyxy = [[x, y, x + w, y + h] for x, y, w, h in gt_bboxes]

# ------------------------------
# FUNCIÓN IoU
# ------------------------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# ------------------------------
# MATRIZ DE CONFUSIÓN Y ROC
# ------------------------------
#matriz de confusionen bakarrik eukitzeko conf > 0.5 !! ze da FINKO utziko deun filtro bat
conf_threshold = 0.5

# Filtrar predicciones antes de todo
mask = pred_scores > conf_threshold
pred_bboxes = pred_bboxes[mask]
pred_labels = pred_labels[mask]
pred_scores = pred_scores[mask]

# Ahora hacemos el matching solo con estas predicciones
y_true_cm, y_pred_cm = [], []
matched_gt = set()
iou_thresh = 0.5

for gt_idx, gt_box in enumerate(gt_bboxes_xyxy):
    best_iou = 0
    best_pred_idx = -1
    for pred_idx, pred_box in enumerate(pred_bboxes):
        score_iou = iou(gt_box, pred_box)
        if score_iou > best_iou:
            best_iou = score_iou
            best_pred_idx = pred_idx
    if best_iou >= iou_thresh:
        y_true_cm.append(gt_labels[gt_idx])
        y_pred_cm.append(pred_labels[best_pred_idx])
        matched_gt.add(best_pred_idx)
    else:
        y_true_cm.append(gt_labels[gt_idx])
        y_pred_cm.append(-1)

# False Positives solo de predicciones filtradas
for pred_idx in range(len(pred_bboxes)):
    if pred_idx not in matched_gt:
        y_true_cm.append(-1)
        y_pred_cm.append(pred_labels[pred_idx])

# Matriz de confusión
cm = confusion_matrix(y_true_cm, y_pred_cm)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicho')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión (conf > 0.5)')
plt.savefig("confusion_matrix_fasterrcnn_conf_0_5.png")
plt.show()


# ------------------------------
# MATRIZ DE CONFUSIÓN
# ------------------------------
# Filtrar predicciones antes de cualquier cálculo
conf_threshold = 0.5
mask = pred_scores > conf_threshold
pred_bboxes = pred_bboxes[mask]
pred_labels = pred_labels[mask]
pred_scores = pred_scores[mask]

# ROC curve
y_true_roc, y_score_roc = [], []
matched_gt = set()
iou_thresh = 0.5

for pred_idx, pred_box in enumerate(pred_bboxes):
    best_iou = 0
    best_gt_idx = -1
    for gt_idx, gt_box in enumerate(gt_bboxes_xyxy):
        score_iou = iou(pred_box, gt_box)
        if score_iou > best_iou:
            best_iou = score_iou
            best_gt_idx = gt_idx
    if best_iou >= iou_thresh:
        y_true_roc.append(1)                  # TP
        y_score_roc.append(pred_scores[pred_idx])
        matched_gt.add(best_gt_idx)
    else:
        y_true_roc.append(0)                  # FP
        y_score_roc.append(pred_scores[pred_idx])

# Calcular ROC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_true_roc, y_score_roc)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='gray', linestyle='--')
plt.xlabel('Ratio Falsos Positivos')
plt.ylabel('Ratio Verdaderos Positivos')
plt.title('Curva de ROC (conf > 0.5)')
plt.legend()
plt.savefig("roc_curve_fasterrcnn_conf_0_5.png")
plt.show()
