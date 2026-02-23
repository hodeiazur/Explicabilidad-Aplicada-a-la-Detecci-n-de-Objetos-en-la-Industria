import torch
import cv2
import os
import glob
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from sort import Sort
import numpy as np
import json
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
# ======================
# CONFIGURACIÓN
# ======================

MODEL_PATH = "modelos_faster/best_fasterrcnn_data_augmentation_transferLearning_SOLO_HUMANO_conReal.pth"  
IMAGES_DIR = "cam_ta1_ws2/"       
OUTPUT_VIDEO = "detecciones_faster_new_data_augmentation_video_original_ta1_ws2_IoU_0.2_SORT_new_labelHuman_conReal_3.mp4"  
CONF_THRESH = 0.5                    
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
NUM_CLASSES = 2 #humano + fondo
HUMAN_CLASS_ID = 1


# ======================
# CARGAR MODELO
# ======================

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)  
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    #ajustes para que no coja 3 bboxes uno encima del otro
    model.roi_heads.nms_thresh = 0.3 #Si dos cajas se solapan > 30% → se queda solo la mejor
    return model


# ==============================
# CARGA DEL MODELO
# ==============================

model = get_model(NUM_CLASSES)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()

# Inicializamos tracker SORT
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3) #hauek pixkat aldau ze oain gutxigi hartzettu

#max_age --> cuantos frames mantiene un track sin deteccion antes de borrarlo
#min_hits --> cuantos frames de deteccion consecutivos necesita un track para aparecer
#iou_threshold --> umbral de asociacion deteccion--> track

# ==============================
# CARGAR GROUND TRUTH COCO
# ==============================

GT_JSON = "ground_truth_sinMujerAtras.json"

with open(GT_JSON, "r") as f:
    coco = json.load(f)

# Mapeo image_id -> nombre de archivo
image_id_to_name = {
    img["id"]: img["file_name"] for img in coco["images"]
}

# Mapeo nombre de archivo -> GT boxes
gt_boxes_per_image = defaultdict(list)

for ann in coco["annotations"]:

    if ann["category_id"] != HUMAN_CLASS_ID:
        continue  # ❌ ignorar clases que no sean humano

    image_id = ann["image_id"]
    file_name = image_id_to_name[image_id]

    x, y, w, h = ann["bbox"]
    x1, y1, x2, y2 = x, y, x + w, y + h

    gt_boxes_per_image[file_name].append([x1, y1, x2, y2])


# Convertimos listas a np.array
for k in gt_boxes_per_image:
    gt_boxes_per_image[k] = np.array(gt_boxes_per_image[k])

# ==============================
# IoU
# ==============================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    if areaA + areaB - inter == 0:
        return 0.0

    return inter / (areaA + areaB - inter)


# ==============================
# MATRIZ DE CONFUSION
# ==============================
def match_detections_detector_plus_sort(
    final_detections,
    gt_boxes,
    iou_thresh=0.5
):
    """
    final_detections: lista de dicts
        {
          "bbox": np.array([x1,y1,x2,y2]),
          "source": "detector" | "kalman"
        }

    gt_boxes: np.array [N,4]
    """

    if len(gt_boxes) == 0:
        # No GT → solo FP si el detector dispara
        FP = sum(1 for d in final_detections if d["source"] == "detector")
        return 0, FP, 0

    matched_gt = set()
    TP = 0
    FP = 0

    # ==============================
    # 1️⃣ MATCH DETECTOR → GT
    # ==============================
    detector_dets = [d for d in final_detections if d["source"] == "detector"]

    for det in detector_dets:
        det_box = det["bbox"]
        ious = [compute_iou(det_box, gt) for gt in gt_boxes]

        if len(ious) == 0 or max(ious) < iou_thresh:
            FP += 1  # detección falsa real
        else:
            idx = np.argmax(ious)
            if idx not in matched_gt:
                TP += 1
                matched_gt.add(idx)
            else:
                FP += 1  # doble detección del mismo GT

    # ==============================
    # 2️⃣ KALMAN CUBRE FN TEMPORALES
    # ==============================
    kalman_dets = [d for d in final_detections if d["source"] == "kalman"]

    for det in kalman_dets:
        det_box = det["bbox"]
        ious = [compute_iou(det_box, gt) for gt in gt_boxes]

        if len(ious) == 0:
            continue

        if max(ious) >= iou_thresh:
            idx = np.argmax(ious)
            if idx not in matched_gt:
                TP += 1              # 🔥 FN recuperado por tracking
                matched_gt.add(idx)

    # ==============================
    # 3️⃣ FN FINALES
    # ==============================
    FN = len(gt_boxes) - len(matched_gt)

    return TP, FP, FN
 

# ==============================
# PROCESAR LOS FRAMES
# ==============================

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
y_true_all = []
y_score_all = []

for path in image_paths:
    frame = cv2.imread(path)

    img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
    with torch.no_grad():
        preds = model([img_tensor.to(DEVICE)])[0]

    # --------------------------
    # DETECCIONES DEL MODELO
    # --------------------------
    boxes = preds["boxes"].detach().cpu().numpy()
    scores = preds["scores"].detach().cpu().numpy()
    labels = preds["labels"].detach().cpu().numpy()

    # Filtrar solo humanos + threshold
    mask = (scores >= CONF_THRESH) & (labels == HUMAN_CLASS_ID)

    boxes = boxes[mask]
    scores = scores[mask]
    labels = labels[mask]


    if len(boxes) > 0:
        dets = np.concatenate([boxes, scores[:, None]], axis=1)
    else:
        dets = np.empty((0, 5))

    # --------------------------
    # SORT (Kalman)
    # --------------------------
    tracks = tracker.update(dets)

    final_detections = []

    for i, track in enumerate(tracks):

        x1, y1, x2, y2, track_id = track
        track_box = np.array([x1, y1, x2, y2])
        kalman_tracker = tracker.trackers[i]
        if kalman_tracker.time_since_update == 0:
            # 🟢 detección real
            color = (0, 255, 0)
            source = "detector"
            score_det = scores[i] if i < len(scores) else 1.0
        else:
            # 🔴 predicción Kalman
            color = (0, 0, 255)
            source = "kalman"
            score_det = 1.0 #kalman no tiene score

        final_detections.append({
            "bbox": track_box,
            "id": int(track_id),
            "source": source,
            "score": score_det
        })


        x1, y1, x2, y2 = map(int, track_box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"ID {int(track_id)} ({source})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )
       
    
    # --------------------------
    # GROUND TRUTH POR FRAME
    # --------------------------
    frame_name = os.path.basename(path)
    gt_boxes_frame = gt_boxes_per_image.get(
        frame_name,
        np.empty((0, 4))
    )

    # --------------------------
    # MATRIZ DE CONFUSIÓN
    # --------------------------
    TP, FP, FN = match_detections_detector_plus_sort(
        final_detections,
        gt_boxes_frame,
        iou_thresh=0.5
    )

    total_TP += TP
    total_FP += FP
    total_FN += FN

    out.write(frame)

    # --------------------------
    # ROC DATA
    # --------------------------
    for det in final_detections:
        if det["source"] != "detector":
            continue  # ❌ Kalman no entra en ROC

        det_box = det["bbox"]
        score = det['score']

        matched = False
        for gt in gt_boxes_frame:
            if compute_iou(det_box, gt) >= 0.5:
                matched = True
                break

        y_true_all.append(1 if matched else 0)
        y_score_all.append(score)


out.release()
print(f"Vídeo guardado en {OUTPUT_VIDEO}")
print("\n===== MATRIZ DE CONFUSIÓN GLOBAL =====")
print(f"TP: {total_TP}")
print(f"FP: {total_FP}")
print(f"FN: {total_FN}")

precision = total_TP / (total_TP + total_FP + 1e-6)
recall = total_TP / (total_TP + total_FN + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================
# MATRIZ DE CONFUSIÓN
# ==============================

TN = 0  # no se usa en object detection

conf_matrix = np.array([
    [total_TP, total_FN],
    [total_FP, TN]
])

plt.figure(figsize=(6, 5))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicho Objeto", "Predicho No Objeto"],
    yticklabels=["GT Objeto", "GT No Objeto"]
)

plt.title("Matriz de Confusión (Detector + SORT)")
plt.ylabel("Verdadero")
plt.xlabel("Predicho")
plt.tight_layout()
plt.savefig("confusion_matrix_SORT_labelHuman_conReal_3.png", dpi=300)
plt.close()


from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true_all, y_score_all)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

plt.xlabel("Ratio Falsos Positivos")
plt.ylabel("Ratio Verdaderos Positivos")
plt.title("Curva ROC (Detector + SORT)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_SORT_labelHuman_conReal_3.png", dpi=300)
plt.close()

