# ======================================================
# IMPORTS
# ======================================================
import os
import cv2
import glob
import json
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from sort import Sort
from skimage.segmentation import slic
from scipy.stats import pearsonr

import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm


# ======================================================
# CONFIGURACIÓN
# ======================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONF_THRESH = 0.5
DCL_SCORE_THRESH = 0.6
HUMAN_CLASS_ID = 1
NUM_CLASSES_LIME = 2   # fondo / humano

IMG_DIR = "cam_ta1_ws2/"
GT_JSON = "ground_truth_sinMujerAtras.json"

OUTPUT_DIR = "LIME_SORT_RESULTS"
SAL_MAPS_DIR = os.path.join(OUTPUT_DIR, "saliency_maps")
os.makedirs(SAL_MAPS_DIR, exist_ok=True)


# ======================================================
# MODELO
# ======================================================
def get_model(num_classes=2):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = 0.3
    return model

model = get_model()
model.load_state_dict(
    torch.load(
        "modelos_faster_best_fasterrcnn_data_augmentation_transferLearning_SOLO_HUMANO_conReal.pth",
        map_location=DEVICE
    )
)
model.to(DEVICE).eval()


# ======================================================
# SORT
# ======================================================
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)


# ======================================================
# LIME SETUP
# ======================================================
explainer = lime_image.LimeImageExplainer()
segmenter = SegmentationAlgorithm(
    'slic',
    n_segments=200,
    compactness=10,
    start_label=0
)


# ======================================================
# FUNCIONES AUXILIARES
# ======================================================
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-8)


def predict(img_rgb):
    t = torch.from_numpy(img_rgb / 255.).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return model(t)[0]


# ======================================================
# WRAPPER LIME PARA DETECCIÓN
# ======================================================
def get_probab_class_wrapper(model, target_class, num_classes):
    def get_probab_class(imgs):
        batch = []
        for img in imgs:
            t = torchvision.transforms.ToTensor()(Image.fromarray(img))
            batch.append(t.to(DEVICE))

        with torch.no_grad():
            outputs = model(batch)

        probab_results = []
        for out in outputs:
            scores = [
                float(out["scores"][i].item())
                for i, lbl in enumerate(out["labels"])
                if int(lbl.item()) == target_class
            ]

            if scores:
                prob_cls = float(np.mean(scores))
                other = (1.0 - prob_cls) / (num_classes - 1)
                prob_vec = np.full(num_classes, other, dtype=np.float32)
                prob_vec[target_class] = prob_cls
            else:
                prob_vec = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)

            probab_results.append(prob_vec)

        return np.array(probab_results)

    return get_probab_class


# ======================================================
# MAPA LIME GLOBAL
# ======================================================
def lime_map_global(img_rgb, model, target_class):
    explanation = explainer.explain_instance(
        image=img_rgb,
        classifier_fn=get_probab_class_wrapper(
            model, target_class, NUM_CLASSES_LIME
        ),
        top_labels=1,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=segmenter
    )

    segments = explanation.segments
    sal_map = np.zeros(segments.shape, dtype=np.float32)

    # PROTECCIÓN CLAVE - para cuando no haya detecciones
    if target_class not in explanation.local_exp:
        # No hay explicación para esta clase → mapa nulo
        return sal_map

    for sp, w in explanation.local_exp[target_class]:
        sal_map[segments == sp] = w

    sal_map -= sal_map.min()
    sal_map /= (sal_map.max() + 1e-8)

    return sal_map


# ======================================================
# MÉTRICAS
# ======================================================
def deletion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map):
    img_float = img_rgb.astype(np.float32)/255.0
    segments = slic(img_float, n_segments=100, compactness=10, start_label=0)
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]

    perturbed = img_rgb.copy()
    c_scores, s_scores = [], []

    for idx in sorted_idx:
        mask = segments == seg_vals[idx]
        s_scores.append(seg_scores[idx])
        perturbed[mask] = 0

        out = predict(perturbed)
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(out["boxes"].cpu().numpy(),
                           out["scores"].cpu().numpy(),
                           out["labels"].cpu().numpy()):
            if l != orig_class_id:
                continue
            this_iou = iou(orig_box, b)
            if this_iou > best_iou:
                best_iou, best_score = this_iou, s

        c_scores.append(best_score if best_iou > 0.3 else 0.0)

    if len(c_scores) < 2:
        return 0.0

    v = np.array(c_scores[:-1]) - np.array(c_scores[1:])
    s = np.array(s_scores[:-1])
    if np.std(v)==0 or np.std(s)==0:
        return 0.0

    corr, _ = pearsonr(v, s)
    return corr


def insertion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map):
    img_float = img_rgb.astype(np.float32)/255.0
    segments = slic(img_float, n_segments=100, compactness=10, start_label=0)
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]

    perturbed = cv2.GaussianBlur(img_rgb, (21,21), 0)
    c_scores, s_scores = [], []

    for idx in sorted_idx:
        mask = segments == seg_vals[idx]
        s_scores.append(seg_scores[idx])
        perturbed[mask] = img_rgb[mask]

        out = predict(perturbed)
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(out["boxes"].cpu().numpy(),
                           out["scores"].cpu().numpy(),
                           out["labels"].cpu().numpy()):
            if l != orig_class_id:
                continue
            this_iou = iou(orig_box, b)
            if this_iou > best_iou:
                best_iou, best_score = this_iou, s

        c_scores.append(best_score if best_iou > 0.3 else 0.0)

    if len(c_scores) < 2:
        return 0.0

    v = np.array(c_scores[1:]) - np.array(c_scores[:-1])
    s = np.array(s_scores[1:])
    if np.std(v)==0 or np.std(s)==0:
        return 0.0

    corr, _ = pearsonr(v, s)
    return corr


def sparsity(sal_map):
    return 1.0 / (sal_map.mean() + 1e-8)


def compute_pointing_game(sal_map, gt_box):
    x1,y1,x2,y2 = map(int,gt_box)
    max_y, max_x = np.unravel_index(np.argmax(sal_map), sal_map.shape)
    return 1 if (x1<=max_x<=x2 and y1<=max_y<=y2) else 0


def compute_EBPG(sal_map, gt_box):
    x1,y1,x2,y2 = map(int,gt_box)
    S = sal_map.astype(np.float32)
    S /= (S.sum() + 1e-12)
    return float(S[y1:y2+1, x1:x2+1].sum())


def match_prediction_to_gt(pred_box, gt_boxes):
    best_gt, best_iou = None, 0.0
    for gt in gt_boxes:
        val = iou(pred_box, gt)
        if val > best_iou:
            best_iou, best_gt = val, gt
    return best_gt if best_iou >= 0.5 else None


# ======================================================
# CARGAR GT
# ======================================================
with open(GT_JSON, 'r') as f:
    coco = json.load(f)

image_id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
gt_boxes_per_image = {}

for ann in coco["annotations"]:
    if ann["category_id"] != HUMAN_CLASS_ID:
        continue
    img_name = image_id_to_name[ann["image_id"]]
    x,y,w,h = ann["bbox"]
    box = [int(x), int(y), int(x+w), int(y+h)]
    gt_boxes_per_image.setdefault(img_name, []).append(box)


# ======================================================
# LOOP PRINCIPAL
# ======================================================
results = []
image_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.jpg")))

for frame_idx, path in enumerate(image_paths):
    frame = cv2.imread(path)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_name = os.path.basename(path)
    gt_boxes = gt_boxes_per_image.get(frame_name, [])

    out = predict(img_rgb)
    boxes = out["boxes"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    labels = out["labels"].cpu().numpy()

    mask = (scores >= CONF_THRESH) & (labels == HUMAN_CLASS_ID)
    boxes, scores = boxes[mask], scores[mask]
    dets = np.concatenate([boxes, scores[:,None]], axis=1) if len(boxes) else np.empty((0,5))
    tracks = tracker.update(dets)

    for i, track in enumerate(tracks):
        x1,y1,x2,y2,track_id = track.astype(int)
        box = [x1,y1,x2,y2]
        k = tracker.trackers[i]

        source = "detector" if k.time_since_update == 0 else "kalman"
        score = scores[i] if source=="detector" and i < len(scores) else 1.0

        row = {
            "frame": frame_idx,
            "track_id": int(track_id),
            "source": source,
            "score": score,
            "DC": np.nan,
            "IC": np.nan,
            "Sparsity": np.nan,
            "PG_Hit": np.nan,
            "EBPG": np.nan
        }

        if source=="detector" and score >= DCL_SCORE_THRESH:
            sal = lime_map_global(img_rgb, model, HUMAN_CLASS_ID)

            row["DC"] = deletion_correlation(img_rgb, model, box, HUMAN_CLASS_ID, score, sal)
            row["IC"] = insertion_correlation(img_rgb, model, box, HUMAN_CLASS_ID, score, sal)
            row["Sparsity"] = sparsity(sal)

            gt_match = match_prediction_to_gt(box, gt_boxes)
            if gt_match is not None:
                row["PG_Hit"] = compute_pointing_game(sal, gt_match)
                row["EBPG"] = compute_EBPG(sal, gt_match)

            plt.imsave(
                os.path.join(SAL_MAPS_DIR, f"frame{frame_idx}_track{track_id}.png"),
                sal, cmap="jet"
            )

        results.append(row)


# ======================================================
# GUARDAR CSV
# ======================================================
df = pd.DataFrame(results)
csv_path = os.path.join(OUTPUT_DIR, "lime_sort_metrics.csv")
df.to_csv(csv_path, index=False)

print("✔ Proceso terminado")
print(f"✔ CSV: {csv_path}")
print(f"✔ Saliency maps: {SAL_MAPS_DIR}")
