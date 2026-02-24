import os
import gc
import cv2
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
from torchvision import transforms
from drise import DRISEBatch
from skimage.segmentation import slic
from scipy.stats import pearsonr

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best.pt"

IMAGE_DIR  = "cam_ta1_ws2/"
OUTPUT_DIR = "DRISE_MULTI_NO_LETTERBOX"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIXED_SIZE = (640, 640)

CONF_DET  = 0.5
CONF_ATTR = 0.05     # atribución

GT_JSON = "ground_truth_sinMujerAtras.json"

YOLO_HUMAN_CLASS_ID = 0
COCO_HUMAN_CATEGORY_ID = 1


# =========================
# UTILIDADES
# =========================
def iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB-xA)*max(0, yB-yA)
    areaA = max(0, boxA[2]-boxA[0]) * max(0, boxA[3]-boxA[1])
    areaB = max(0, boxB[2]-boxB[0]) * max(0, boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-8)


def load_coco_gt(json_path, image_filename):
    """
    Devuelve (gt_boxes, gt_labels) siempre.
    Si el JSON no existe, no contiene la imagen o no hay anotaciones, devuelve ([], []).
    gt_boxes: list[[x1,y1,x2,y2], ...] en coords de la imagen ORIGINAL.
    gt_labels: list[int] category_id COCO por caja.
    """
    # Defensa: json_path nulo o archivo no existe
    if not json_path or not os.path.isfile(json_path):
        return [], []

    # Leer COCO
    with open(json_path, "r") as f:
        coco = json.load(f)

    # Buscar la imagen por file_name
    img_id = None
    for img in coco.get("images", []):
        if img.get("file_name") == image_filename:
            img_id = img.get("id")
            break

    # Si no se encuentra la imagen en el JSON
    if img_id is None:
        return [], []

    # Recoger anotaciones para esa imagen
    gt_boxes, gt_labels = [], []
    for ann in coco.get("annotations", []):
        if ann.get("image_id") == img_id:
            x, y, w, h = ann.get("bbox", [0,0,0,0])
            gt_boxes.append([int(x), int(y), int(x + w), int(y + h)])
            gt_labels.append(int(ann.get("category_id", -1)))

    return gt_boxes, gt_labels

def sparsity(s):
    return 1.0 / (s.mean() + 1e-8)


def compute_pointing_game(sal, gt):
    x1,y1,x2,y2 = map(int, gt)
    yy, xx = np.unravel_index(np.argmax(sal), sal.shape)
    return int(x1 <= xx <= x2 and y1 <= yy <= y2)


def compute_EBPG(sal, gt):
    x1,y1,x2,y2 = map(int, gt)
    S = sal / (sal.sum() + 1e-12)
    return float(S[y1:y2+1, x1:x2+1].sum())


def _scale_box_from_small_to_orig(box_s, W0, H0):
    sx, sy = W0 / 640.0, H0 / 640.0
    x1,y1,x2,y2 = box_s
    return [x1*sx, y1*sy, x2*sx, y2*sy]


def match_to_gt_human(box_o, gt_boxes, gt_labels, thr=0.5):
    best_gt, best_iou = None, 0.0
    for gt, gt_cls in zip(gt_boxes, gt_labels):
        if gt_cls != COCO_HUMAN_CATEGORY_ID:
            continue
        v = iou(box_o, gt)
        if v > best_iou:
            best_gt, best_iou = gt, v
    return (best_gt, best_iou) if best_iou >= thr else (None, 0.0)


# =========================
# WRAPPER YOLO batch-aware
# =========================
class YOLOWrapperBatch(torch.nn.Module):
    def __init__(self, yolo_model, conf_thres):
        super().__init__()
        self.yolo = yolo_model
        self.conf = conf_thres
        self.yolo.model.eval()

    def forward(self, x):
        B = x.shape[0]
        dev = x.device
        outs = []
        for i in range(B):
            img = (x[i].permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            r = self.yolo(img, conf=self.conf, verbose=False)[0]
            if (r.boxes is None) or (len(r.boxes)==0):
                outs.append({
                    "boxes": torch.empty((0,4),device=dev),
                    "scores": torch.empty((0,),device=dev),
                    "labels": torch.empty((0,),dtype=torch.long,device=dev)
                })
            else:
                outs.append({
                    "boxes":  r.boxes.xyxy.to(dev).float(),
                    "scores": r.boxes.conf.to(dev).float(),
                    "labels": r.boxes.cls.to(dev).long()
                })
        return outs


# =========================
# MÉTRICAS (con predicción en 640 y remapeo)
# =========================
def _predict_small(wrapper, img_small):
    t = transforms.ToTensor()(img_small).unsqueeze(0).to(DEVICE)
    return wrapper(t)[0]


def deletion_correlation(img_orig, img_small, model_attr, box_orig, cls, sal_norm, mask=114):
    H0, W0 = img_orig.shape[:2]
    segments = slic(img_orig.astype(np.float32)/255., n_segments=120, compactness=10)
    vals = np.unique(segments)
    seg_scores = [sal_norm[segments==v].mean() for v in vals]
    order = np.argsort(seg_scores)[::-1]

    pert = img_orig.copy()
    c_scores = []

    for idx in order:
        pert[segments==vals[idx]] = mask
        pert_small = cv2.resize(pert, (640,640))
        out = _predict_small(model_attr, pert_small)

        best = 0
        for b,s,l in zip(out["boxes"].cpu().numpy(),
                         out["scores"].cpu().numpy(),
                         out["labels"].cpu().numpy()):
            if int(l) != YOLO_HUMAN_CLASS_ID:
                continue
            b_orig = _scale_box_from_small_to_orig(b, W0, H0)
            if iou(box_orig, b_orig) > 0.3:
                best = max(best, float(s))
        c_scores.append(best)

    if len(c_scores)<3: return 0.0
    v = np.array(c_scores[:-1]) - np.array(c_scores[1:])
    if np.std(v)==0: return 0.0
    return pearsonr(v, np.arange(len(v)))[0]


def insertion_correlation(img_orig, img_small, model_attr, box_orig, cls, sal_norm):
    H0, W0 = img_orig.shape[:2]
    segments = slic(img_orig.astype(np.float32)/255., n_segments=120, compactness=10)
    vals = np.unique(segments)
    seg_scores = [sal_norm[segments==v].mean() for v in vals]
    order = np.argsort(seg_scores)[::-1]

    pert = cv2.GaussianBlur(img_orig, (21,21),0)
    c_scores = []

    for idx in order:
        pert[segments==vals[idx]] = img_orig[segments==vals[idx]]
        pert_small = cv2.resize(pert, (640,640))
        out = _predict_small(model_attr, pert_small)

        best = 0
        for b,s,l in zip(out["boxes"].cpu().numpy(),
                         out["scores"].cpu().numpy(),
                         out["labels"].cpu().numpy()):
            if int(l) != YOLO_HUMAN_CLASS_ID:
                continue
            b_orig = _scale_box_from_small_to_orig(b, W0, H0)
            if iou(box_orig, b_orig) > 0.3:
                best = max(best, float(s))
        c_scores.append(best)

    if len(c_scores)<3: return 0.0
    v = np.array(c_scores[1:]) - np.array(c_scores[:-1])
    if np.std(v)==0: return 0.0
    return pearsonr(v, np.arange(len(v)))[0]


# =========================
# CARGAR MODELO
# =========================
yolo = YOLO(MODEL_PATH)
model_det  = YOLOWrapperBatch(yolo, CONF_DET ).to(DEVICE)
model_attr = YOLOWrapperBatch(yolo, CONF_ATTR).to(DEVICE)

drise = DRISEBatch(
    model=model_attr,
    input_size=(640,640),
    device=DEVICE,
    N=5000,
    p1=0.25,
    gpu_batch=8
)
drise.generate_masks_rise(N=5000, s=16, p1=0.25)


# =========================
# LOOP IMÁGENES
# =========================
rows = []

images = [f for f in os.listdir(IMAGE_DIR)
          if f.lower().endswith((".jpg",".jpeg",".png"))]

for imgname in tqdm(images):
    path = os.path.join(IMAGE_DIR, imgname)
    bgr = cv2.imread(path)
    if bgr is None:
        continue

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H0,W0 = rgb.shape[:2]

    # resize duro para YOLO y D-RISE
    rgb_small = cv2.resize(rgb, (640,640))
    t_small   = transforms.ToTensor()(rgb_small).unsqueeze(0).to(DEVICE)

    # detección base
    pred = model_det(t_small)[0]
    boxes_s, lbls_s, scrs_s = pred["boxes"], pred["labels"], pred["scores"]

    # filtrar detecciones humanas
    cand = [(i, scrs_s[i].item(), boxes_s[i].cpu().numpy().astype(float))
            for i in range(len(lbls_s))
            if int(lbls_s[i].item()) == YOLO_HUMAN_CLASS_ID]

    if len(cand)==0:
        continue

    # remap a original y emparejar con GT humano
    gt_boxes, gt_labels = load_coco_gt(GT_JSON, imgname)

    valid = []
    for idx,score,box_s in cand:
        box_o = _scale_box_from_small_to_orig(box_s, W0, H0)
        gt, iouv = match_to_gt_human(box_o, gt_boxes, gt_labels)
        if gt is not None:
            valid.append((score, idx, box_s, box_o, gt))

    # si no hay detecciones humanas válidas (según GT), saltar imagen
    if len(valid)==0:
        continue

    # solo 1 mapa: top‑1 por score
    valid.sort(key=lambda t: t[0], reverse=True)
    score, idx, box_s, box_o, best_gt = valid[0]

    # target bbox en 640×640
    target_bbox = [(box_s[0],box_s[1]), (box_s[2],box_s[1]),
                   (box_s[2],box_s[3]), (box_s[0],box_s[3])]

    # D-RISE
    sal_out = drise.forward(t_small,
                            target_class_indices=[YOLO_HUMAN_CLASS_ID],
                            target_bbox=target_bbox)
    if isinstance(sal_out,(list,tuple)):
        sal_out = sal_out[0]

    sal_small = sal_out[YOLO_HUMAN_CLASS_ID]
    sal_big   = cv2.resize(sal_small, (W0,H0))
    sal_norm  = (sal_big-sal_big.min())/(sal_big.max()-sal_big.min()+1e-12)

    # VISUAL
    heat = cv2.applyColorMap((sal_norm*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(bgr,0.6,heat,0.4,0)
    x1,y1,x2,y2 = map(int, box_o)
    cv2.rectangle(overlay,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.putText(overlay,f"score={score:.2f}",(x1,max(0,y1-8)),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    out_img = os.path.join(OUTPUT_DIR, f"{os.path.splitext(imgname)[0]}_human.png")
    cv2.imwrite(out_img, overlay)

    # METRICAS
    dc  = deletion_correlation(rgb, rgb_small, model_attr, box_o, YOLO_HUMAN_CLASS_ID, sal_norm)
    ic  = insertion_correlation(rgb, rgb_small, model_attr, box_o, YOLO_HUMAN_CLASS_ID, sal_norm)
    sp  = sparsity(sal_norm)
    pg  = compute_pointing_game(sal_norm, best_gt)
    eb  = compute_EBPG(sal_norm, best_gt)

    rows.append({
        "Image": imgname,
        "Human_score": float(score),
        "DC": float(dc),
        "IC": float(ic),
        "Sparsity": float(sp),
        "PG": int(pg),
        "EBPG": float(eb)
    })

    torch.cuda.empty_cache()
    gc.collect()


df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTPUT_DIR,"metrics.csv"),index=False)
print("✔ Saliency + métricas generadas correctamente.")