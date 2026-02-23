import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from ultralytics import YOLO

from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import slic
from scipy.stats import pearsonr

# ============================
# CONFIGURACIÓN
# ============================
device = 'cpu'
model_path = 'best.pt'
datadir = './imagenes_testear/'
saliency_map_dir = 'LIME_METRICAS_YOLO_BUENO'
conf_thre = 0.5

height, width = 480, 640
num_classes = 12
target_classes = list(range(num_classes))

os.makedirs(saliency_map_dir, exist_ok=True)

# ============================
# MODELO YOLO
# ============================
model = YOLO(model_path)
print("✅ Modelo YOLO cargado")

# ============================
# UTILIDADES
# ============================
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-8)

def predict_yolo(model, img, conf=0.0):
    r = model(img, conf=conf, iou=0.7, max_det=50,device=0 if device=='cuda' else 'cpu', verbose=False)[0]
    if r.boxes is None:
        return np.empty((0,4)), np.array([]), np.array([])
    return (
        r.boxes.xyxy.cpu().numpy(),
        r.boxes.conf.cpu().numpy(),
        r.boxes.cls.cpu().numpy().astype(int)
    )

# ============================
# MÉTRICAS
# ============================
def deletion_correlation(img, model, orig_box, cls_id, sal):
    segments = slic(img.astype(float)/255, n_segments=100, start_label=0)
    seg_ids = np.unique(segments)
    scores = [sal[segments==s].mean() for s in seg_ids]
    order = np.argsort(scores)[::-1]

    pert = img.copy()
    confs, salvals = [], []

    for i in order:
        pert[segments == seg_ids[i]] = 0
        boxes, scs, cls = predict_yolo(model, pert)
        best = 0.0
        for b, s, c in zip(boxes, scs, cls):
            if c == cls_id and iou(orig_box, b) > 0.3:
                best = max(best, s)
        confs.append(best)
        salvals.append(scores[i])

    if len(confs) < 2:
        return 0.0
    v = np.diff(confs) * -1
    return pearsonr(v, salvals[:-1])[0] if np.std(v) else 0.0

def insertion_correlation(img, model, orig_box, cls_id, sal):
    segments = slic(img.astype(float)/255, n_segments=100, start_label=0)
    seg_ids = np.unique(segments)
    scores = [sal[segments==s].mean() for s in seg_ids]
    order = np.argsort(scores)[::-1]

    pert = cv2.GaussianBlur(img, (21,21), 0)
    confs, salvals = [], []

    for i in order:
        pert[segments == seg_ids[i]] = img[segments == seg_ids[i]]
        boxes, scs, cls = predict_yolo(model, pert)
        best = 0.0
        for b, s, c in zip(boxes, scs, cls):
            if c == cls_id and iou(orig_box, b) > 0.3:
                best = max(best, s)
        confs.append(best)
        salvals.append(scores[i])

    if len(confs) < 2:
        return 0.0
    v = np.diff(confs)
    return pearsonr(v, salvals[1:])[0] if np.std(v) else 0.0

def sparsity(sal):
    return 1 / (sal.mean() + 1e-8)

def pointing_game(sal, gt):
    y, x = np.unravel_index(np.argmax(sal), sal.shape)
    x1,y1,x2,y2 = map(int, gt)
    return int(x1 <= x <= x2 and y1 <= y <= y2)

def ebpg(sal, gt):
    x1,y1,x2,y2 = map(int, gt)
    s = sal / (sal.sum() + 1e-8)
    return s[y1:y2+1, x1:x2+1].sum()

# ============================
# COCO GT
# ============================
def load_coco_gt(path, img_name):
    with open(path) as f:
        coco = json.load(f)
    img_id = [i["id"] for i in coco["images"] if i["file_name"] == img_name][0]
    gts = []
    for a in coco["annotations"]:
        if a["image_id"] == img_id:
            x,y,w,h = a["bbox"]
            gts.append([x,y,x+w,y+h])
    return gts

def match_gt(pred, gts, thr=0.5):
    best = None
    best_iou = 0
    for g in gts:
        i = iou(pred, g)
        if i > best_iou:
            best, best_iou = g, i
    return best if best_iou >= thr else None

# ============================
# LIME
# ============================
explainer = lime_image.LimeImageExplainer()
segmenter = SegmentationAlgorithm('slic', n_segments=200)

def lime_wrapper(model, cls, n_cls):
    def f(imgs):
        res = []
        for img in imgs:
            _, scs, cls_ids = predict_yolo(model, img)
            c = scs[cls_ids == cls]
            p = c.mean() if len(c) else 1/n_cls
            v = np.full(n_cls, (1-p)/(n_cls-1))
            v[cls] = p
            res.append(v)
        return np.array(res)
    return f

# ============================
# LOOP PRINCIPAL
# ============================
results = []

for img_file in os.listdir(datadir):
    if not img_file.endswith(('.jpg','.png')):
        continue

    img_path = os.path.join(datadir, img_file)
    img = np.array(Image.open(img_path).resize((width,height)))

    boxes, scores, labels = predict_yolo(model, img, conf_thre)

    gt_json = f"instances_default_{os.path.splitext(img_file)[0]}.json"
    gts = load_coco_gt(gt_json, img_file)

    for i,(b,s,c) in enumerate(zip(boxes, scores, labels)):
        if s < conf_thre or c not in target_classes:
            continue

        explanation = explainer.explain_instance(
            img,
            lime_wrapper(model, c, num_classes),
            num_samples=1000,
            segmentation_fn=segmenter
        )

        sal = np.zeros(img.shape[:2])
        for sp,w in explanation.local_exp[c]:
            sal[explanation.segments == sp] = w
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)

        dc = deletion_correlation(img, model, b, c, sal)
        ic = insertion_correlation(img, model, b, c, sal)
        sp = sparsity(sal)

        gt = match_gt(b, gts)
        pg = pointing_game(sal, gt) if gt else 0
        e = ebpg(sal, gt) if gt else 0

        results.append({
            "Image": img_file,
            "BBox": i,
            "Class": c,
            "Score": s,
            "DC": dc,
            "IC": ic,
            "Sparsity": sp,
            "PG": pg,
            "EBPG": e
        })

        # Guardar visualización
        fig,ax = plt.subplots()
        ax.imshow(img)
        ax.imshow(sal, cmap='jet', alpha=0.5)
        x1,y1,x2,y2 = b
        ax.add_patch(patches.Rectangle((x1,y1),x2-x1,y2-y1,
                                       edgecolor='red', fill=False, lw=2))
        plt.axis('off')
        plt.savefig(f"{saliency_map_dir}/{img_file}_bb{i}.png",
                    bbox_inches='tight')
        plt.close()

# ============================
# CSV FINAL
# ============================
df = pd.DataFrame(results)
df.to_csv(f"{saliency_map_dir}/metrics_LIME_YOLO.csv", index=False)
print("✅ CSV guardado")
