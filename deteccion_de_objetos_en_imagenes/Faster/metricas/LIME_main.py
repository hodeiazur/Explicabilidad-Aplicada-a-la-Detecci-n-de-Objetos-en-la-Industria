import os
import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.models.detection as detection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import pandas as pd
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import slic
from scipy.stats import pearsonr
import json

# ============================
# Configuración
# ============================
device = 'cuda'  # Cambia a 'cuda:0' si tienes GPU
datadir = './imagenes_testear/'
saliency_map_dir = "LIME-METRICAS_nseg_200"
conf_thre = 0.5
height, width = 480, 640
num_classes = 12
target_classes = [i for i in range(num_classes)]
model_path = 'fasterrcnn_trained_4.pth'
os.makedirs(saliency_map_dir, exist_ok=True)

# ============================
# Cargar modelo
# ============================
model = detection.fasterrcnn_resnet50_fpn(pretrained=False, num_classes=num_classes)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval().to(device)
print("Modelo cargado correctamente")

# ============================
# Funciones auxiliares
# ============================
def normalize_bboxes(bboxes, img_height, img_width):
    # Devuelve coordenadas normalizadas [0,1]
    norm_boxes = []
    for corners in bboxes:
        norm_c = [(x/img_width, y/img_height) for x,y in corners]
        norm_boxes.append(norm_c)
    return norm_boxes

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-8)

def predict(model, img_rgb):
    tensor = torchvision.transforms.ToTensor()(img_rgb).unsqueeze(0).to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(tensor)[0]
    return out

def deletion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map, mask_value=0, n_levels=[100,200]):
    H, W = img_rgb.shape[:2]
    img_float = img_rgb.astype(np.float32)/255.0
    segments = slic(img_float, n_segments=n_levels[0], compactness=10, start_label=0)
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]
    perturbed = img_rgb.copy()
    c_scores, s_scores = [], []
    for idx in sorted_idx:
        seg_mask = (segments == seg_vals[idx])
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = mask_value
        out_pert = predict(model, perturbed)
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(out_pert["boxes"].cpu().numpy(), out_pert["scores"].cpu().numpy(), out_pert["labels"].cpu().numpy()):
            if l != orig_class_id: continue
            this_iou = iou(orig_box, b)
            if this_iou > best_iou:
                best_iou, best_score = this_iou, s
        c_scores.append(best_score if best_iou > 0.3 else 0.0)
    if len(c_scores) < 2: return 0.0
    v = np.array(c_scores[:-1]) - np.array(c_scores[1:])
    s = np.array(s_scores[:-1])
    if np.std(v)==0 or np.std(s)==0: return 0.0
    corr, _ = pearsonr(v, s)
    return corr

def insertion_correlation(img_rgb, model, orig_box, orig_class_id, orig_score, sal_map, blur_kernel=21, n_levels=[100,200]):
    H, W = img_rgb.shape[:2]
    img_float = img_rgb.astype(np.float32)/255.0
    segments = slic(img_float, n_segments=n_levels[0], compactness=10, start_label=0)
    seg_vals = np.unique(segments)
    seg_scores = [sal_map[segments==v].mean() for v in seg_vals]
    sorted_idx = np.argsort(seg_scores)[::-1]
    perturbed = cv2.GaussianBlur(img_rgb, (blur_kernel, blur_kernel), 0)
    c_scores, s_scores = [], []
    for idx in sorted_idx:
        seg_mask = (segments == seg_vals[idx])
        s_scores.append(seg_scores[idx])
        perturbed[seg_mask] = img_rgb[seg_mask]
        out_pert = predict(model, perturbed)
        best_iou, best_score = 0.0, 0.0
        for b, s, l in zip(out_pert["boxes"].cpu().numpy(), out_pert["scores"].cpu().numpy(), out_pert["labels"].cpu().numpy()):
            if l != orig_class_id: continue
            this_iou = iou(orig_box, b)
            if this_iou > best_iou:
                best_iou, best_score = this_iou, s
        c_scores.append(best_score if best_iou > 0.3 else 0.0)
    if len(c_scores) < 2: return 0.0
    v = np.array(c_scores[1:]) - np.array(c_scores[:-1])
    s = np.array(s_scores[1:])
    if np.std(v)==0 or np.std(s)==0: return 0.0
    corr, _ = pearsonr(v, s)
    return corr

def sparsity(sal_map):
    return 1.0 / (sal_map.mean() + 1e-8)

def compute_pointing_game(sal_map, gt_box):
    x1, y1, x2, y2 = map(int, gt_box)
    max_y, max_x = np.unravel_index(np.argmax(sal_map), sal_map.shape)
    hit = (x1 <= max_x <= x2) and (y1 <= max_y <= y2)
    return 1 if hit else 0

def compute_EBPG(sal_map, gt_box):
    x1, y1, x2, y2 = map(int, gt_box)
    S = sal_map.astype(np.float32)
    S_norm = S / (S.sum() + 1e-12)
    sub = S_norm[y1:y2+1, x1:x2+1]
    return float(sub.sum())

def match_prediction_to_gt(pred_box, gt_boxes, iou_thresh=0.5):
    best_gt, best_iou_val = None, 0.0
    for gt in gt_boxes:
        val = iou(pred_box, gt)
        if val > best_iou_val:
            best_iou_val = val
            best_gt = gt
    if best_gt is None or best_iou_val < iou_thresh: return None
    return best_gt

def load_coco_gt(json_path, image_filename):
    with open(json_path,'r') as f: coco=json.load(f)
    img_id=None
    for img in coco["images"]:
        if img["file_name"]==image_filename: img_id=img["id"]; break
    if img_id is None: raise ValueError(f"No se encontró {image_filename}")
    gt_boxes, gt_labels=[],[]
    for ann in coco["annotations"]:
        if ann["image_id"]==img_id:
            x,y,w,h=ann["bbox"]
            gt_boxes.append([int(x),int(y),int(x+w),int(y+h)])
            gt_labels.append(int(ann["category_id"]))
    return gt_boxes, gt_labels

# ============================
# LIME setup
# ============================
explainer = lime_image.LimeImageExplainer()
segmenter = SegmentationAlgorithm('slic', n_segments=200) #con n_segments = 1_000 va super mal

def get_probab_class_wrapper(img_np, model, target_class, num_classes):
    def get_probab_class(imgs):
        with torch.no_grad():
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
            batch = [transform(Image.fromarray(img)).to(next(model.parameters()).device) for img in imgs]
            results = model(batch)
        probab_results=[]
        for result in results:
            conf_scores=[float(result['scores'][i].item()) for i,label in enumerate(result['labels']) if label.item()==target_class]
            if conf_scores:
                prob_cls=np.mean(conf_scores)
                other_probs=(1-prob_cls)/(num_classes-1)
                prob_vector=np.full(num_classes, other_probs)
                prob_vector[target_class]=prob_cls
            else:
                prob_vector=np.full(num_classes, 1/num_classes)
            probab_results.append(prob_vector)
        return np.array(probab_results)
    return get_probab_class

# ============================
# Loop principal
# ============================
results_all = []

imgs_name = [os.path.splitext(f)[0] for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir,f))]

for img_name in imgs_name:
    # Cargar imagen
    img_path_jpg = os.path.join(datadir, img_name+'.jpg')
    img_path_png = os.path.join(datadir, img_name+'.png')
    img_path = img_path_jpg if os.path.exists(img_path_jpg) else img_path_png
    if not os.path.exists(img_path): continue

    orig_img = Image.open(img_path).convert("RGB")
    resized_img = orig_img.resize((width,height), Image.LANCZOS)
    img_np = np.array(resized_img)

    tensor = torchvision.transforms.ToTensor()(resized_img).unsqueeze(0).to(device)
    results_model = model(tensor)[0]
    labels = results_model['labels'].cpu().numpy()

    # Cargar GT
    json_path = f"instances_default_{img_name}.json"
    gt_boxes, gt_labels = load_coco_gt(json_path, os.path.basename(img_path))

    for _b, box in enumerate(results_model['boxes']):
        _score = float(results_model['scores'][_b].item())
        _target_class = int(results_model['labels'][_b].item())

        if _score < conf_thre or _target_class not in target_classes: continue

        x1, y1, x2, y2 = box.detach().cpu().numpy()
        # LIME explanation
        explanation = explainer.explain_instance(
            image=img_np,
            classifier_fn=get_probab_class_wrapper(img_np, model, _target_class, num_classes),
            top_labels=1, hide_color=0, num_samples=1000,
            segmentation_fn=segmenter
        )
        segments = explanation.segments
        heatmap = np.zeros(img_np.shape[:2])
        for sp, w in explanation.local_exp[_target_class]:
            heatmap[segments==sp] = w
        heatmap = (heatmap - heatmap.min()) / (heatmap.max()-heatmap.min()+1e-8)

        # Guardar heatmap .npy
        np.save(os.path.join(saliency_map_dir,f"{img_name}_class{_target_class}_bb{_b}.npy"), heatmap)

        # Calcular métricas
        dc = deletion_correlation(img_np, model, [x1,y1,x2,y2], _target_class, _score, heatmap)
        ic = insertion_correlation(img_np, model, [x1,y1,x2,y2], _target_class, _score, heatmap)
        sp = sparsity(heatmap)
        gt_box = match_prediction_to_gt([x1,y1,x2,y2], gt_boxes)
        pg = compute_pointing_game(heatmap, gt_box) if gt_box is not None else 0
        ebpg = compute_EBPG(heatmap, gt_box) if gt_box is not None else 0.0

        results_all.append({
            "Image": img_name,
            "BBox": _b,
            "Class": _target_class,
            "Score": _score,
            "DC": dc,
            "IC": ic,
            "Sparsity": sp,
            "PG_Hit": pg,
            "EBPG": ebpg
        })

        # Guardar visualización
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img_np)
        ax.imshow(heatmap, cmap='jet', alpha=0.5)
        rect = patches.Rectangle((x1,y1),x2-x1,y2-y1,linewidth=2,edgecolor='red',facecolor='none')
        ax.add_patch(rect)
        ax.set_title(f"Clase {_target_class}, BBox {_b}")
        plt.axis('off')
        plt.savefig(os.path.join(saliency_map_dir,f"{img_name}_class{_target_class}_bb{_b}.png"), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

# Guardar CSV
df = pd.DataFrame(results_all)
csv_path = os.path.join(saliency_map_dir,"metrics_LIME_completo.csv")
df.to_csv(csv_path,index=False)
print(f"CSV guardado en {csv_path}")
