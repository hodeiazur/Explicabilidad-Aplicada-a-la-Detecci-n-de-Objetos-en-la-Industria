import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from torchvision.ops import box_iou
import random



# ============================================================
# CONFIGURACIÓN
# ============================================================

DATASET_ROOT = "."   # carpeta que contiene las _out_sdrec_01 ... _out_sdrec_15
TRAIN_FOLDERS = [f"_out_sdrec_{i:02d}" for i in range(1, 15)]  # folders 1-14
VAL_FOLDER = "_out_sdrec_15"  # Fixed validation folder
NUM_CLASSES = 11 + 1  # 10 objetos + fondo
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DATASET
# ============================================================

class DetectionDataset(Dataset):
    def __init__(self, folders, transforms=None):
        self.samples = []
        self.transforms = transforms
        for folder in folders:
            folder_path = os.path.join(DATASET_ROOT, folder)
            if not os.path.isdir(folder_path):
                continue

            # Detectamos la imagen RGB y las cajas
            rgb_files = [f for f in os.listdir(folder_path) if f.startswith("rgb_") and f.endswith(".png")]
            for rgb in rgb_files:
                base = rgb.split("_")[-1].split(".")[0]
                box_file = os.path.join(folder_path, f"bounding_box_2d_tight_{base}.npy")
                if os.path.exists(box_file):
                    self.samples.append((os.path.join(folder_path, rgb), box_file))

            # Cargamos mapa de clases (lo asumimos igual en todos)
            label_map_path = os.path.join(folder_path, "bounding_box_2d_tight_labels_0000.json")
            if os.path.exists(label_map_path):
                with open(label_map_path) as f:
                    self.label_map = {int(k): v["class"] for k, v in json.load(f).items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, box_path = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        data = np.load(box_path, allow_pickle=True)

        boxes, labels = [], []
        for obj in data:
            if obj['x_max'] > obj['x_min'] and obj['y_max'] > obj['y_min']:
                boxes.append([obj['x_min'], obj['y_min'], obj['x_max'], obj['y_max']])
                labels.append(int(obj['semanticId']))

        boxes = np.array(boxes)
        labels = np.array(labels)

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
            image = transformed["image"].float() / 255.0  #normalizar aquí
            boxes = torch.as_tensor(transformed["bboxes"], dtype=torch.float32)
            labels = torch.as_tensor(transformed["labels"], dtype=torch.int64)
        else:
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  #normalizar
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return image, target

# ============================================================
# TRANSFORMACIONES (Data Augmentation)
# ============================================================

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.Blur(p=0.2),
            A.MotionBlur(p=0.2),
            A.RandomGamma(p=0.3),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    else:
        return A.Compose([ToTensorV2()],
                         bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

# ============================================================
# MODELO
# ============================================================

def get_model(num_classes):
    backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    return_layers = {
        'layer1': '0',
        'layer2': '1',
        'layer3': '2',
        'layer4': '3'
    }

    in_channels = [64, 128, 256, 512]
    out_channels = 256 #iual jetxi erdia, 256

    backbone_with_fpn = BackboneWithFPN(
        backbone,
        return_layers=return_layers,
        in_channels_list=in_channels,
        out_channels=out_channels,
    )

    model = FasterRCNN(backbone_with_fpn, num_classes=num_classes)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# ============================================================
# MÉTRICAS (IoU y mAP simple)
# ============================================================

def evaluate_model(model, dataloader, device):
    model.eval()
    ious, aps = [], []

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device) for img in images]
            preds = model(images)

            for pred, target in zip(preds, targets):
                if len(pred["boxes"]) == 0 or len(target["boxes"]) == 0:
                    continue
                iou = box_iou(pred["boxes"].cpu(), target["boxes"])
                max_iou, _ = iou.max(dim=1)
                ious.extend(max_iou.tolist())

                # cálculo mAP simple (IoU>0.5 como TP)
                aps.append((max_iou > 0.5).float().mean().item())

    mean_iou = np.mean(ious) if ious else 0
    mean_ap = np.mean(aps) if aps else 0
    return mean_iou, mean_ap

# ============================================================
# ENTRENAMIENTO
# ============================================================

def train():
    print("Cargando datasets...")
    train_dataset = DetectionDataset(TRAIN_FOLDERS, get_transforms(train=True))
    val_dataset = DetectionDataset([VAL_FOLDER], get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.005, momentum=0.9, weight_decay=0.0005
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    print("Entrenando modelo Faster R-CNN... 6\n")

    best_map = 0.0
    best_iou = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            imgs = [img.to(DEVICE) for img in imgs]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        scheduler.step()
        mean_iou, mean_ap = evaluate_model(model, val_loader, DEVICE)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {total_loss:.4f} | IoU: {mean_iou:.3f} | mAP: {mean_ap:.3f}")

        if mean_ap > best_map:
            best_map = mean_ap
            best_iou = mean_iou
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "fasterrcnn_trained_4.pth")
            print(f"Nuevo mejor modelo guardado (Epoch {best_epoch}) con mAP={best_map:.3f}, IoU={best_iou:.3f}") #así guardo el modelo de las mejores metricas y evito el overfitting
    #torch.save(model.state_dict(), "fasterrcnn_trained_6.pth")



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()
