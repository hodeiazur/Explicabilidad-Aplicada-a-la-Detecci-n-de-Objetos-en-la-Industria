import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection import FasterRCNN, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import box_iou
torch.backends.cudnn.benchmark = True  # Acelera convoluciones en GPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Usando dispositivo: {DEVICE}")

CATEGORY_MAP = {
    "human": 1
}

class CocoDataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        with open(ann_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = self.coco_data["images"]
        self.annotations = self.coco_data["annotations"]

        # ---- CREAR MAPA ID_ORIGINAL -> ID_NORMALIZADO ----
        self.id_map = {}
        for cat in self.coco_data.get("categories", []):
            name = cat["name"].lower()
            if name in CATEGORY_MAP:
                self.id_map[cat["id"]] = CATEGORY_MAP[name]

        # ---- AGRUPAR ANOTACIONES POR IMAGEN ----
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            self.img_to_anns.setdefault(img_id, []).append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.img_dir, img_info["file_name"])

        # cargar imagen
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f" Error con {img_info['file_name']}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        anns = self.img_to_anns.get(img_id, [])

        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in anns:
            # ignorar categorías desconocidas
            if ann["category_id"] not in self.id_map:
                continue

            # convertir bbox COCO -> x1 y1 x2 y2
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue  # caja inválida

            boxes.append([x, y, x + w, y + h])
            labels.append(self.id_map[ann["category_id"]])
            areas.append(ann.get("area", w * h))
            iscrowd.append(ann.get("iscrowd", 0))

        # manejar imágenes sin anotaciones
        if len(boxes) == 0:
            target = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
                "area": torch.zeros((0,), dtype=torch.float32),
                "iscrowd": torch.zeros((0,), dtype=torch.int64),
                "image_id": torch.tensor([img_id])
            }
        else:
            target = {
                "boxes": torch.as_tensor(boxes, dtype=torch.float32),
                "labels": torch.as_tensor(labels, dtype=torch.int64),
                "area": torch.as_tensor(areas, dtype=torch.float32),
                "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
                "image_id": torch.tensor([img_id])
            }

        if self.transforms:
            img = self.transforms(img)

        return img, target

class MultiCocoDataset(Dataset):
    def __init__(self, dataset_paths, transforms=None):
        """
        dataset_paths: lista de tuplas (img_dir, ann_file)
        """
        self.datasets = [CocoDataset(img_dir, ann_file, transforms) for img_dir, ann_file in dataset_paths]
        self.lengths = [len(d) for d in self.datasets]

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, idx):
        for d, l in zip(self.datasets, self.lengths):
            if idx < l:
                return d[idx]
            idx -= l
def create_datasets(root_dir, val_folder_idx=0, transform=None):
    """
    root_dir: carpeta que contiene las 11 subcarpetas (cada una con sus imágenes + annotations.json)
    val_folder_idx: índice (0–12) de la carpeta que será usada para validación
    """
    folders = sorted(os.listdir(root_dir))
    all_paths = [
        (os.path.join(root_dir, f), os.path.join(root_dir, f, "instances_default.json"))
        for f in folders
    ]

    val_paths = [all_paths[val_folder_idx]]
    train_paths = [p for i, p in enumerate(all_paths) if i != val_folder_idx]

    train_dataset = MultiCocoDataset(train_paths)
    val_dataset = MultiCocoDataset(val_paths)
    return train_dataset, val_dataset

def get_model(num_classes=2):
    """
    Devuelve un Faster-RCNN preentrenado en COCO (human + fondo),
    listo para fine-tuning con tus imágenes sintéticas.
    """
    # Cargar modelo Faster-RCNN preentrenado en COCO
    model = fasterrcnn_resnet50_fpn(weights="COCO_V1") # porque el de coco si que tiene humanos no como el de imagenetik

    # Reemplazar el predictor final para solo NUM_CLASSES (human + fondo)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model
# def get_model(num_classes):
#     backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
#     return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
#     in_channels = [64, 128, 256, 512]
#     out_channels = 256

#     backbone_with_fpn = BackboneWithFPN(
#         backbone,
#         return_layers=return_layers,
#         in_channels_list=in_channels,
#         out_channels=out_channels
#     )

#     model = FasterRCNN(backbone_with_fpn, num_classes=num_classes)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model

def evaluate_model(model, dataloader, device):
    model.eval()
    ious, aps = [], []
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            # Convierte a tensor si no lo es y mueve a device
            images = [img if isinstance(img, torch.Tensor) else transforms.ToTensor()(img) for img in images]
            images = [img.to(device) for img in images]

            preds = model(images)
            for pred, target in zip(preds, targets):
                if len(pred["boxes"]) == 0 or len(target["boxes"]) == 0:
                    continue
                iou = box_iou(pred["boxes"].cpu(), target["boxes"])
                max_iou, _ = iou.max(dim=1)
                ious.extend(max_iou.tolist())
                aps.append((max_iou > 0.5).float().mean().item())
    return np.mean(ious) if ious else 0, np.mean(aps) if aps else 0

import numpy as np
import torchvision.transforms.functional as F

class MotionBlur(object):
    """
    Aplica motion blur simulando movimiento horizontal o vertical.
    Cuando se mueve rápido que se ve como borroso
    """
    def __init__(self, kernel_size=7, direction="horizontal"):
        self.kernel_size = kernel_size
        self.direction = direction

    def __call__(self, img):
        img_np = np.array(img).astype(np.float32)

        # Crear kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        if self.direction == "horizontal":
            kernel[self.kernel_size // 2, :] = 1.0 / self.kernel_size
        else:
            kernel[:, self.kernel_size // 2] = 1.0 / self.kernel_size

        # Aplicar convolución (simple)
        import cv2
        blurred = cv2.filter2D(img_np, -1, kernel)

        return Image.fromarray(blurred.astype(np.uint8))


class PixelDropout(object):
    """
    Elimina píxeles aleatoriamente (occlusion augmentation).
    Fallos del sensor (por si se apaga en un segundo)
    Para sombras o reflejos pequeños (puede aportar en nuestro caso)
    """
    def __init__(self, dropout_prob=0.02):
        self.dropout_prob = dropout_prob

    def __call__(self, img):
        img_np = np.array(img)
        mask = np.random.rand(*img_np.shape[:2]) < self.dropout_prob
        img_np[mask] = 0  # Poner los píxeles a negro
        return Image.fromarray(img_np)

def collate_fn(batch):
    return tuple(zip(*batch))

def train():
    NUM_CLASSES = 2  # humano + fondo
    root_dir = "videos"
    val_idx = 3  # usa la carpeta nº3 para validación/test
    #transform = transforms.Compose([transforms.ToTensor()])

    #data augmentation para mejorar la precisión en imágenes reales

    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.4, contrast=0.2, saturation=0.2, hue=0.02),
        MotionBlur(kernel_size=7, direction="horizontal"),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        PixelDropout(dropout_prob=0.02),
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    print(" Cargando datasets...")
    train_dataset, val_dataset = create_datasets(root_dir, val_folder_idx=val_idx, transform=None)

    train_dataset.transforms = train_transform
    val_dataset.transforms = val_transform
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2,
                              collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=2,
                            collate_fn=collate_fn, pin_memory=True)

    model = get_model(NUM_CLASSES).to(DEVICE)
    # Congelar backbone (solo primeros epochs)
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 10
    best_map = 0


    print(" Entrenando modelo Faster R-CNN en GPU...\n")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            imgs = [img if isinstance(img, torch.Tensor) else transforms.ToTensor()(img) for img in imgs]
            imgs = [img.to(DEVICE) for img in imgs]

            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            loss = sum(loss for loss in loss_dict.values())
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        mean_iou, mean_ap = evaluate_model(model, val_loader, DEVICE)
        print(f" Epoch {epoch+1}: Loss={total_loss:.4f} | IoU={mean_iou:.3f} | mAP={mean_ap:.3f}")

        if mean_ap > best_map:
            best_map = mean_ap
            save_path = f"modelos_faster/best_fasterrcnn_data_augmentation_transferLearning_SOLO_HUMANO_conReal.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Modelo guardado en: {save_path}")

            print(f" Nuevo mejor modelo guardado (mAP={best_map:.3f})")

    print(" Entrenamiento completado.")

if __name__ == "__main__":
    train()