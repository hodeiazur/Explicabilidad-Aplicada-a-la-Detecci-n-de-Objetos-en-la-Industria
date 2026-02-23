
import torch
import torch.nn as nn
import numpy as np
import cv2
import math

# ------------------------------
# Utilidades
# ------------------------------
def create_heatmap(output_width, output_height, p_x, p_y, sigma):
    """Máscara gaussiana normalizada [0,1] centrada en (p_x, p_y) sobre una rejilla (output_width, output_height)."""
    X1 = np.linspace(0, output_width - 1, output_width)
    Y1 = np.linspace(0, output_height - 1, output_height)
    X, Y = np.meshgrid(X1, Y1)
    D2 = (X - p_x) ** 2 + (Y - p_y) ** 2
    mask = np.exp(-D2 / (2.0 * (sigma ** 2) + 1e-12))
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-12)
    return mask

# ------------------------------
# Clase GCAME (Ultralytics)
# ------------------------------
class GCAME(object):
    def __init__(self, model, arch="ultralytics", img_size=(640, 640)):
        """
        model puede ser:
         - ultralytics.YOLO (v8+)
         - un objeto con .model (tasks-like BaseModel) o ._predict_once
         - nn.Module crudo
        """
        assert arch == "ultralytics", "Este archivo implementa la variante Ultralytics YOLO."
        self.tm = self._unwrap_to_tasks_like(model)  # objeto "tasks-like" que orquesta el forward
        self.arch = arch
        self.img_size = img_size
        self.activations = {}   # P3/P4/P5 capturadas en forward hook
        self.handlers = []
        self.detect_module = None

        # Registrar SÓLO forward hook en el grafo interno (donde vive Detect)
        self._register_ultralytics_forward_hook()

        # Empezamos en eval; durante la explicación activaremos train() para autograd
        self._set_mode(eval_mode=True)

    # ---------- utilidades de modo ----------
    def _set_mode(self, eval_mode=True):
        target = self._raw_graph_root()
        if eval_mode:
            target.eval()
        else:
            target.train()

    # ---------- obtener “tasks-like” y raíz del grafo ----------
    def _unwrap_to_tasks_like(self, m):
        """
        Devuelve un objeto que:
          - tenga _predict_once(img) -> ideal para forward crudo
          - o, si no, tenga .model con _predict_once(img)
          - si ninguna, usaremos su .forward(img) directamente
        Además, guarda referencias para localizar el grafo crudo (.model si existe).
        """
        obj = m
        # Si es el wrapper YOLO, suele tener .model (tasks-like BaseModel)
        if hasattr(obj, "model"):
            obj = obj.model
        # Si ese objeto aún tiene .model (el nn.Sequential crudo), lo retenemos
        self._maybe_raw = getattr(obj, "model", None)
        # Target preferente: el propio obj si tiene _predict_once
        self._predict_target = obj if hasattr(obj, "_predict_once") else None
        return obj

    def _raw_graph_root(self):
        """
        Devuelve el nn.Module raíz donde registraremos hooks.
        Preferimos .model (grafo interno); si no existe, usamos el propio obj tasks-like.
        """
        return self._maybe_raw if isinstance(self._maybe_raw, nn.Module) else self.tm

    # ---------- registro de hooks ----------
    def _register_ultralytics_forward_hook(self):
        root = self._raw_graph_root()
        # localizar Detect dentro del grafo
        detect = None
        for name, m in root.named_modules():
            if m.__class__.__name__.lower() == "detect":
                detect = m
                break
        if detect is None:
            raise RuntimeError("No se encontró módulo 'Detect' en el grafo.")
        self.detect_module = detect

        def fwd_hook(module, inp, out):
            # inp[0] suele ser lista/tupla [P3, P4, P5] de features multiescala
            feats = inp[0]
            self.activations.clear()
            for i, f in enumerate(feats):
                key = f"P{i+3}"  # P3, P4, P5
                self.activations[key] = f  # NO retain_grad(); haremos leaf con clone+requires_grad

        # Registrar SOLO forward hook (evitamos warnings de backward con outputs tipo list)
        self.handlers.append(self.detect_module.register_forward_hook(fwd_hook))

    # ---------- helpers ----------
    def _preprocess(self, img_np, device):
        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=-1)
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
        return img_t.unsqueeze(0).to(device)

    def _stride_from_map(self, H, W, h, w):
        """Stride (x,y) estimado por escala a partir de la relación imagen:feature."""
        sx = W / float(w)
        sy = H / float(h)
        return sx, sy

    def _choose_scale(self, H, W, box_xyxy):
        """
        Selecciona la escala (P3/P4/P5) más adecuada en función del tamaño del bbox.
        """
        x1, y1, x2, y2 = map(float, box_xyxy)
        bw, bh = (x2 - x1), (y2 - y1)
        best_key, best_stride, best_score = None, (8.0, 8.0), float("inf")
        for key, fmap in self.activations.items():
            _, C, h, w = fmap.shape
            sx, sy = self._stride_from_map(H, W, h, w)
            cw, ch = bw / sx, bh / sy  # tamaño del bbox en celdas de esa escala
            score = max(cw, ch)
            if score < 6:
                score += (6 - score)
            elif score > 40:
                score += (score - 40)
            if score < best_score:
                best_score = score
                best_key = key
                best_stride = (sx, sy)
        return best_key, best_stride

    def _sigma_gcame(self, H, W, h, w):
        """σ robusto a escala (aprox inspirado en la sección 3.4 del paper)."""
        S = math.sqrt((H * W) / (h * w + 1e-6))
        kn_size = max(int((math.sqrt(h * w) - 1) // 2) // 3, 1)
        sigma = max(abs((math.log(S + 1e-6) / kn_size)), 1.0)
        return float(sigma)

    # ---------- forward explicativo ----------
    def forward_ultralytics_yolo(self, img_np, box_xyxy, cls_id=None, device=None):
        """
        Genera el mapa de saliencia G-CAME para una bbox.
        Nota: hacemos el forward interno para poblar P3/P4/P5 y luego
              creamos un leaf tensor (clone+requires_grad) sobre la escala elegida.
        """
        eps = 1e-7
        H, W = img_np.shape[:2]
        device = device or next(self._raw_graph_root().parameters()).device

        # habilitar autograd y usar el forward interno (sin predict/nms)
        torch.set_grad_enabled(True)
        self._set_mode(eval_mode=False)  # train() para mantener grafo (aunque luego haremos leaf)
        self._raw_graph_root().zero_grad(set_to_none=True)

        img_t = self._preprocess(img_np, device)

        # 1) forward: preferir _predict_once si existe; si no, usar .forward()
        if self._predict_target is not None:
            _ = self._predict_target._predict_once(img_t)
        else:
            _ = self._raw_graph_root()(img_t)  # fallback seguro

        # 2) selección de escala y centro
        x1, y1, x2, y2 = map(float, box_xyxy)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        key, (sx, sy) = self._choose_scale(H, W, box_xyxy)
        fmap = self.activations[key]           # Tensor (1, C, h, w)
        _, C, h, w = fmap.shape
        gx = np.clip(cx / sx, 0, w - 1)
        gy = np.clip(cy / sy, 0, h - 1)

        # 3) Creamos un LEAF con gradiente a partir de fmap (para evitar problemas de grafo/list)
        fmap_leaf = fmap.detach().clone().requires_grad_(True)   # (1, C, h, w)

        # 4) objetivo escalar: energía cuadrática bajo gaussiana en esa escala
        sigma = self._sigma_gcame(H, W, h, w)
        mask = create_heatmap(w, h, gx, gy, sigma)
        mask_t = torch.from_numpy(mask).to(device).float()
        score = ((fmap_leaf.squeeze(0) ** 2) * mask_t[None, :, :]).sum()

        # 5) backward sobre el LEAF para obtener gradiente canal-a-canal
        # (no dependemos de backward hooks del módulo Detect)
        self._raw_graph_root().zero_grad(set_to_none=True)
        score.backward(retain_graph=True)

        grad = fmap_leaf.grad  # (1, C, h, w)

        # 6) saliency G‑CAME: α+ / α− y máscara gaussiana
        fmap_np = fmap_leaf.detach().cpu().squeeze(0).numpy()     # (C, h, w)
        grad_np = grad.detach().cpu().squeeze(0).numpy()          # (C, h, w)
        pos_grad = grad_np.copy(); pos_grad[pos_grad < 0] = 0
        neg_grad = grad_np.copy(); neg_grad[neg_grad > 0] = 0

        sal = np.zeros((h, w), dtype=np.float32)
        for j in range(C):
            new_map = fmap_np[j]
            mean_pos = float(pos_grad[j].mean())
            mean_neg = float(neg_grad[j].mean())
            sal += (new_map * mean_pos - new_map * abs(mean_neg)) * mask

        sal = cv2.resize(sal, (W, H), interpolation=cv2.INTER_LINEAR)
        sal[sal < 0.0] = 0.0

        # normalización robusta (recorte percentil 80 para evitar azules planos)
        # p80 = np.percentile(sal, 80)
        # sal = np.clip(sal, 0, p80)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + eps)

        # volver a eval
        self._set_mode(eval_mode=True)
        torch.set_grad_enabled(False)
        return sal
