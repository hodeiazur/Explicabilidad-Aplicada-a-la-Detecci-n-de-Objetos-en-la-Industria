import copy
import cv2
import math
import numpy as np
from math import floor
import torch

def create_heatmap(output_width, output_height, p_x, p_y, sigma):
    """
    Gaussian heatmap centered at (p_x, p_y) in [0,1]
    """
    X1 = np.linspace(1, output_width, output_width)
    Y1 = np.linspace(1, output_height, output_height)
    X, Y = np.meshgrid(X1, Y1)
    X = X - floor(p_x)
    Y = Y - floor(p_y)
    D2 = X*X + Y*Y
    E2 = 2.0 * sigma**2
    mask = np.exp(-D2/E2)
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask

class GCAME(object):
    def __init__(self, model, target_layers, arch="fasterrcnn", img_size=(640,640)):
        self.model = model.eval()
        self.arch = arch
        self.img_size = img_size
        self.gradients = dict()
        self.activations = dict()
        self.target_layers = target_layers
        self.handlers = []

        def save_grads(key):
            def backward_hook(module, grad_inp, grad_out):
                self.gradients[key] = grad_out[0].detach()
            return backward_hook

        def save_fmaps(key):
            def forward_hook(module, inp, output):
                self.activations[key] = output.detach()
            return forward_hook

        for name, module in list(self.model.named_modules())[1:]:
            if name in self.target_layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def remove_hook(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, img, box, obj_idx=None):
        if self.arch == "fasterrcnn":
            return self.forward_fasterrcnn(img, box, obj_idx)
        else:
            raise NotImplementedError("Only fasterrcnn implemented.")

    def forward_fasterrcnn(self, img, box, index=None):
        """
        img: [C,H,W] tensor
        box: [x1,y1,x2,y2] of the object
        """
        eps = 1e-7
        c,h,w = img.shape
        org_size = (h,w)

        self.model.zero_grad()
        output = self.model([img])
        output[0]['scores'][index].backward(retain_graph=True)

        x1,y1,x2,y2 = map(int, box)
        roi_h = y2 - y1 + 1
        roi_w = x2 - x1 + 1

        score_saliency_map = np.zeros(org_size, dtype=np.float32)

        for layer in self.target_layers:
            if layer not in self.activations or layer not in self.gradients:
                continue

            act = self.activations[layer].squeeze().cpu().numpy()  # (C,H_ROI,W_ROI)
            grad = self.gradients[layer].squeeze().cpu().numpy()   # (C,H_ROI,W_ROI)

            # Grad-CAM
            weights = np.mean(grad, axis=(1,2))  # (C,)
            cam = np.zeros_like(act[0], dtype=np.float32)
            for i, w in enumerate(weights):
                cam += w * act[i]
            cam = np.maximum(cam, 0)  # ReLU

            # Redimensionar al ROI
            cam_resized = cv2.resize(cam, (roi_w, roi_h))

            # Mapear al tamaño completo
            score_saliency_map[y1:y2+1, x1:x2+1] += cam_resized

        score_saliency_map = (score_saliency_map - score_saliency_map.min()) / (score_saliency_map.max() - score_saliency_map.min() + eps)
        score_saliency_map = 1.0 - score_saliency_map

        return score_saliency_map
