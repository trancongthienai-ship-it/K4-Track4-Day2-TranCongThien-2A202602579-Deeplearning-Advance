"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm/lớp có `raise NotImplementedError`.
Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện bạn phải giữ:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted"."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    elif kind == "ls":
        return LabelSmoothingCE(smoothing=kw.get("smoothing", 0.1))
    elif kind == "focal":
        return FocalLoss(gamma=kw.get("gamma", 2.0), alpha=kw.get("alpha", None))
    elif kind == "ce_weighted":
        return nn.CrossEntropyLoss(weight=kw.get("weight", None))
    else:
        raise ValueError(f"Unknown kind {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing."""
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    def forward(self, logits, targets):
        return self.criterion(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp."""
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if not isinstance(alpha, torch.Tensor):
                alpha = torch.tensor(alpha, dtype=torch.float32)
        self.alpha = alpha

    def forward(self, logits, targets):
        log_pt = nn.functional.log_softmax(logits, dim=-1)
        pt = torch.exp(log_pt)
        
        log_pt_c = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt_c = pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        loss = - (1 - pt_c) ** self.gamma * log_pt_c
        
        if self.alpha is not None:
            alpha_c = self.alpha.to(targets.device)[targets]
            loss = loss * alpha_c
            
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN."""
    counts = torch.tensor(counts, dtype=torch.float32)
    if beta == 0.0:
        weights = 1.0 / counts
    else:
        weights = (1.0 - beta) / (1.0 - beta ** counts)
        
    weights = weights / weights.sum() * len(counts)
    return weights


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn một batch ảnh và nhãn."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
        
    perm = torch.randperm(x.size(0)).to(x.device)
    y_a, y_b = y, y[perm]
    
    if mode == "mixup":
        x_mix = lam * x + (1 - lam) * x[perm]
        return x_mix, (y_a, y_b, lam)
    
    elif mode == "cutmix":
        B, C, H, W = x.size()
        cut_rat = np.sqrt(1. - lam)
        rw, rh = int(W * cut_rat), int(H * cut_rat)
        
        rx = np.random.randint(W)
        ry = np.random.randint(H)
        
        x1 = np.clip(rx - rw // 2, 0, W)
        y1 = np.clip(ry - rh // 2, 0, H)
        x2 = np.clip(rx + rw // 2, 0, W)
        y2 = np.clip(ry + rh // 2, 0, H)
        
        x_mix = x.clone()
        x_mix[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        
        lam_actual = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
        return x_mix, (y_a, y_b, lam_actual)
    else:
        return x, (y, y, 1.0)


def mixed_loss(criterion, logits, targets):
    """Loss cho batch đã trộn: lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
