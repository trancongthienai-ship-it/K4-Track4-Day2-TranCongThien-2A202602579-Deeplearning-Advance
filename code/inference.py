"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Giao diện bạn nên giữ:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast
import torch.nn.functional as F
from scipy.optimize import minimize


def predict_logits(model, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_labels = []
    all_logits = []
    
    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            if view is not None:
                images = view(images)
                
            with autocast():
                logits = model(images)
                
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_filenames.extend(filenames)
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_filenames, all_labels, all_logits


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x, crop: int):
    """5 crop (4 góc + giữa) kích thước `crop`, và tuỳ chọn thêm bản lật. Trả về list các batch."""
    N, C, H, W = x.size()
    crops = []
    
    # 4 corners
    crops.append(x[:, :, :crop, :crop]) # top-left
    crops.append(x[:, :, :crop, W-crop:]) # top-right
    crops.append(x[:, :, H-crop:, :crop]) # bottom-left
    crops.append(x[:, :, H-crop:, W-crop:]) # bottom-right
    
    # Center
    center_y, center_x = H // 2, W // 2
    crops.append(x[:, :, center_y - crop//2 : center_y + crop//2 + crop%2, 
                         center_x - crop//2 : center_x + crop//2 + crop%2])
                         
    return crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes`, trả về list các batch."""
    scaled_batches = []
    for s in sizes:
        scaled = F.interpolate(x, size=(s, s), mode='bilinear', align_corners=False)
        scaled_batches.append(scaled)
    return scaled_batches


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62)."""
    # logits_per_view is a list of numpy arrays, each shape (N, 9)
    logits_array = np.stack(logits_per_view, axis=0) # (K, N, 9)
    
    if space == "prob":
        # Softmax each view, then average
        max_logits = np.max(logits_array, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_array - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        avg_probs = np.mean(probs, axis=0)
        return avg_probs
    elif space == "logit":
        # Average logits, then softmax
        avg_logits = np.mean(logits_array, axis=0)
        max_logits = np.max(avg_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(avg_logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        return probs
    else:
        raise ValueError(f"Unknown space {space}")


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    probs_array = np.stack(list_of_probs, axis=0) # (M, N, 9)
    return np.mean(probs_array, axis=0)


def fit_temperature(val_logits, val_labels) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)."""
    def nll_loss(T):
        T = T[0]
        logits_t = torch.tensor(val_logits) / T
        labels_t = torch.tensor(val_labels, dtype=torch.long)
        return F.cross_entropy(logits_t, labels_t).item()
        
    res = minimize(nll_loss, x0=[1.0], bounds=[(1e-3, 10.0)])
    return float(res.x[0])


def apply_temperature(logits, T: float):
    """Trả về softmax(logits / T)."""
    if isinstance(logits, np.ndarray):
        logits_t = logits / T
        max_logits = np.max(logits_t, axis=-1, keepdims=True)
        exp_logits = np.exp(logits_t - max_logits)
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    else:
        logits_t = logits / T
        return F.softmax(logits_t, dim=-1)


def fuse_conv_bn_weights(conv, bn):
    """Hàm phụ trợ gộp trọng số của 1 cặp Conv2d và BatchNorm2d."""
    with torch.no_grad():
        w = conv.weight
        mean = bn.running_mean
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        gamma = bn.weight
        beta = bn.bias
        
        if conv.bias is not None:
            b = conv.bias
        else:
            b = mean.new_zeros(mean.shape)
            
        w_fused = w * (gamma / var_sqrt).reshape([-1, 1, 1, 1])
        b_fused = (b - mean) / var_sqrt * gamma + beta
        
        fused_conv = nn.Conv2d(
            conv.in_channels, conv.out_channels, conv.kernel_size,
            conv.stride, conv.padding, conv.dilation, conv.groups, bias=True
        )
        fused_conv.weight.data = w_fused
        fused_conv.bias.data = b_fused
        return fused_conv

def fuse_conv_bn(model):
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75)."""
    model.eval()
    
    # Simple sequential replacement logic
    def _fuse_recursive(module):
        prev_name = None
        prev_mod = None
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                prev_name = name
                prev_mod = child
            elif isinstance(child, nn.BatchNorm2d) and isinstance(prev_mod, nn.Conv2d):
                # Fuse them!
                fused = fuse_conv_bn_weights(prev_mod, child)
                setattr(module, prev_name, fused)
                setattr(module, name, nn.Identity())
                prev_name = None
                prev_mod = None
            else:
                _fuse_recursive(child)
                prev_name = None
                prev_mod = None
                
    _fuse_recursive(model)
    return model
