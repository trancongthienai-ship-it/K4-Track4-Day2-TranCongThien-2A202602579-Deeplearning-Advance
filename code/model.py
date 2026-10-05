"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Giao diện bạn phải giữ:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
"""
from __future__ import annotations

import torch
import torch.nn as nn
import timm

# Gợi ý backbone (GUIDE.md mục 2.1). Tag trọng số của timm có thể đổi theo phiên bản:
# dùng timm.list_pretrained("resnet50*") để xem, và GHI LẠI tag bạn dùng trong results.xlsx.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",      # hoặc vit_small_patch16_224
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",        # mạng nhẹ
    "mobilenetv3": "mobilenetv3_large_100",      # mạng nhẹ
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp."""
    if init == "scratch":
        pretrained = False
        
    actual_name = SUGGESTED_BACKBONES.get(name, name)
    
    model = timm.create_model(
        actual_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate
    )
    
    if init == "frozen":
        freeze_backbone(model)
        
    return model


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head."""
    head = model.get_classifier()
    
    # Lấy các tham số thuộc về head
    head_params = set(head.parameters())
    
    for name, param in model.named_parameters():
        if param not in head_params:
            param.requires_grad = False


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """Chia tham số thành 3 nhóm như slide Day 2, trang 52."""
    head = model.get_classifier()
    head_params = set(head.parameters())
    
    backbone_decay = []
    backbone_no_decay = []
    head_group = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        if param in head_params:
            head_group.append(param)
        else:
            if param.ndim <= 1:
                backbone_no_decay.append(param)
            else:
                backbone_decay.append(param)
                
    groups = []
    if len(backbone_decay) > 0:
        groups.append({"params": backbone_decay, "lr": lr_backbone, "weight_decay": weight_decay})
    if len(backbone_no_decay) > 0:
        groups.append({"params": backbone_no_decay, "lr": lr_backbone, "weight_decay": 0.0})
    if len(head_group) > 0:
        groups.append({"params": head_group, "lr": lr_head, "weight_decay": weight_decay})
        
    return groups


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    total_params = sum(p.numel() for p in model.parameters())
    return total_params / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size (slide tính MAC, không phải FLOPs 2x)."""
    try:
        from fvcore.nn import FlopCountAnalysis, flop_count_str
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
            flops = FlopCountAnalysis(model, dummy_input)
            macs = flops.total()
            return macs / 1e9
    except ImportError:
        try:
            from thop import profile
            device = next(model.parameters()).device
            dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
            macs, _ = profile(model, inputs=(dummy_input, ), verbose=False)
            return macs / 1e9
        except ImportError:
            print("Both fvcore and thop are not installed. Returning 0.")
            return 0.0
