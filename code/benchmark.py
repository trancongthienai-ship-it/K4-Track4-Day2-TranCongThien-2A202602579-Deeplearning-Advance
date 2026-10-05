"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Quy tắc đo (vi phạm bị trừ điểm, RUBRIC mục 3):
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU: torch.cuda.synchronize() (hoặc CUDA event) TRƯỚC và SAU đoạn cần đo
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ GPU, dtype (FP32/AMP/FP16), batch, độ phân giải, có/không gộp BN, phiên bản torch
  - chọn và ghi rõ có tính tiền xử lý hay không
"""
from __future__ import annotations

import time
import torch
import numpy as np
from torch.cuda.amp import autocast


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây."""
    for _ in range(warmup):
        fn()
        
    times = []
    for _ in range(iters):
        if sync:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
        
    return {
        "p50": np.percentile(times, 50),
        "p95": np.percentile(times, 95),
        "p99": np.percentile(times, 99),
        "mean": np.mean(times),
        "n": iters
    }


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên."""
    model.eval()
    model = model.to(device)
    
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()
        
    sync_fn = torch.cuda.synchronize if device.startswith("cuda") else None
    
    @torch.inference_mode()
    def fn():
        if dtype == "amp":
            with autocast():
                model(dummy_input)
        else:
            model(dummy_input)
            
    stats = bench(fn, warmup, iters, sync=sync_fn)
    
    try:
        gpu_name = torch.cuda.get_device_name() if device.startswith("cuda") else "CPU"
    except Exception:
        gpu_name = "Unknown"
        
    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": stats["p50"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "mean": stats["mean"],
        "images_per_s": batch_size / (stats["p50"] / 1000.0),
        "torch": torch.__version__
    }


def tta_latency(model, k_views: int, batch_size: int = 1, img_size: int = 224, 
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Độ trễ của TTA K view: xấp xỉ K lần một lượt chạy."""
    model.eval()
    model = model.to(device)
    
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    if dtype == "fp16":
        model = model.half()
        dummy_input = dummy_input.half()
        
    sync_fn = torch.cuda.synchronize if device.startswith("cuda") else None
    
    @torch.inference_mode()
    def fn():
        if dtype == "amp":
            with autocast():
                for _ in range(k_views):
                    model(dummy_input)
        else:
            for _ in range(k_views):
                model(dummy_input)
                
    stats = bench(fn, warmup, iters, sync=sync_fn)
    
    try:
        gpu_name = torch.cuda.get_device_name() if device.startswith("cuda") else "CPU"
    except Exception:
        gpu_name = "Unknown"
        
    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "k_views": k_views,
        "p50": stats["p50"],
        "p95": stats["p95"],
        "p99": stats["p99"],
        "mean": stats["mean"],
        "images_per_s": batch_size / (stats["p50"] / 1000.0),
        "torch": torch.__version__
    }
