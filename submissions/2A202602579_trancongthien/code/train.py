"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

PSEUDO-CODE: chỉ có khung (cấu hình và quy ước đặt tên file); bạn tự hoàn thiện mọi hàm có
`raise NotImplementedError` và các bước TODO trong `run()`. Dùng MỘT hàm `run(cfg)` cho mọi cấu hình
(RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) phải tính bằng eval.compute_metrics của repo gốc,
để cùng định nghĩa với lúc chấm:
    sys.path.insert(0, "<thư mục chứa eval.py>");  from eval import compute_metrics
"""
from __future__ import annotations

import os
import sys
import math
import time
import json
import random
import argparse
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

import dataset
import model as mymodel
import losses

# Add parent directory to sys.path to import eval
sys.path.insert(0, str(Path(__file__).parent.parent))
from eval import compute_metrics, save_predictions


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug ...
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "../data/images"
    labels_dir: str = "../data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_optimizer(model, cfg: Config):
    """AdamW với 3 nhóm tham số (xem model.param_groups)."""
    groups = mymodel.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về ~0."""
    total_steps = int(cfg.epochs * steps_per_epoch)
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
        
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Trung bình động trọng số."""
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
        self.buffers = {}
        for name, buffer in model.named_buffers():
            self.buffers[name] = buffer.data.clone().detach()

    def update(self, model) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
        for name, buffer in model.named_buffers():
            self.buffers[name] = buffer.data.clone().detach()
            
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
        for name, buffer in model.named_buffers():
            buffer.data.copy_(self.buffers[name])


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    model.train()
    
    # Giữ backbone ở chế độ eval nếu init="frozen"
    if cfg.init == "frozen":
        mymodel.freeze_backbone(model)
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d) and m not in list(model.get_classifier().modules()):
                m.eval()
                
    total_loss = 0.0
    
    for i, (images, labels, _) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if cfg.mix:
            images, targets = losses.mix_batch(images, labels, cfg.mix_alpha, cfg.mix)
            
        with autocast(enabled=cfg.amp):
            logits = model(images)
            if cfg.mix:
                loss = losses.mixed_loss(criterion, logits, targets)
            else:
                loss = criterion(logits, labels)
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        if ema is not None:
            ema.update(model)
            
        total_loss += loss.item() * images.size(0)
        
    return {"train_loss": total_loss / len(loader.dataset), "lr": scheduler.get_last_lr()[-1]}


def evaluate(model, loader, criterion, device):
    """Chạy model trên một loader ở chế độ eval, KHÔNG tính gradient."""
    model.eval()
    
    all_filenames = []
    all_targets = []
    all_logits = []
    total_loss = 0.0
    
    with torch.inference_mode():
        for images, labels, filenames in loader:
            images, labels = images.to(device), labels.to(device)
            
            with autocast(enabled=True):
                logits = model(images)
                loss = criterion(logits, labels)
                
            total_loss += loss.item() * images.size(0)
            
            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
            all_filenames.extend(filenames)
            
    all_logits = np.concatenate(all_logits, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    avg_loss = total_loss / len(loader.dataset)
    
    return all_filenames, all_targets, all_logits, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training."""
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_f1 = [h['val_f1'] for h in history]
    lr = [h['lr'] for h in history]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs, train_loss, color=color, linestyle='-', label='Train Loss')
    ax1.plot(epochs, val_loss, color=color, linestyle='--', label='Val Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Macro F1', color=color)
    ax2.plot(epochs, val_f1, color=color, linestyle='-', label='Val Macro F1')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(title)
    fig.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết."""
    # 1. Setup
    set_seed(cfg.seed)
    r_dir = run_dir(cfg)
    r_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    
    with open(r_dir / "config.json", "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)
        
    # 2. Dataset
    train_df, val_df, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
    dataset.check_split(train_df, val_df, test_df, cfg.images_dir)
    
    # 3. Loaders
    train_tf = dataset.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = dataset.build_transforms(train=False, img_size=cfg.img_size)
    
    train_loader = dataset.make_loader(train_df, cfg.images_dir, train_tf, cfg.batch_size, 
                                       train=True, sampler=cfg.sampler, num_workers=cfg.num_workers)
    val_loader = dataset.make_loader(val_df, cfg.images_dir, val_tf, cfg.batch_size, 
                                     train=False, num_workers=cfg.num_workers)
    
    if cfg.save_test_predictions:
        test_loader = dataset.make_loader(test_df, cfg.images_dir, val_tf, cfg.batch_size, 
                                          train=False, num_workers=cfg.num_workers)
                                          
    # 4. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mymodel.build_model(cfg.backbone, pretrained=True, num_classes=9, 
                                drop_rate=cfg.drop_rate, init=cfg.init).to(device)
                                
    # Criterion
    if cfg.loss == "ce_weighted":
        counts = train_df['Label'].value_counts().sort_index().values
        weight = losses.class_weights(counts, beta=cfg.class_weight_beta if cfg.class_weight_beta else 0.0)
        criterion = losses.build_criterion("ce_weighted", weight=weight).to(device)
    elif cfg.loss == "ls":
        criterion = losses.build_criterion("ls", smoothing=cfg.label_smoothing).to(device)
    elif cfg.loss == "focal":
        criterion = losses.build_criterion("focal", gamma=cfg.focal_gamma).to(device)
    else:
        criterion = losses.build_criterion("ce").to(device)
        
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = GradScaler(enabled=cfg.amp)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    
    history = []
    best_val_f1 = -1.0
    best_epoch = -1
    
    # For evaluate without mix
    val_criterion = torch.nn.CrossEntropyLoss().to(device)
    
    # 5. Train
    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema)
        
        # Eval
        eval_model = model
        if ema:
            eval_model = mymodel.build_model(cfg.backbone, pretrained=False, num_classes=9).to(device)
            eval_model.load_state_dict(model.state_dict())
            ema.apply_shadow(eval_model)
            
        filenames, y_true, logits, val_loss = evaluate(eval_model, val_loader, val_criterion, device)
        y_pred = np.argmax(logits, axis=1)
        val_probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        val_f1 = compute_metrics(y_true, y_pred, val_probs)["macro_f1"]
        
        train_time = time.time() - t0
        
        h = {
            "epoch": epoch,
            "train_loss": train_stats["train_loss"],
            "val_loss": val_loss,
            "val_f1": val_f1,
            "lr": train_stats["lr"],
            "time": train_time
        }
        history.append(h)
        print(f"Epoch {epoch}: Train Loss {h['train_loss']:.4f}, Val Loss {val_loss:.4f}, Val F1 {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(eval_model.state_dict(), r_dir / "best.pth")
            
    # 6. Load best model & eval on val
    best_model = mymodel.build_model(cfg.backbone, pretrained=False, num_classes=9).to(device)
    best_model.load_state_dict(torch.load(r_dir / "best.pth"))
    
    val_files, val_y_true, val_logits, _ = evaluate(best_model, val_loader, val_criterion, device)
    val_y_pred = np.argmax(val_logits, axis=1)
    val_probs = torch.nn.functional.softmax(torch.tensor(val_logits), dim=1).numpy()
    save_predictions(pred_path(cfg, "val"), val_files, val_y_pred, val_probs)
    
    # 7. Eval on test
    if cfg.save_test_predictions:
        test_files, _, test_logits, _ = evaluate(best_model, test_loader, val_criterion, device)
        test_y_pred = np.argmax(test_logits, axis=1)
        test_probs = torch.nn.functional.softmax(torch.tensor(test_logits), dim=1).numpy()
        save_predictions(pred_path(cfg, "test"), test_files, test_y_pred, test_probs)
        
    # 8. Post-processing
    pd.DataFrame(history).to_csv(r_dir / "history.csv", index=False)
    plot_curves(history, r_dir / "curves.png", title=f"{cfg.exp_id} - Best Val F1: {best_val_f1:.4f}")
    
    n_params = mymodel.count_params(best_model)
    n_gmacs = mymodel.count_gmacs(best_model, cfg.img_size)
    
    return {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "mean_epoch_time": np.mean([h["time"] for h in history]),
        "params": n_params,
        "gmacs": n_gmacs
    }


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict."""
    cfg_fields = {f.name: f.type for f in dataclasses.fields(Config)}
    overrides = {}
    
    for pair in pairs:
        if "=" not in pair:
            continue
        key, val = pair.split("=", 1)
        if key not in cfg_fields:
            raise ValueError(f"Unknown config key: {key}")
            
        ftype = cfg_fields[key]
        
        # Simple string to bool
        if val.lower() == "none" or val == "":
            val = None
        elif val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        else:
            # Handle unions like str | None or float | None
            # Extract actual type via simple heuristics
            if ftype == int:
                val = int(val)
            elif ftype == float:
                val = float(val)
            elif "int" in str(ftype):
                val = int(val)
            elif "float" in str(ftype):
                val = float(val)
            elif "bool" in str(ftype):
                val = (val.lower() == "true")
            elif "str" in str(ftype):
                val = str(val)
                
        overrides[key] = val
        
    return overrides


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", nargs="*", default=[], help="Override config keys: key=val")
    args = parser.parse_args()
    
    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    
    res = run(cfg)
    print(res)


if __name__ == "__main__":
    main()
