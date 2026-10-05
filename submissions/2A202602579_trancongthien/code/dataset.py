"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

PSEUDO-CODE: bạn tự hoàn thiện mọi hàm có `raise NotImplementedError`.
Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1. Đọc trước khi viết.

Giao diện bạn phải giữ (để notebook, train.py và eval.py ghép được với nhau):
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)
"""
from __future__ import annotations

from pathlib import Path
import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.v2 as v2

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # đổi nếu trọng số timm bạn dùng yêu cầu mean/std khác
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1)."""
    labels_dir = Path(labels_dir)
    train_df = pd.read_csv(labels_dir / f"train_subset{fold}.csv")
    val_df = pd.read_csv(labels_dir / f"val_subset{fold}.csv")
    test_df = pd.read_csv(labels_dir / f"test_subset{fold}.csv")
    return train_df, val_df, test_df


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). In ra và trả về dict số liệu."""
    images_dir = Path(images_dir)
    
    # 1. số ảnh mỗi tập và mỗi lớp
    def get_counts(df): return {"total": len(df), "per_class": df['Label'].value_counts().to_dict()}
    counts = {
        "train": get_counts(train_df),
        "val": get_counts(val_df),
        "test": get_counts(test_df)
    }
    
    # 2. giao của từng cặp RỖNG
    train_f = set(train_df['Filename'])
    val_f = set(val_df['Filename'])
    test_f = set(test_df['Filename'])
    assert len(train_f.intersection(val_f)) == 0, "Train and Val overlap!"
    assert len(train_f.intersection(test_f)) == 0, "Train and Test overlap!"
    assert len(val_f.intersection(test_f)) == 0, "Val and Test overlap!"
    
    # 3. hợp 3 tập bằng 17509
    total_imgs = len(train_f.union(val_f).union(test_f))
    assert total_imgs == 17509, f"Total images expected 17509, got {total_imgs}"
    
    # 4. Filename tồn tại
    for f in train_f.union(val_f).union(test_f):
        assert (images_dir / f).exists(), f"File {f} not found in {images_dir}"
        
    return counts


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Tạo transform. `aug` chọn mức augmentation; bạn tự định nghĩa các giá trị."""
    if train:
        if aug == "trivial":
            return v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(), # Cỏ dại có thể lật dọc
                v2.TrivialAugmentWide(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        elif aug == "randaug":
            return v2.Compose([
                v2.ToImage(),
                v2.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandAugment(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        else: # basic or color
            ts = [
                v2.ToImage(),
                v2.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
            ]
            if aug == "color":
                ts.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
            ts.extend([
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            return v2.Compose(ts)
    else:
        return v2.Compose([
            v2.ToImage(),
            v2.Resize(256),
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


class DeepWeedsDataset(Dataset):
    """Dataset đọc ảnh từ `images_dir` theo DataFrame (Filename, Label)."""
    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = row['Filename']
        label = int(row['Label'])
        
        img_path = self.images_dir / filename
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label, filename


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)

def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Tạo DataLoader."""
    dataset = DeepWeedsDataset(df, images_dir, transform)
    
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": seed_worker
    }
    
    if train:
        loader_kwargs["drop_last"] = True
        if sampler == "balanced":
            class_counts = df['Label'].value_counts().sort_index().values
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[label] for label in df['Label']]
            pt_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            loader_kwargs["sampler"] = pt_sampler
            loader_kwargs["shuffle"] = False
        else:
            loader_kwargs["shuffle"] = True
    else:
        loader_kwargs["shuffle"] = False
        loader_kwargs["drop_last"] = False
        
    return DataLoader(dataset, **loader_kwargs)
