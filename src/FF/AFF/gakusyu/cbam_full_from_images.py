#!/usr/bin/env python3
"""Complete CBAM_check pipeline that computes handcrafted features from images.

Based on the latest "CNNで使用するデータ直し" section in CBAM_check.ipynb.
The original code reads h0-h7, size_count, and R from CSV files. This version
keeps the ConvNeXt + 1D attention fusion model, but computes those 10 features
from each sample's mask and combined image.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.ops.stochastic_depth import StochasticDepth


FEATURE_COLS = ["h0", "h1", "h2", "h3", "h4", "h5", "h6", "h7", "size_count", "R"]
LABEL_MAP = {"A": 0, "B": 1, "C": 2}


@dataclass(frozen=True)
class Config:
    img_root: Path
    train_csv: Path
    val_csv: Path
    test_csv: Path
    save_dir: Path
    num_classes: int = 3
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 0.0001
    weight_decay: float = 0.05
    use_scheduler: bool = False
    cnn_inner_dropout: float = 0.2
    projector_dropout: float = 0.2
    classifier_dropout: float = 0.3
    pretrained: bool = True
    hida_n: int = 9
    hida_t: float = 0.4
    hida_method: str = "45rotate"
    num_workers: int = 0
    seed: int = 42
    feature_cache: Path | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo_root() / path


def load_config(path: Path) -> Config:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    cache = raw.get("feature_cache")
    return Config(
        img_root=resolve_path(raw["img_root"]),
        train_csv=resolve_path(raw["train_csv"]),
        val_csv=resolve_path(raw["val_csv"]),
        test_csv=resolve_path(raw["test_csv"]),
        save_dir=resolve_path(raw["save_dir"]),
        num_classes=int(raw.get("num_classes", 3)),
        epochs=int(raw.get("epochs", 100)),
        batch_size=int(raw.get("batch_size", 16)),
        learning_rate=float(raw.get("learning_rate", 0.0001)),
        weight_decay=float(raw.get("weight_decay", 0.05)),
        use_scheduler=bool(raw.get("use_scheduler", False)),
        cnn_inner_dropout=float(raw.get("cnn_inner_dropout", 0.2)),
        projector_dropout=float(raw.get("projector_dropout", 0.2)),
        classifier_dropout=float(raw.get("classifier_dropout", 0.3)),
        pretrained=bool(raw.get("pretrained", True)),
        hida_n=int(raw.get("hida_n", 9)),
        hida_t=float(raw.get("hida_t", 0.4)),
        hida_method=str(raw.get("hida_method", "45rotate")),
        num_workers=int(raw.get("num_workers", 0)),
        seed=int(raw.get("seed", 42)),
        feature_cache=resolve_path(cache) if cache else None,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def actual_combined_filename(mask_filename: str) -> str:
    return mask_filename.replace("_mask", "_combined")


def sample_paths(img_root: Path, category: str, mask_filename: str) -> tuple[Path, Path]:
    return (
        img_root / "mask" / category / mask_filename,
        img_root / "combined" / category / actual_combined_filename(mask_filename),
    )


def normalize_split_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"filename", "category"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"split CSV missing required columns: {sorted(missing)}")
    df = df.copy()
    if "Label" not in df.columns:
        df["Label"] = df["category"].map(LABEL_MAP)
    return df[["filename", "category", "Label"]].reset_index(drop=True)


class FeatureExtractor:
    """Compute h0-h7, size_count, and R from mask and combined image."""

    def __init__(self, n: int = 9, t: float = 0.4, method: str = "45rotate") -> None:
        self.n = n
        self.t = t
        self.method = method
        if method not in {"45rotate", "top2average"}:
            raise ValueError("hida_method must be '45rotate' or 'top2average'")

    def compute(self, mask_path: Path, combined_path: Path) -> np.ndarray:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"mask not found: {mask_path}")
        image = cv2.imread(str(combined_path))
        if image is None:
            raise FileNotFoundError(f"combined image not found: {combined_path}")
        return np.array([*self.hu_features(mask), self.size_count(mask), self.hida_r(image, mask)], dtype=np.float32)

    @staticmethod
    def hu_features(mask: np.ndarray) -> list[float]:
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return [0.0] * 8
        largest = max(contours, key=cv2.contourArea)
        hu = cv2.HuMoments(cv2.moments(largest)).flatten()
        signed_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
        values = signed_log.tolist()
        values.append(abs(values[-1]))
        return [float(v) for v in values]

    @staticmethod
    def _walk_until_outside(binary: np.ndarray, center: np.ndarray, direction: np.ndarray) -> np.ndarray | None:
        h, w = binary.shape[:2]
        point = center.copy()
        while 0 <= int(point[0]) < w and 0 <= int(point[1]) < h:
            if binary[int(point[1]), int(point[0])] == 0:
                return point.copy()
            point += direction
        return None

    def size_count(self, mask: np.ndarray) -> float:
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return 0.0
        center = np.array([int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])], dtype=np.float32)
        min_distance = float("inf")
        max_distance = 0.0
        for angle in range(0, 180):
            rad = np.deg2rad(angle)
            direction = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)
            positive = self._walk_until_outside(binary, center, direction)
            negative = self._walk_until_outside(binary, center, -direction)
            if positive is None or negative is None:
                continue
            distance = float(np.linalg.norm(positive - negative))
            min_distance = min(min_distance, distance)
            max_distance = max(max_distance, distance)
        if min_distance == float("inf"):
            min_distance = 0.0
        return float((min_distance + max_distance) / 2.0)

    @staticmethod
    def _g12(theta1: np.ndarray, theta2: np.ndarray) -> np.ndarray:
        condition1 = (theta2 - np.pi < theta1) & (theta1 < theta2) & (theta2 >= 0)
        result1 = np.abs(theta1 - theta2)
        condition2 = (-np.pi < theta1) & (theta1 < (theta2 - np.pi)) & (theta2 >= 0)
        result2 = theta2 - 2 * np.pi - theta1
        condition3 = (-np.pi < theta1) & (theta1 < (theta2 + np.pi)) & (theta2 < 0)
        result3 = np.abs(theta1 - theta2)
        condition4 = (theta2 + np.pi < theta1) & (theta1 < np.pi) & (theta2 < 0)
        result4 = theta1 - theta2 - 2 * np.pi
        return np.where(condition1, result1, np.where(condition2, result2, np.where(condition3, result3, np.where(condition4, result4, 0))))

    @staticmethod
    def _ratio(mask_part: np.ndarray, hida_part: np.ndarray) -> float:
        count_mask = np.sum(mask_part == 255)
        count_hida = np.sum(hida_part == 1)
        return float(count_hida / count_mask) if count_mask > 0 else 0.0

    def hida_r(self, image_bgr: np.ndarray, mask: np.ndarray) -> float:
        h, w = mask.shape[:2]
        ys, xs = np.where(mask == 255)
        xc, yc = (float(np.mean(xs)), float(np.mean(ys))) if ys.size > 0 else (w / 2, h / 2)
        xc_int, yc_int = int(round(xc)), int(round(yc))
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        fdy, fdx = np.gradient(gray.astype(float))
        f0xy = np.arctan2(fdy, fdx)
        absfxy = np.sqrt(fdx**2 + fdy**2).astype(np.uint8)
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        c0xy_dx = xx - xc
        c0xy_dy = yy - yc
        c0xy = np.zeros_like(c0xy_dx, dtype=float)
        non_zero_dx = c0xy_dx != 0
        c0xy[non_zero_dx] = np.arctan(c0xy_dy[non_zero_dx] / c0xy_dx[non_zero_dx])
        fdisxy = np.minimum(self._g12(c0xy + np.pi / 2, f0xy) ** 2, self._g12(c0xy - np.pi / 2, f0xy) ** 2)
        mxy = cv2.medianBlur(absfxy, 2 * self.n + 1)
        fmxy_val = np.where(absfxy > mxy, 1, 0)
        rdisxy = fmxy_val * fdisxy
        iruv = cv2.integral(rdisxy)
        imyv = cv2.integral(fmxy_val.astype(np.uint8))
        sdisval = np.zeros((h, w), dtype=np.float32)
        for y in range(h):
            for x in range(w):
                x1, y1 = max(0, x - self.n), max(0, y - self.n)
                x2, y2 = min(w - 1, x + self.n), min(h - 1, y + self.n)
                tr = iruv[y2 + 1, x2 + 1] - iruv[y1, x2 + 1] - iruv[y2 + 1, x1] + iruv[y1, x1]
                tm = imyv[y2 + 1, x2 + 1] - imyv[y1, x2 + 1] - imyv[y2 + 1, x1] + imyv[y1, x1]
                if tm > 0:
                    sdisval[y, x] = tr / tm
        hxy = np.where(sdisval < self.t, 1, 0).astype(np.uint8)
        hxy2 = cv2.bitwise_and(hxy, hxy, mask=mask)
        if self.method == "top2average":
            return self._hida_top2average(mask, hxy2, xc_int, yc_int)
        return self._hida_45rotate(mask, hxy2, xc, yc, xc_int, yc_int)

    def _hida_top2average(self, mask: np.ndarray, hida: np.ndarray, xc_int: int, yc_int: int) -> float:
        h, w = mask.shape[:2]
        values = [
            self._ratio(mask[0:yc_int, 0:xc_int], hida[0:yc_int, 0:xc_int]),
            self._ratio(mask[0:yc_int, xc_int:w], hida[0:yc_int, xc_int:w]),
            self._ratio(mask[yc_int:h, 0:xc_int], hida[yc_int:h, 0:xc_int]),
            self._ratio(mask[yc_int:h, xc_int:w], hida[yc_int:h, xc_int:w]),
        ]
        return float(sum(sorted(values, reverse=True)[:2]) / 2.0)

    def _hida_45rotate(self, mask: np.ndarray, hida: np.ndarray, xc: float, yc: float, xc_int: int, yc_int: int) -> float:
        h, w = mask.shape[:2]
        values = []
        for angle in [0, 45, 90, 135]:
            matrix = cv2.getRotationMatrix2D((xc, yc), angle, 1.0)
            mask_rot = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST)
            hida_rot = cv2.warpAffine(hida, matrix, (w, h), flags=cv2.INTER_NEAREST)
            values.extend(
                [
                    self._ratio(mask_rot[:, :xc_int], hida_rot[:, :xc_int]),
                    self._ratio(mask_rot[:, xc_int:], hida_rot[:, xc_int:]),
                    self._ratio(mask_rot[:yc_int, :], hida_rot[:yc_int, :]),
                    self._ratio(mask_rot[yc_int:, :], hida_rot[yc_int:, :]),
                ]
            )
        return max(values) if values else 0.0


class FeatureCache:
    def __init__(self, cache_path: Path | None, extractor: FeatureExtractor) -> None:
        self.cache_path = cache_path
        self.extractor = extractor
        self.values: dict[str, list[float]] = {}
        if cache_path and cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                self.values = json.load(f)

    @staticmethod
    def key(mask_path: Path, combined_path: Path) -> str:
        return f"{mask_path}|{combined_path}"

    def get(self, mask_path: Path, combined_path: Path) -> np.ndarray:
        key = self.key(mask_path, combined_path)
        if key not in self.values:
            self.values[key] = self.extractor.compute(mask_path, combined_path).astype(float).tolist()
        return np.array(self.values[key], dtype=np.float32)

    def save(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as f:
            json.dump(self.values, f, ensure_ascii=False, indent=2)
            f.write("\n")


def dataframe_with_image_features(df: pd.DataFrame, cfg: Config, cache: FeatureCache) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in normalize_split_df(df).itertuples(index=False):
        mask_path, combined_path = sample_paths(cfg.img_root, row.category, row.filename)
        features = cache.get(mask_path, combined_path)
        record = {"filename": row.filename, "category": row.category, "Label": int(row.Label)}
        record.update({col: float(value) for col, value in zip(FEATURE_COLS, features)})
        records.append(record)
    return pd.DataFrame(records)


class ImageFeatureDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, img_root: Path, scaler: StandardScaler | None = None, transform=None, is_train: bool = True) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform
        features = self.df[FEATURE_COLS].values
        if is_train:
            self.scaler = StandardScaler()
            self.hc_features = self.scaler.fit_transform(features).astype("float32")
        else:
            if scaler is None:
                raise ValueError("scaler is required for non-training datasets")
            self.scaler = scaler
            self.hc_features = self.scaler.transform(features).astype("float32")
        self.labels = self.df["Label"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        combined_path = self.img_root / "combined" / row["category"] / actual_combined_filename(row["filename"])
        try:
            image = Image.open(combined_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.hc_features[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class Permute(nn.Module):
    def __init__(self, dims: list[int]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.permute(x, self.dims)


class CNBlock(nn.Module):
    def __init__(self, dim: int, layer_scale: float, stochastic_depth_prob: float, dropout_p: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            OrderedDict(
                [
                    ("0", nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True)),
                    ("1", Permute([0, 2, 3, 1])),
                    ("2", nn.LayerNorm(dim, eps=1e-6)),
                    ("3", nn.Linear(dim, 4 * dim, bias=True)),
                    ("4", nn.GELU()),
                    ("custom_drop", nn.Dropout(p=dropout_p)),
                    ("5", nn.Linear(4 * dim, dim, bias=True)),
                ]
            )
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stoch_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = self.block(input).permute(0, 3, 1, 2)
        return input + self.stoch_depth(self.layer_scale * result)


def create_custom_convnext_features(pretrained: bool, dropout_p: float) -> nn.Sequential:
    block_setting = [3, 3, 9, 3]
    dims = [96, 192, 384, 768]
    features = nn.Sequential()
    features.add_module("0", nn.Sequential(nn.Conv2d(3, dims[0], kernel_size=4, stride=4), LayerNorm2d(dims[0], eps=1e-6)))
    curr_stage = 0
    dp_rates = [x.item() for x in torch.linspace(0, 0.1, sum(block_setting))]
    for i in range(4):
        if i > 0:
            features.add_module(str(2 * i), nn.Sequential(LayerNorm2d(dims[i - 1], eps=1e-6), nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2)))
        blocks = []
        for _ in range(block_setting[i]):
            blocks.append(CNBlock(dims[i], 1e-6, dp_rates[curr_stage], dropout_p))
            curr_stage += 1
        features.add_module(str(2 * i + 1), nn.Sequential(*blocks))
    if pretrained:
        print("Loading official ConvNeXt-Tiny pre-trained weights...")
        orig_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        features.load_state_dict(orig_model.features.state_dict(), strict=False)
    return features


class ChannelAttentionGate1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, initial_prob: float = 0.5) -> None:
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.attention = nn.Sequential(nn.Linear(channels, reduced_channels), nn.ReLU(inplace=True), nn.Linear(reduced_channels, channels))
        self.sigmoid = nn.Sigmoid()
        logit_value = math.log(initial_prob / (1 - initial_prob))
        with torch.no_grad():
            nn.init.constant_(self.attention[2].bias, logit_value)
            self.attention[2].weight.data *= 0.01

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        weights = self.sigmoid(self.attention(x + y))
        return weights * x + (1 - weights) * y


class FusionModel(nn.Module):
    def __init__(self, cfg: Config, load_pretrained: bool) -> None:
        super().__init__()
        channels = 768
        self.dl_extractor = create_custom_convnext_features(load_pretrained, cfg.cnn_inner_dropout)
        self.dl_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(channels))
        self.hc_projector = nn.Sequential(nn.Linear(len(FEATURE_COLS), channels), nn.BatchNorm1d(channels), nn.ReLU(inplace=True), nn.Dropout(p=cfg.projector_dropout))
        self.fusion_gate = ChannelAttentionGate1D(channels, initial_prob=0.5)
        self.classifier = nn.Sequential(nn.Dropout(p=cfg.classifier_dropout), nn.Linear(channels, cfg.num_classes))

    def forward(self, img: torch.Tensor, hc_vec: torch.Tensor) -> torch.Tensor:
        img_feat = self.dl_pool(self.dl_extractor(img))
        hc_feat = self.hc_projector(hc_vec)
        return self.classifier(self.fusion_gate(hc_feat, img_feat))


def train_transform():
    return transforms.Compose(
        [
            transforms.Resize((236, 236)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def prepare_dataframes(cfg: Config, limit: int | None = None):
    extractor = FeatureExtractor(n=cfg.hida_n, t=cfg.hida_t, method=cfg.hida_method)
    cache = FeatureCache(cfg.feature_cache, extractor)
    train_source = pd.read_csv(cfg.train_csv)
    val_source = pd.read_csv(cfg.val_csv)
    test_source = pd.read_csv(cfg.test_csv)
    if limit:
        train_source = train_source.head(limit)
        val_source = val_source.head(max(1, min(limit, len(val_source))))
        test_source = test_source.head(max(1, min(limit, len(test_source))))
    train_df = dataframe_with_image_features(train_source, cfg, cache)
    val_df = dataframe_with_image_features(val_source, cfg, cache)
    test_df = dataframe_with_image_features(test_source, cfg, cache)
    cache.save()
    return train_df, val_df, test_df


def save_learning_curve(history: dict[str, list[float]], save_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "r-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "g-", label="Val Loss")
    plt.title("Training & Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_acc"], "b-", label="Val Acc")
    plt.title("Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def train_model(cfg: Config, limit: int | None = None) -> None:
    set_seed(cfg.seed)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, _ = prepare_dataframes(cfg, limit=limit)
    train_ds = ImageFeatureDataset(train_df, cfg.img_root, transform=train_transform(), is_train=True)
    val_ds = ImageFeatureDataset(val_df, cfg.img_root, scaler=train_ds.scaler, transform=eval_transform(), is_train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    model = FusionModel(cfg, load_pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs) if cfg.use_scheduler else None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for images, hc_feats, labels in train_loader:
            images, hc_feats, labels = images.to(device), hc_feats.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images, hc_feats), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        if scheduler:
            scheduler.step()
        train_loss = running_loss / len(train_ds)
        model.eval()
        val_loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, hc_feats, labels in val_loader:
                images, hc_feats, labels = images.to(device), hc_feats.to(device), labels.to(device)
                outputs = model(images, hc_feats)
                val_loss_sum += criterion(outputs, labels).item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += torch.sum(preds == labels.data).item()
        val_loss = val_loss_sum / len(val_ds)
        val_acc = correct / total if total else 0.0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch + 1}/{cfg.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.save_dir / "best_fusion_model.pth")
            print(f"  --> Best Model Saved! (Acc: {best_acc:.4f})")
        if (epoch + 1) % 5 == 0:
            save_learning_curve(history, cfg.save_dir / "learning_curve.png")
    save_learning_curve(history, cfg.save_dir / "learning_curve.png")
    with (cfg.save_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
        f.write("\n")


def evaluate_model(cfg: Config, limit: int | None = None) -> None:
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_weight_path = cfg.save_dir / "best_fusion_model.pth"
    if not model_weight_path.exists():
        raise FileNotFoundError(f"model weight not found: {model_weight_path}")
    train_df, _, test_df = prepare_dataframes(cfg, limit=limit)
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLS].values)
    test_ds = ImageFeatureDataset(test_df, cfg.img_root, scaler=scaler, transform=eval_transform(), is_train=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    model = FusionModel(cfg, load_pretrained=False).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, hc_feats, labels in test_loader:
            outputs = model(images.to(device), hc_feats.to(device))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4)
    print(f"Overall Test Accuracy: {accuracy:.4f}\n")
    print(report)
    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.save_dir / "evaluation_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Overall Test Accuracy: {accuracy:.4f}\n\n{report}")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix (Late Fusion Model)")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(cfg.save_dir / "test_confusion_matrix.png")
    plt.close()


def export_features(cfg: Config, output_path: Path) -> None:
    train_df, val_df, test_df = prepare_dataframes(cfg)
    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"
    output = pd.concat([train_df, val_df, test_df], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(f"features exported: {output_path}")


def dry_run(cfg: Config, limit: int) -> None:
    df = normalize_split_df(pd.read_csv(cfg.train_csv)).head(limit)
    cache = FeatureCache(None, FeatureExtractor(n=cfg.hida_n, t=cfg.hida_t, method=cfg.hida_method))
    for row in df.itertuples(index=False):
        mask_path, combined_path = sample_paths(cfg.img_root, row.category, row.filename)
        features = cache.get(mask_path, combined_path)
        print(json.dumps({"filename": row.filename, "category": row.category, "label": int(row.Label), "features": dict(zip(FEATURE_COLS, map(float, features)))}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=repo_root() / "configs" / "cbam_full_from_images.json")
    parser.add_argument("--mode", choices=["train", "eval", "export-features", "dry-run"], default="dry-run")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config.resolve())
    if args.mode == "dry-run":
        dry_run(cfg, args.limit or 3)
    elif args.mode == "train":
        train_model(cfg, limit=args.limit)
    elif args.mode == "eval":
        evaluate_model(cfg, limit=args.limit)
    elif args.mode == "export-features":
        export_features(cfg, resolve_path(args.output or (cfg.save_dir / "features_from_images.csv")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
