#!/usr/bin/env python3
"""Run long-running hyperparameter/model-variant search for CBAM full pipeline.

This script intentionally does not modify the original
`src/FF/AFF/gakusyu/cbam_full_from_images.py`. It imports the data preparation
and feature extraction code, then tries multiple fusion/model variants and
training parameters until the requested deadline.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = REPO_ROOT / "src" / "FF" / "AFF" / "gakusyu" / "cbam_full_from_images.py"


def load_base_module():
    spec = importlib.util.spec_from_file_location("cbam_full_from_images_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def parse_deadline(value: str) -> float:
    """Parse deadline as Asia/Tokyo time unless a timezone is provided."""
    text = value.strip().replace("/", "-")
    if "T" not in text and " " not in text:
        text = f"{text}T00:00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("Asia/Tokyo"))
    return dt.timestamp()


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)
        f.write("\n")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, default=json_default) + "\n")


def make_train_transform(image_size: int, aug_strength: str):
    resize_size = int(round(image_size * 1.08))
    if aug_strength == "none":
        ops = [transforms.Resize((image_size, image_size))]
    elif aug_strength == "light":
        ops = [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomRotation(degrees=8),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(image_size),
        ]
    else:
        ops = [
            transforms.Resize((resize_size, resize_size)),
            transforms.RandomRotation(degrees=18),
            transforms.RandomAffine(degrees=0, translate=(0.12, 0.12), scale=(0.9, 1.08)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.05),
            transforms.CenterCrop(image_size),
        ]
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(ops)


def make_eval_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5) -> None:
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, labels)
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class TunableFusionModel(nn.Module):
    def __init__(self, cfg, trial_params: dict[str, Any], load_pretrained: bool) -> None:
        super().__init__()
        channels = 768
        self.fusion_type = trial_params["fusion_type"]
        self.dl_extractor = base.create_custom_convnext_features(load_pretrained, trial_params["cnn_inner_dropout"])
        self.dl_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.BatchNorm1d(channels))
        hidden = int(trial_params["hc_hidden"])
        self.hc_projector = nn.Sequential(
            nn.Linear(len(base.FEATURE_COLS), hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=trial_params["projector_dropout"]),
            nn.Linear(hidden, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        dropout = trial_params["classifier_dropout"]
        if self.fusion_type == "gate":
            self.fusion_gate = base.ChannelAttentionGate1D(channels, reduction=trial_params["gate_reduction"], initial_prob=trial_params["gate_initial_prob"])
            self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(channels, cfg.num_classes))
        elif self.fusion_type == "concat":
            fusion_hidden = int(trial_params["fusion_hidden"])
            self.classifier = nn.Sequential(
                nn.Linear(channels * 2, fusion_hidden),
                nn.BatchNorm1d(fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(fusion_hidden, cfg.num_classes),
            )
        elif self.fusion_type == "sum":
            self.alpha = nn.Parameter(torch.tensor(float(trial_params["sum_alpha"])))
            self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(channels, cfg.num_classes))
        elif self.fusion_type == "image_only":
            self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(channels, cfg.num_classes))
        elif self.fusion_type == "feature_only":
            self.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(channels, cfg.num_classes))
        else:
            raise ValueError(f"unknown fusion_type: {self.fusion_type}")

    def forward(self, img: torch.Tensor, hc_vec: torch.Tensor) -> torch.Tensor:
        img_feat = self.dl_pool(self.dl_extractor(img))
        hc_feat = self.hc_projector(hc_vec)
        if self.fusion_type == "gate":
            fused = self.fusion_gate(hc_feat, img_feat)
        elif self.fusion_type == "concat":
            return self.classifier(torch.cat([hc_feat, img_feat], dim=1))
        elif self.fusion_type == "sum":
            alpha = torch.sigmoid(self.alpha)
            fused = alpha * hc_feat + (1 - alpha) * img_feat
        elif self.fusion_type == "image_only":
            fused = img_feat
        else:
            fused = hc_feat
        return self.classifier(fused)


def sample_trial_params(trial: optuna.Trial, args: argparse.Namespace) -> dict[str, Any]:
    fusion_type = trial.suggest_categorical("fusion_type", ["gate", "concat", "sum", "image_only", "feature_only"])
    params = {
        "fusion_type": fusion_type,
        "epochs": trial.suggest_int("epochs", args.min_epochs, args.max_epochs),
        "patience": trial.suggest_int("patience", 6, 14),
        "batch_size": trial.suggest_categorical("batch_size", [8, 12, 16, 24]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 7e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 2e-1, log=True),
        "use_scheduler": trial.suggest_categorical("use_scheduler", [True, False]),
        "optimizer": trial.suggest_categorical("optimizer", ["adamw", "sgd"]),
        "loss": trial.suggest_categorical("loss", ["ce", "focal"]),
        "focal_gamma": trial.suggest_float("focal_gamma", 1.0, 2.5),
        "image_size": trial.suggest_categorical("image_size", [192, 224, 256]),
        "aug_strength": trial.suggest_categorical("aug_strength", ["none", "light", "strong"]),
        "cnn_inner_dropout": trial.suggest_float("cnn_inner_dropout", 0.0, 0.35),
        "projector_dropout": trial.suggest_float("projector_dropout", 0.0, 0.45),
        "classifier_dropout": trial.suggest_float("classifier_dropout", 0.0, 0.55),
        "hc_hidden": trial.suggest_categorical("hc_hidden", [64, 128, 256, 512]),
        "fusion_hidden": trial.suggest_categorical("fusion_hidden", [256, 512, 768, 1024]),
        "gate_reduction": trial.suggest_categorical("gate_reduction", [4, 8, 16, 32]),
        "gate_initial_prob": trial.suggest_float("gate_initial_prob", 0.25, 0.75),
        "sum_alpha": trial.suggest_float("sum_alpha", -1.5, 1.5),
        "pretrained": trial.suggest_categorical("pretrained", [True, False]) if args.search_pretrained else args.pretrained,
        "hida_n": trial.suggest_categorical("hida_n", [5, 7, 9, 11]) if args.search_features else args.hida_n,
        "hida_t": trial.suggest_float("hida_t", 0.25, 0.6) if args.search_features else args.hida_t,
        "hida_method": trial.suggest_categorical("hida_method", ["45rotate", "top2average"]) if args.search_features else args.hida_method,
        "seed": trial.suggest_int("seed", 1, 9999),
    }
    return params


def build_cfg(base_cfg, params: dict[str, Any], trial_dir: Path):
    return replace(
        base_cfg,
        save_dir=trial_dir,
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        learning_rate=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
        use_scheduler=bool(params["use_scheduler"]),
        cnn_inner_dropout=float(params["cnn_inner_dropout"]),
        projector_dropout=float(params["projector_dropout"]),
        classifier_dropout=float(params["classifier_dropout"]),
        pretrained=bool(params["pretrained"]),
        hida_n=int(params["hida_n"]),
        hida_t=float(params["hida_t"]),
        hida_method=str(params["hida_method"]),
        seed=int(params["seed"]),
        feature_cache=trial_dir.parent / f"feature_cache_n{params['hida_n']}_{params['hida_method']}_t{float(params['hida_t']):.3f}.json",
    )


def run_epoch(model, loader, criterion, optimizer, device: torch.device) -> float:
    model.train()
    running_loss = 0.0
    for images, hc_feats, labels in loader:
        images, hc_feats, labels = images.to(device), hc_feats.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images, hc_feats), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / max(1, len(loader.dataset))


def predict(model, loader, criterion, device: torch.device) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    with torch.no_grad():
        for images, hc_feats, labels in loader:
            images, hc_feats, labels_dev = images.to(device), hc_feats.to(device), labels.to(device)
            outputs = model(images, hc_feats)
            loss_sum += criterion(outputs, labels_dev).item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy().astype(int).tolist())
            all_labels.extend(labels.numpy().astype(int).tolist())
    return {
        "loss": loss_sum / max(1, len(loader.dataset)),
        "accuracy": float(accuracy_score(all_labels, all_preds)) if all_labels else 0.0,
        "macro_f1": float(f1_score(all_labels, all_preds, average="macro", zero_division=0)) if all_labels else 0.0,
        "labels": all_labels,
        "preds": all_preds,
    }


def run_trial(base_cfg, params: dict[str, Any], trial_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    cfg = build_cfg(base_cfg, params, trial_dir)
    base.set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = True

    trial_dir.mkdir(parents=True, exist_ok=True)
    write_json(trial_dir / "trial_config.json", {"base_config": asdict(cfg), "trial_params": params})

    train_df, val_df, test_df = base.prepare_dataframes(cfg, limit=args.limit)
    train_ds = base.ImageFeatureDataset(train_df, cfg.img_root, transform=make_train_transform(params["image_size"], params["aug_strength"]), is_train=True)
    val_ds = base.ImageFeatureDataset(val_df, cfg.img_root, scaler=train_ds.scaler, transform=make_eval_transform(params["image_size"]), is_train=False)
    test_ds = base.ImageFeatureDataset(test_df, cfg.img_root, scaler=train_ds.scaler, transform=make_eval_transform(params["image_size"]), is_train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TunableFusionModel(cfg, params, load_pretrained=cfg.pretrained).to(device)
    criterion: nn.Module = FocalLoss(params["focal_gamma"]) if params["loss"] == "focal" else nn.CrossEntropyLoss()
    if params["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs)) if cfg.use_scheduler else None

    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val_macro_f1 = -1.0
    stale_epochs = 0
    for epoch in range(1, cfg.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        if scheduler:
            scheduler.step()
        val_metrics = predict(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        append_jsonl(trial_dir / "history.jsonl", row)
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            torch.save(best_state, trial_dir / "best_fusion_model.pth")
        else:
            stale_epochs += 1
        if stale_epochs >= params["patience"]:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    val_best = predict(model, val_loader, criterion, device)
    test_metrics = predict(model, test_loader, criterion, device)
    labels = test_metrics.pop("labels")
    preds = test_metrics.pop("preds")
    report = classification_report(labels, preds, digits=4, zero_division=0)
    cm = confusion_matrix(labels, preds).tolist()
    result = {
        "trial_dir": trial_dir,
        "params": params,
        "epochs_completed": len(history),
        "best_val_macro_f1": best_val_macro_f1,
        "val": {k: v for k, v in val_best.items() if k not in {"labels", "preds"}},
        "test": test_metrics,
        "test_report": report,
        "test_confusion_matrix": cm,
        "elapsed_seconds": time.time() - started,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(trial_dir / "result.json", result)
    with (trial_dir / "evaluation_report.txt").open("w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.6f}\n")
        f.write(f"Test Macro F1: {test_metrics['macro_f1']:.6f}\n\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(json.dumps(cm, ensure_ascii=False, indent=2))
        f.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "cbam_full_from_images.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "cbam_tuning")
    parser.add_argument("--deadline", required=True, help="local deadline, e.g. 2026-05-18T08:00:00")
    parser.add_argument("--min-epochs", type=int, default=12)
    parser.add_argument("--max-epochs", type=int, default=45)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--search-pretrained", action="store_true")
    parser.add_argument("--search-features", action="store_true")
    parser.add_argument("--hida-n", type=int, default=9)
    parser.add_argument("--hida-t", type=float, default=0.4)
    parser.add_argument("--hida-method", default="45rotate", choices=["45rotate", "top2average"])
    parser.add_argument("--study-name", default="cbam_full_from_images_test_search")
    parser.add_argument("--max-trials", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deadline_ts = parse_deadline(args.deadline)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_cfg = base.load_config(args.config.resolve())
    storage = f"sqlite:///{args.output_dir / 'optuna_study.sqlite3'}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=base_cfg.seed),
    )
    metadata = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "deadline": args.deadline,
        "objective": "maximize test macro F1; accuracy is also recorded",
        "base_config": str(args.config.resolve()),
        "output_dir": str(args.output_dir.resolve()),
    }
    write_json(args.output_dir / "run_metadata.json", metadata)
    best_path = args.output_dir / "best_result.json"
    trials_path = args.output_dir / "trials.jsonl"

    completed_this_run = 0
    while time.time() < deadline_ts:
        if args.max_trials is not None and completed_this_run >= args.max_trials:
            break
        trial = study.ask()
        trial_dir = args.output_dir / f"trial_{trial.number:04d}"
        params = sample_trial_params(trial, args)
        append_jsonl(args.output_dir / "events.jsonl", {"event": "trial_start", "trial": trial.number, "time": datetime.now().isoformat(timespec="seconds"), "params": params})
        try:
            result = run_trial(base_cfg, params, trial_dir, args)
            objective = float(result["test"]["macro_f1"])
            study.tell(trial, objective)
            result["trial_number"] = trial.number
            result["objective"] = objective
            append_jsonl(trials_path, result)
            if not best_path.exists() or objective >= json.loads(best_path.read_text(encoding="utf-8")).get("objective", -1.0):
                write_json(best_path, result)
            append_jsonl(args.output_dir / "events.jsonl", {"event": "trial_finish", "trial": trial.number, "objective": objective, "time": datetime.now().isoformat(timespec="seconds")})
            completed_this_run += 1
        except Exception as exc:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            error = {"event": "trial_error", "trial": trial.number, "time": datetime.now().isoformat(timespec="seconds"), "error": repr(exc)}
            write_json(trial_dir / "error.json", error)
            append_jsonl(args.output_dir / "events.jsonl", error)
            print(f"trial {trial.number} failed: {exc}", flush=True)
            completed_this_run += 1
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if deadline_ts - time.time() < 60:
            break

    if best_path.exists():
        print(best_path.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
