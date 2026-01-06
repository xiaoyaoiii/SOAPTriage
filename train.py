#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gc

# ==============================
# Utils
# ==============================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class ZScoreStat:
    mean: torch.Tensor
    std: torch.Tensor


def fit_zscore(x: torch.Tensor) -> ZScoreStat:
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return ZScoreStat(mean=mean, std=std)


def apply_zscore(x: torch.Tensor, st: ZScoreStat) -> torch.Tensor:
    return (x - st.mean) / st.std


# ==============================
# Data loading
# ==============================

def load_feature_file(path: str) -> Dict[str, Tuple[torch.Tensor, int]]:
    """
    input jsonl: {"id":..., "feature":{"user_embed":[...]}, "label":1..5}
    """
    out: Dict[str, Tuple[torch.Tensor, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = str(obj["id"])
            emb = torch.tensor(obj["feature"]["user_embed"], dtype=torch.float32)
            label = int(obj["label"]) - 1  # 1..5 -> 0..4
            out[_id] = (emb, label)
    return out


def align_streams(feature_files: List[str]) -> Tuple[List[torch.Tensor], torch.Tensor, List[str]]:
    assert len(feature_files) > 0, "feature_files is empty"
    streams = [load_feature_file(p) for p in feature_files]

    common_ids = set(streams[0].keys())
    for s in streams[1:]:
        common_ids &= set(s.keys())

    ids = sorted(common_ids)
    if len(ids) == 0:
        raise ValueError("No common ids across streams. Ensure files refer to the same sample set.")

    all_X: List[torch.Tensor] = []
    for s in streams:
        embs = [s[_id][0] for _id in ids]
        X = torch.stack(embs, dim=0)  # [N, D]
        all_X.append(X)

    y = torch.tensor([streams[0][_id][1] for _id in ids], dtype=torch.long)
    return all_X, y, ids


class MultiStreamLabelDataset(Dataset):
    def __init__(self, streams_X: List[torch.Tensor], indices: List[int], labels: torch.Tensor, ids: Optional[List[str]] = None):
        self.streams_X = streams_X
        self.indices = list(indices)
        self.labels = labels.clone().long()
        self.ids = ids
        assert len(self.indices) == len(self.labels)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = [X[idx] for X in self.streams_X]
        y = self.labels[i]
        _id = None if self.ids is None else self.ids[idx]
        return x, y, _id


def collate_fn(batch):
    num_streams = len(batch[0][0])
    streams = [[] for _ in range(num_streams)]
    labels = []
    ids = []
    for per_stream, lbl, _id in batch:
        for i, s in enumerate(per_stream):
            streams[i].append(s)
        labels.append(lbl)
        ids.append(_id)
    streams = [torch.stack(s, dim=0) for s in streams]
    labels = torch.stack(labels, dim=0).long()
    return streams, labels, ids


# ==============================
# Helper: Top-K Checkpoint Manager
# ==============================

class TopKCheckpointManager:
    """保存 Val Acc 最高的 Top-K"""
    def __init__(self, top_k=3, save_dir=".", prefix="model", ext=".pt"):
        self.top_k = top_k
        self.save_dir = save_dir
        self.prefix = prefix
        self.ext = ext
        self.checkpoints = []  # [{'acc': float, 'epoch': int, 'path': str}]

    def update(self, epoch: int, acc: float, save_dict: dict):
        filename = f"{self.prefix}_ep{epoch:03d}_acc{acc:.4f}{self.ext}"
        filepath = os.path.join(self.save_dir, filename)

        should_save = False
        if len(self.checkpoints) < self.top_k:
            should_save = True
        else:
            min_acc = self.checkpoints[-1]["acc"]
            if acc > min_acc:
                should_save = True

        if not should_save:
            return

        torch.save(save_dict, filepath)

        self.checkpoints.append({"acc": acc, "epoch": epoch, "path": filepath})
        self.checkpoints.sort(key=lambda x: (x["acc"], x["epoch"]), reverse=True)

        if len(self.checkpoints) > self.top_k:
            to_remove = self.checkpoints.pop()
            if os.path.exists(to_remove["path"]):
                try:
                    os.remove(to_remove["path"])
                    print(f"  [Manager] Removed old checkpoint: {to_remove['path']}")
                except OSError as e:
                    print(f"  [Manager] Error removing file: {e}")

        rank = self.checkpoints.index(next(x for x in self.checkpoints if x["path"] == filepath)) + 1
        print(f"  [Manager] Saved Top-{self.top_k} Model (Rank #{rank}): {filepath}")

    def get_best_path(self):
        return self.checkpoints[0]["path"] if self.checkpoints else None


# ==============================
# Normalization
# ==============================

def normalize_streams(all_X: List[torch.Tensor], norm: str):
    if norm == "none":
        return all_X, None
    out, stats = [], []
    for X in all_X:
        if norm == "l2":
            out.append(F.normalize(X, dim=1))
            stats.append(None)
        elif norm == "zscore":
            st = fit_zscore(X)
            out.append(apply_zscore(X, st))
            stats.append(st)
        else:
            raise ValueError(f"Unknown norm: {norm}")
    return out, stats


# ==============================
# Class weights & CORAL helpers
# ==============================

def maybe_compute_class_weights(
    y: torch.Tensor,
    use_class_weight: bool,
    max_class_weight: float,
) -> Optional[torch.Tensor]:
    if not use_class_weight:
        return None
    num_classes = int(y.max().item() + 1)
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / (counts + 1e-6)
    weights = torch.clamp(weights, max=max_class_weight)
    weights = weights / weights.mean()
    print("Class counts:", counts.tolist())
    print("Clamped class weights:", weights.tolist())
    return weights


def coral_targets(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    device = labels.device
    K = num_classes
    labels_expanded = labels.unsqueeze(1).repeat(1, K - 1)
    thresholds = torch.arange(K - 1, device=device).unsqueeze(0)
    targets = (labels_expanded > thresholds).float()
    return targets


def coral_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    targets = coral_targets(labels, num_classes)
    loss_per_elem = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if class_weights is not None:
        w = class_weights[labels].unsqueeze(1)
        loss_per_elem = loss_per_elem * w
    return loss_per_elem.mean()


def coral_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    passed = (probs > threshold).sum(dim=1)
    return passed.long()


# ==============================
# Gate regularization
# ==============================

def gate_regularization(weights: torch.Tensor, eps: float = 1e-8):
    """
    weights: [B, S], sum=1
    entropy: 越小越尖锐（最小化）
    balance: batch mean 越接近均匀越小（最小化）
    """
    B, S = weights.shape
    entropy = -(weights * (weights + eps).log()).sum(dim=1).mean()
    mean_w = weights.mean(dim=0)
    uniform = torch.full_like(mean_w, 1.0 / S)
    balance = F.mse_loss(mean_w, uniform)
    return entropy, balance


def gate_diversity(weights: torch.Tensor) -> torch.Tensor:
    """
    希望不同样本的 gate 分配不要都一样：std 越大越好（最大化）
    """
    return weights.std(dim=0).mean()


def stream_dropout(streams: List[torch.Tensor], p: float, training: bool):
    if (not training) or p <= 0:
        return streams
    out = []
    for s in streams:
        mask = (torch.rand(s.size(0), 1, device=s.device) > p).float()
        out.append(s * mask)
    return out


# ==============================
# Models
# ==============================

class GatingNetwork(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_streams: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_gumbel: bool = False,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * num_streams, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_streams),
        )
        self.temperature = float(temperature)
        self.use_gumbel = bool(use_gumbel)

    def forward(self, streams: List[torch.Tensor]):
        x = torch.cat(streams, dim=1)
        gate_logits = self.net(x)
        tau = max(self.temperature, 1e-6)

        if self.training and self.use_gumbel:
            weights = F.gumbel_softmax(gate_logits, tau=tau, hard=False, dim=1)
        else:
            weights = F.softmax(gate_logits / tau, dim=1)

        return weights, gate_logits


class StreamBertEncoder(nn.Module):
    """
    BERT-style Transformer encoder, but for *vector tokens*.
    We build a sequence of length L (e.g., [fused_token] + stream_tokens),
    add learnable positional embeddings, pass through TransformerEncoder,
    and pool by taking the first token.
    """
    def __init__(
        self,
        embed_dim: int,
        max_len: int,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim({embed_dim}) must be divisible by num_heads({num_heads})")

        self.embed_dim = embed_dim
        self.max_len = max_len

        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_ln = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor, pool: str = "first") -> torch.Tensor:
        """
        tokens: [B, L, D], L <= max_len
        pool: "first" or "mean"
        """
        B, L, D = tokens.shape
        assert D == self.embed_dim
        if L > self.max_len:
            raise ValueError(f"Sequence length L={L} exceeds max_len={self.max_len}")

        x = tokens + self.pos_emb[:, :L, :]
        x = self.encoder(x)
        x = self.final_ln(x)

        if pool == "first":
            return x[:, 0]
        elif pool == "mean":
            return x.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool: {pool}")


# -------- Stage 1 Binary heads --------

class BinaryLinearHead(nn.Module):
    def __init__(self, embed_dim: int, num_streams: int = 1):
        super().__init__()
        self.fc = nn.Linear(embed_dim * num_streams, 2)

    def forward(self, streams: List[torch.Tensor]):
        x = torch.cat(streams, dim=1)
        logits = self.fc(x)
        return logits, {}


class BinaryMLPHead(nn.Module):
    def __init__(self, embed_dim: int, num_streams: int = 1, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim * num_streams),
            nn.Linear(embed_dim * num_streams, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, streams: List[torch.Tensor]):
        x = torch.cat(streams, dim=1)
        logits = self.net(x)
        return logits, {}


class BinaryGatedHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_streams: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        gate_temperature: float = 0.7,
        gate_use_gumbel: bool = False,
        stream_dropout_p: float = 0.1,
    ):
        super().__init__()
        self.stream_dropout_p = float(stream_dropout_p)
        self.gate = GatingNetwork(
            embed_dim, num_streams,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=gate_temperature,
            use_gumbel=gate_use_gumbel
        )
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, streams: List[torch.Tensor]):
        streams = stream_dropout(streams, p=self.stream_dropout_p, training=self.training)
        weights, gate_logits = self.gate(streams)                  # [B,S]
        stack = torch.stack(streams, dim=1)                         # [B,S,D]
        fused = torch.sum(weights.unsqueeze(-1) * stack, dim=1)     # [B,D]
        logits = self.net(fused)
        return logits, {"weights": weights, "gate_logits": gate_logits}


class BinaryBertHead(nn.Module):
    """
    用“BERT风格 TransformerEncoder”处理多stream向量：
    - 先 gate 得到 fused token (sum w*s)
    - 再把 [fused_token] + [stream_tokens] 组成序列喂给 Transformer
    - pooled 用序列第一个 token（即 fused token 的上下文表示）
    """
    def __init__(
        self,
        embed_dim: int,
        num_streams: int,
        bert_layers: int = 2,
        bert_heads: int = 8,
        bert_ffn_dim: int = 1024,
        dropout: float = 0.1,
        gate_hidden_dim: int = 256,
        gate_temperature: float = 0.7,
        gate_use_gumbel: bool = False,
        stream_dropout_p: float = 0.1,
    ):
        super().__init__()
        self.stream_dropout_p = float(stream_dropout_p)

        self.gate = GatingNetwork(
            embed_dim, num_streams,
            hidden_dim=gate_hidden_dim,
            dropout=dropout,
            temperature=gate_temperature,
            use_gumbel=gate_use_gumbel,
        )

        # max_len = 1 (fused token) + num_streams
        self.bert = StreamBertEncoder(
            embed_dim=embed_dim,
            max_len=num_streams + 1,
            num_layers=bert_layers,
            num_heads=bert_heads,
            ffn_dim=bert_ffn_dim,
            dropout=dropout,
        )
        self.cls = nn.Linear(embed_dim, 2)

    def forward(self, streams: List[torch.Tensor]):
        streams = stream_dropout(streams, p=self.stream_dropout_p, training=self.training)
        weights, gate_logits = self.gate(streams)                  # [B,S]
        stack = torch.stack(streams, dim=1)                        # [B,S,D]
        fused = torch.sum(weights.unsqueeze(-1) * stack, dim=1)    # [B,D]

        seq = torch.cat([fused.unsqueeze(1), stack], dim=1)        # [B,1+S,D]
        pooled = self.bert(seq, pool="first")                      # [B,D]
        logits = self.cls(pooled)                                  # [B,2]
        return logits, {"weights": weights, "gate_logits": gate_logits}


def build_binary_model(
    model_type: str,
    embed_dim: int,
    num_streams: int,
    hidden_dim: int,
    dropout: float,
    gate_temperature: float,
    gate_use_gumbel: bool,
    stream_dropout_p: float,
    # bert params:
    bert_layers: int = 2,
    bert_heads: int = 8,
    bert_ffn_dim: int = 1024,
):
    if model_type == "linear":
        return BinaryLinearHead(embed_dim, num_streams)
    elif model_type == "mlp":
        return BinaryMLPHead(embed_dim, num_streams, hidden_dim, dropout)
    elif model_type == "gated":
        return BinaryGatedHead(
            embed_dim, num_streams, hidden_dim, dropout,
            gate_temperature=gate_temperature,
            gate_use_gumbel=gate_use_gumbel,
            stream_dropout_p=stream_dropout_p
        )
    elif model_type == "bert":
        return BinaryBertHead(
            embed_dim=embed_dim,
            num_streams=num_streams,
            bert_layers=bert_layers,
            bert_heads=bert_heads,
            bert_ffn_dim=bert_ffn_dim,
            dropout=dropout,
            gate_hidden_dim=hidden_dim,
            gate_temperature=gate_temperature,
            gate_use_gumbel=gate_use_gumbel,
            stream_dropout_p=stream_dropout_p,
        )
    else:
        raise ValueError(f"Unknown model_type for stage1: {model_type}")


# -------- Stage 2 CORAL heads --------

class CORALLinearHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, num_streams: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(embed_dim * num_streams, self.num_thresholds)

    def forward(self, streams: List[torch.Tensor]):
        x = torch.cat(streams, dim=1)
        logits = self.fc(x)
        return logits, {}


class CORALMLPHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, num_streams: int = 1, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim * num_streams),
            nn.Linear(embed_dim * num_streams, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_thresholds),
        )

    def forward(self, streams: List[torch.Tensor]):
        x = torch.cat(streams, dim=1)
        logits = self.net(x)
        return logits, {}


class CORALGatedHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_streams: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        gate_temperature: float = 0.7,
        gate_use_gumbel: bool = False,
        stream_dropout_p: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.stream_dropout_p = float(stream_dropout_p)

        self.gate = GatingNetwork(
            embed_dim, num_streams,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=gate_temperature,
            use_gumbel=gate_use_gumbel
        )
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_thresholds),
        )

    def forward(self, streams: List[torch.Tensor]):
        streams = stream_dropout(streams, p=self.stream_dropout_p, training=self.training)
        weights, gate_logits = self.gate(streams)
        stack = torch.stack(streams, dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * stack, dim=1)
        logits = self.net(fused)
        return logits, {"weights": weights, "gate_logits": gate_logits}


class CORALBertHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_streams: int,
        bert_layers: int = 2,
        bert_heads: int = 8,
        bert_ffn_dim: int = 1024,
        dropout: float = 0.1,
        gate_hidden_dim: int = 256,
        gate_temperature: float = 0.7,
        gate_use_gumbel: bool = False,
        stream_dropout_p: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.stream_dropout_p = float(stream_dropout_p)

        self.gate = GatingNetwork(
            embed_dim, num_streams,
            hidden_dim=gate_hidden_dim,
            dropout=dropout,
            temperature=gate_temperature,
            use_gumbel=gate_use_gumbel,
        )

        self.bert = StreamBertEncoder(
            embed_dim=embed_dim,
            max_len=num_streams + 1,
            num_layers=bert_layers,
            num_heads=bert_heads,
            ffn_dim=bert_ffn_dim,
            dropout=dropout,
        )
        self.out = nn.Linear(embed_dim, self.num_thresholds)

    def forward(self, streams: List[torch.Tensor]):
        streams = stream_dropout(streams, p=self.stream_dropout_p, training=self.training)
        weights, gate_logits = self.gate(streams)
        stack = torch.stack(streams, dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * stack, dim=1)

        seq = torch.cat([fused.unsqueeze(1), stack], dim=1)   # [B,1+S,D]
        pooled = self.bert(seq, pool="first")
        logits = self.out(pooled)
        return logits, {"weights": weights, "gate_logits": gate_logits}


def build_coral_model(
    model_type: str,
    embed_dim: int,
    num_classes: int,
    num_streams: int,
    hidden_dim: int,
    dropout: float,
    gate_temperature: float,
    gate_use_gumbel: bool,
    stream_dropout_p: float,
    # bert params:
    bert_layers: int = 2,
    bert_heads: int = 8,
    bert_ffn_dim: int = 1024,
):
    if model_type == "linear":
        return CORALLinearHead(embed_dim, num_classes, num_streams)
    elif model_type == "mlp":
        return CORALMLPHead(embed_dim, num_classes, num_streams, hidden_dim, dropout)
    elif model_type == "gated":
        return CORALGatedHead(
            embed_dim, num_classes, num_streams,
            hidden_dim, dropout,
            gate_temperature=gate_temperature,
            gate_use_gumbel=gate_use_gumbel,
            stream_dropout_p=stream_dropout_p
        )
    elif model_type == "bert":
        return CORALBertHead(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_streams=num_streams,
            bert_layers=bert_layers,
            bert_heads=bert_heads,
            bert_ffn_dim=bert_ffn_dim,
            dropout=dropout,
            gate_hidden_dim=hidden_dim,
            gate_temperature=gate_temperature,
            gate_use_gumbel=gate_use_gumbel,
            stream_dropout_p=stream_dropout_p,
        )
    else:
        raise ValueError(f"Unknown model_type for stage2: {model_type}")


# ==============================
# Train/Eval loops
# ==============================

def train_one_epoch_binary(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: str,
    criterion,
    max_grad_norm: Optional[float] = None,
    gate_entropy_lambda: float = 0.0,
    gate_balance_lambda: float = 0.0,
    gate_diversity_lambda: float = 0.0,
):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    reg_ent_sum, reg_bal_sum, reg_div_sum, reg_n = 0.0, 0.0, 0.0, 0

    for streams, labels, _ids in loader:
        streams = [s.to(device) for s in streams]
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(streams)

        loss = criterion(logits, labels)

        if isinstance(aux, dict) and "weights" in aux and (gate_entropy_lambda > 0 or gate_balance_lambda > 0 or gate_diversity_lambda > 0):
            w = aux["weights"]
            ent, bal = gate_regularization(w)
            div = gate_diversity(w)  # 越大越好
            loss = loss + gate_entropy_lambda * ent + gate_balance_lambda * bal - gate_diversity_lambda * div

            reg_ent_sum += float(ent.detach().cpu().item())
            reg_bal_sum += float(bal.detach().cpu().item())
            reg_div_sum += float(div.detach().cpu().item())
            reg_n += 1

        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    avg_acc = total_correct / total

    reg_info = {}
    if reg_n > 0:
        reg_info = {
            "gate_entropy": reg_ent_sum / reg_n,
            "gate_balance": reg_bal_sum / reg_n,
            "gate_diversity": reg_div_sum / reg_n,
        }
    return avg_loss, avg_acc, reg_info


@torch.no_grad()
def eval_epoch_binary(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    criterion,
):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    gate_weights_all = []

    for streams, labels, _ids in loader:
        streams = [s.to(device) for s in streams]
        labels = labels.to(device)

        logits, aux = model(streams)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

        if isinstance(aux, dict) and "weights" in aux:
            gate_weights_all.append(aux["weights"].detach().cpu())

    gate_stats = None
    if len(gate_weights_all) > 0:
        w = torch.cat(gate_weights_all, dim=0)
        ent, bal = gate_regularization(w)
        div = gate_diversity(w)
        gate_stats = {
            "entropy": float(ent.cpu().item()),
            "balance": float(bal.cpu().item()),
            "diversity": float(div.cpu().item()),
            "mean_per_stream": w.mean(dim=0).tolist(),
            "std_per_stream": w.std(dim=0).tolist(),
            "mean_per_sample_std": float(w.std(dim=1).mean().item()),
        }

    return total_loss / total, total_correct / total, gate_stats


def train_one_epoch_coral(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    device: str,
    num_classes: int,
    class_weights: Optional[torch.Tensor],
    max_grad_norm: Optional[float] = None,
    threshold: float = 0.5,
    gate_entropy_lambda: float = 0.0,
    gate_balance_lambda: float = 0.0,
    gate_diversity_lambda: float = 0.0,
):
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    reg_ent_sum, reg_bal_sum, reg_div_sum, reg_n = 0.0, 0.0, 0.0, 0

    for streams, labels, _ids in loader:
        streams = [s.to(device) for s in streams]
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(streams)

        loss = coral_loss(logits, labels, num_classes, class_weights)

        if isinstance(aux, dict) and "weights" in aux and (gate_entropy_lambda > 0 or gate_balance_lambda > 0 or gate_diversity_lambda > 0):
            w = aux["weights"]
            ent, bal = gate_regularization(w)
            div = gate_diversity(w)
            loss = loss + gate_entropy_lambda * ent + gate_balance_lambda * bal - gate_diversity_lambda * div

            reg_ent_sum += float(ent.detach().cpu().item())
            reg_bal_sum += float(bal.detach().cpu().item())
            reg_div_sum += float(div.detach().cpu().item())
            reg_n += 1

        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = coral_predict(logits, threshold=threshold)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    avg_acc = total_correct / total

    reg_info = {}
    if reg_n > 0:
        reg_info = {
            "gate_entropy": reg_ent_sum / reg_n,
            "gate_balance": reg_bal_sum / reg_n,
            "gate_diversity": reg_div_sum / reg_n,
        }
    return avg_loss, avg_acc, reg_info


@torch.no_grad()
def eval_epoch_coral(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
    class_weights: Optional[torch.Tensor],
    threshold: float = 0.5,
):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    gate_weights_all = []

    for streams, labels, _ids in loader:
        streams = [s.to(device) for s in streams]
        labels = labels.to(device)

        logits, aux = model(streams)
        loss = coral_loss(logits, labels, num_classes, class_weights)
        preds = coral_predict(logits, threshold=threshold)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

        if isinstance(aux, dict) and "weights" in aux:
            gate_weights_all.append(aux["weights"].detach().cpu())

    gate_stats = None
    if len(gate_weights_all) > 0:
        w = torch.cat(gate_weights_all, dim=0)
        ent, bal = gate_regularization(w)
        div = gate_diversity(w)
        gate_stats = {
            "entropy": float(ent.cpu().item()),
            "balance": float(bal.cpu().item()),
            "diversity": float(div.cpu().item()),
            "mean_per_stream": w.mean(dim=0).tolist(),
            "std_per_stream": w.std(dim=0).tolist(),
            "mean_per_sample_std": float(w.std(dim=1).mean().item()),
        }

    return total_loss / total, total_correct / total, gate_stats


# ==============================
# Main (Train/Val only)
# ==============================

def main():
    parser = argparse.ArgumentParser(description="Hierarchical Gated Training (Train/Val only, Top-K by Val Acc)")

    parser.add_argument("--train_files", nargs="+", required=True, help="SOAP streams jsonl (train)")
    parser.add_argument("--save_dir", type=str, default="hier_train_val_top3")

    parser.add_argument("--model_type_stage1", type=str, default="gated", choices=["linear", "mlp", "gated", "bert"])
    parser.add_argument("--model_type_stage2_high", type=str, default="gated", choices=["linear", "mlp", "gated", "bert"])
    parser.add_argument("--model_type_stage2_low", type=str, default="gated", choices=["linear", "mlp", "gated", "bert"])

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    parser.add_argument("--norm", type=str, default="none", choices=["none", "l2", "zscore"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)

    parser.add_argument("--class_weight", action="store_true")
    parser.add_argument("--max_class_weight", type=float, default=10.0)
    parser.add_argument("--coral_threshold", type=float, default=0.5)

    parser.add_argument("--gate_use_gumbel", action="store_true")

    # NEW: BERT-style encoder params (for model_type=bert)
    parser.add_argument("--bert_layers", type=int, default=2)
    parser.add_argument("--bert_heads", type=int, default=8)
    parser.add_argument("--bert_ffn_dim", type=int, default=1024)

    # Stage1
    parser.add_argument("--s1_gate_temperature", type=float, default=0.5)
    parser.add_argument("--s1_stream_dropout_p", type=float, default=0.0)
    parser.add_argument("--s1_gate_entropy_lambda", type=float, default=0.05)
    parser.add_argument("--s1_gate_balance_lambda", type=float, default=1.0)
    parser.add_argument("--s1_gate_diversity_lambda", type=float, default=0.50)

    # Stage2-high
    parser.add_argument("--h_gate_temperature", type=float, default=0.45)
    parser.add_argument("--h_stream_dropout_p", type=float, default=0.0)
    parser.add_argument("--h_gate_entropy_lambda", type=float, default=0.06)
    parser.add_argument("--h_gate_balance_lambda", type=float, default=1.0)
    parser.add_argument("--h_gate_diversity_lambda", type=float, default=0.5)

    # Stage2-low
    parser.add_argument("--l_gate_temperature", type=float, default=0.6)
    parser.add_argument("--l_stream_dropout_p", type=float, default=0.0)
    parser.add_argument("--l_gate_entropy_lambda", type=float, default=0.03)
    parser.add_argument("--l_gate_balance_lambda", type=float, default=1.0)
    parser.add_argument("--l_gate_diversity_lambda", type=float, default=0.5)

    parser.add_argument("--top_k", type=int, default=3)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # --------------------------
    # Load Train
    # --------------------------
    print("Loading TRAIN data...")
    train_X, train_y_all, train_ids = align_streams(args.train_files)

    num_streams = len(train_X)
    embed_dim = train_X[0].size(1)
    print(f"Train samples = {len(train_y_all)} | Streams = {num_streams} | Embed dim = {embed_dim}")

    train_X_norm, stats = normalize_streams(train_X, args.norm)

    # Train / Val split
    all_indices = list(range(len(train_y_all)))
    random.shuffle(all_indices)
    val_size = max(1, len(all_indices) // 9)
    idx_val = all_indices[:val_size]
    idx_train = all_indices[val_size:]
    print(f"Train size: {len(idx_train)}, Val size: {len(idx_val)}")

    # ==========================
    # Stage 1: Binary (0-1 vs 2-4)
    # ==========================
    print("\n=== Stage 1: Binary (0-1 vs 2-4) ===")
    y_coarse_full = torch.where(train_y_all <= 1, 0, 1)
    y_coarse_train = y_coarse_full[idx_train]
    y_coarse_val = y_coarse_full[idx_val]

    ds_train_s1 = MultiStreamLabelDataset(train_X_norm, idx_train, y_coarse_train, ids=train_ids)
    ds_val_s1 = MultiStreamLabelDataset(train_X_norm, idx_val, y_coarse_val, ids=train_ids)

    dl_train_s1 = DataLoader(ds_train_s1, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.num_workers, collate_fn=collate_fn)
    dl_val_s1 = DataLoader(ds_val_s1, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, collate_fn=collate_fn)

    s1_class_w = maybe_compute_class_weights(y_coarse_train, args.class_weight, args.max_class_weight)
    # 保持你原本逻辑：S1不使用class_weight
    if s1_class_w is not None:
        s1_class_w = None

    stage1_model = build_binary_model(
        args.model_type_stage1, embed_dim, num_streams,
        args.hidden_dim, args.dropout,
        gate_temperature=args.s1_gate_temperature,
        gate_use_gumbel=args.gate_use_gumbel,
        stream_dropout_p=args.s1_stream_dropout_p,
        bert_layers=args.bert_layers,
        bert_heads=args.bert_heads,
        bert_ffn_dim=args.bert_ffn_dim,
    ).to(device)

    crit_s1 = nn.CrossEntropyLoss(weight=s1_class_w)
    opt_s1 = torch.optim.AdamW(stage1_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s1, T_max=args.epochs, eta_min=args.lr * 0.1)

    manager_s1 = TopKCheckpointManager(top_k=args.top_k, save_dir=args.save_dir, prefix="stage1_binary", ext=".pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_reg = train_one_epoch_binary(
            stage1_model, dl_train_s1, opt_s1, device, crit_s1,
            max_grad_norm=args.max_grad_norm,
            gate_entropy_lambda=args.s1_gate_entropy_lambda,
            gate_balance_lambda=args.s1_gate_balance_lambda,
            gate_diversity_lambda=args.s1_gate_diversity_lambda,
        )
        val_loss, val_acc, gate_stats = eval_epoch_binary(stage1_model, dl_val_s1, device, crit_s1)
        sch_s1.step()

        msg = (f"[S1] Ep {epoch:03d} | Tr_Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
               f"Val_Loss={val_loss:.4f} Acc={val_acc:.4f}")
        if tr_reg:
            msg += (f" | Reg(ent={tr_reg['gate_entropy']:.4f}, bal={tr_reg['gate_balance']:.4f}, "
                    f"div={tr_reg['gate_diversity']:.4f})")
        print(msg)

        if gate_stats is not None:
            print(f"     [S1 Gate] ent={gate_stats['entropy']:.4f} bal={gate_stats['balance']:.4f} div={gate_stats['diversity']:.4f} "
                  f"mean={['%.3f'%x for x in gate_stats['mean_per_stream']]} "
                  f"std={['%.3f'%x for x in gate_stats['std_per_stream']]} "
                  f"per-sample-std={gate_stats['mean_per_sample_std']:.4f}")

        save_dict = {
            "model_state": stage1_model.state_dict(),
            "model_type": args.model_type_stage1,
            "embed_dim": embed_dim,
            "num_streams": num_streams,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "norm": args.norm,
            "stats": stats,
            "epoch": epoch,
            "val_acc": float(val_acc),
            "gate_temperature": args.s1_gate_temperature,
            "stream_dropout_p": args.s1_stream_dropout_p,
            "gate_use_gumbel": args.gate_use_gumbel,
            "bert_layers": args.bert_layers,
            "bert_heads": args.bert_heads,
            "bert_ffn_dim": args.bert_ffn_dim,
        }
        manager_s1.update(epoch, float(val_acc), save_dict)

    print(f"[S1] Best checkpoint: {manager_s1.get_best_path()}")
    stage1_model.to("cpu")
    del stage1_model, opt_s1, sch_s1, dl_train_s1, dl_val_s1, ds_train_s1, ds_val_s1
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # ==========================
    # Stage 2 High: CORAL on {0,1}
    # ==========================
    print("\n=== Stage 2 HIGH: CORAL on {0,1} ===")
    idx_train_high = [i for i in idx_train if train_y_all[i] <= 1]
    idx_val_high = [i for i in idx_val if train_y_all[i] <= 1]

    y_high_train = train_y_all[idx_train_high]  # 0/1
    y_high_val = train_y_all[idx_val_high]
    num_classes_high = 2

    ds_train_h = MultiStreamLabelDataset(train_X_norm, idx_train_high, y_high_train, ids=train_ids)
    ds_val_h = MultiStreamLabelDataset(train_X_norm, idx_val_high, y_high_val, ids=train_ids)

    dl_train_h = DataLoader(ds_train_h, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    dl_val_h = DataLoader(ds_val_h, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_fn)

    cw_high = maybe_compute_class_weights(y_high_train, args.class_weight, args.max_class_weight)
    # 保持你原本逻辑：S2-high不使用class_weight
    if cw_high is not None:
        cw_high = None

    stage2_high_model = build_coral_model(
        args.model_type_stage2_high, embed_dim, num_classes_high, num_streams,
        args.hidden_dim, args.dropout,
        gate_temperature=args.h_gate_temperature,
        gate_use_gumbel=args.gate_use_gumbel,
        stream_dropout_p=args.h_stream_dropout_p,
        bert_layers=args.bert_layers,
        bert_heads=args.bert_heads,
        bert_ffn_dim=args.bert_ffn_dim,
    ).to(device)

    opt_h = torch.optim.AdamW(stage2_high_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch_h = torch.optim.lr_scheduler.CosineAnnealingLR(opt_h, T_max=args.epochs, eta_min=args.lr * 0.1)
    manager_h = TopKCheckpointManager(top_k=args.top_k, save_dir=args.save_dir, prefix="stage2_high", ext=".pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_reg = train_one_epoch_coral(
            stage2_high_model, dl_train_h, opt_h, device,
            num_classes_high, cw_high,
            max_grad_norm=args.max_grad_norm,
            threshold=args.coral_threshold,
            gate_entropy_lambda=args.h_gate_entropy_lambda,
            gate_balance_lambda=args.h_gate_balance_lambda,
            gate_diversity_lambda=args.h_gate_diversity_lambda,
        )
        val_loss, val_acc, gate_stats = eval_epoch_coral(
            stage2_high_model, dl_val_h, device,
            num_classes_high, cw_high, threshold=args.coral_threshold
        )
        sch_h.step()

        msg = (f"[S2-H] Ep {epoch:03d} | Tr_Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
               f"Val_Loss={val_loss:.4f} Acc={val_acc:.4f}")
        if tr_reg:
            msg += (f" | Reg(ent={tr_reg['gate_entropy']:.4f}, bal={tr_reg['gate_balance']:.4f}, "
                    f"div={tr_reg['gate_diversity']:.4f})")
        print(msg)

        if gate_stats is not None:
            print(f"     [S2-H Gate] ent={gate_stats['entropy']:.4f} bal={gate_stats['balance']:.4f} div={gate_stats['diversity']:.4f} "
                  f"mean={['%.3f'%x for x in gate_stats['mean_per_stream']]} "
                  f"std={['%.3f'%x for x in gate_stats['std_per_stream']]} "
                  f"per-sample-std={gate_stats['mean_per_sample_std']:.4f}")

        save_dict = {
            "model_state": stage2_high_model.state_dict(),
            "model_type": args.model_type_stage2_high,
            "embed_dim": embed_dim,
            "num_streams": num_streams,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "norm": args.norm,
            "stats": stats,
            "num_classes": num_classes_high,
            "epoch": epoch,
            "val_acc": float(val_acc),
            "gate_temperature": args.h_gate_temperature,
            "stream_dropout_p": args.h_stream_dropout_p,
            "gate_use_gumbel": args.gate_use_gumbel,
            "bert_layers": args.bert_layers,
            "bert_heads": args.bert_heads,
            "bert_ffn_dim": args.bert_ffn_dim,
        }
        manager_h.update(epoch, float(val_acc), save_dict)

    print(f"[S2-H] Best checkpoint: {manager_h.get_best_path()}")
    stage2_high_model.to("cpu")
    del stage2_high_model, opt_h, sch_h, dl_train_h, dl_val_h, ds_train_h, ds_val_h
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # ==========================
    # Stage 2 Low: CORAL on {2,3,4} -> remap to {0,1,2}
    # ==========================
    print("\n=== Stage 2 LOW: CORAL on {2,3,4} ===")
    idx_train_low = [i for i in idx_train if train_y_all[i] >= 2]
    idx_val_low = [i for i in idx_val if train_y_all[i] >= 2]

    y_low_train = train_y_all[idx_train_low] - 2
    y_low_val = train_y_all[idx_val_low] - 2
    num_classes_low = 3

    ds_train_l = MultiStreamLabelDataset(train_X_norm, idx_train_low, y_low_train, ids=train_ids)
    ds_val_l = MultiStreamLabelDataset(train_X_norm, idx_val_low, y_low_val, ids=train_ids)

    dl_train_l = DataLoader(ds_train_l, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    dl_val_l = DataLoader(ds_val_l, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_fn)

    cw_low = maybe_compute_class_weights(y_low_train, args.class_weight, args.max_class_weight)
    if cw_low is not None:
        cw_low = cw_low.to(device)

    stage2_low_model = build_coral_model(
        args.model_type_stage2_low, embed_dim, num_classes_low, num_streams,
        args.hidden_dim, args.dropout,
        gate_temperature=args.l_gate_temperature,
        gate_use_gumbel=args.gate_use_gumbel,
        stream_dropout_p=args.l_stream_dropout_p,
        bert_layers=args.bert_layers,
        bert_heads=args.bert_heads,
        bert_ffn_dim=args.bert_ffn_dim,
    ).to(device)

    opt_l = torch.optim.AdamW(stage2_low_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch_l = torch.optim.lr_scheduler.CosineAnnealingLR(opt_l, T_max=args.epochs, eta_min=args.lr * 0.1)
    manager_l = TopKCheckpointManager(top_k=args.top_k, save_dir=args.save_dir, prefix="stage2_low", ext=".pt")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_reg = train_one_epoch_coral(
            stage2_low_model, dl_train_l, opt_l, device,
            num_classes_low, cw_low,
            max_grad_norm=args.max_grad_norm,
            threshold=args.coral_threshold,
            gate_entropy_lambda=args.l_gate_entropy_lambda,
            gate_balance_lambda=args.l_gate_balance_lambda,
            gate_diversity_lambda=args.l_gate_diversity_lambda,
        )
        val_loss, val_acc, gate_stats = eval_epoch_coral(
            stage2_low_model, dl_val_l, device,
            num_classes_low, cw_low, threshold=args.coral_threshold
        )
        sch_l.step()

        msg = (f"[S2-L] Ep {epoch:03d} | Tr_Loss={tr_loss:.4f} Acc={tr_acc:.4f} | "
               f"Val_Loss={val_loss:.4f} Acc={val_acc:.4f}")
        if tr_reg:
            msg += (f" | Reg(ent={tr_reg['gate_entropy']:.4f}, bal={tr_reg['gate_balance']:.4f}, "
                    f"div={tr_reg['gate_diversity']:.4f})")
        print(msg)

        if gate_stats is not None:
            print(f"     [S2-L Gate] ent={gate_stats['entropy']:.4f} bal={gate_stats['balance']:.4f} div={gate_stats['diversity']:.4f} "
                  f"mean={['%.3f'%x for x in gate_stats['mean_per_stream']]} "
                  f"std={['%.3f'%x for x in gate_stats['std_per_stream']]} "
                  f"per-sample-std={gate_stats['mean_per_sample_std']:.4f}")

        save_dict = {
            "model_state": stage2_low_model.state_dict(),
            "model_type": args.model_type_stage2_low,
            "embed_dim": embed_dim,
            "num_streams": num_streams,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "norm": args.norm,
            "stats": stats,
            "num_classes": num_classes_low,
            "epoch": epoch,
            "val_acc": float(val_acc),
            "gate_temperature": args.l_gate_temperature,
            "stream_dropout_p": args.l_stream_dropout_p,
            "gate_use_gumbel": args.gate_use_gumbel,
            "bert_layers": args.bert_layers,
            "bert_heads": args.bert_heads,
            "bert_ffn_dim": args.bert_ffn_dim,
        }
        manager_l.update(epoch, float(val_acc), save_dict)

    print(f"[S2-L] Best checkpoint: {manager_l.get_best_path()}")

    print("\nTraining finished. Best checkpoints are based on validation accuracy (Top-K for each stage).")


if __name__ == "__main__":
    main()
