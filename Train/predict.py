import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error


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


def extract_stream_name(path: str) -> str:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    low = stem.lower()
    if "soap_" in low:
        idx = low.rfind("soap_")
        return stem[idx:]
    return stem


def weights_to_named_dict(weights: Optional[List[float]], names: List[str]) -> Optional[Dict[str, float]]:
    if weights is None:
        return None
    if len(weights) != len(names):
        return None
    return {names[i]: float(weights[i]) for i in range(len(names))}


def stream_dropout(streams: List[torch.Tensor], p: float, training: bool):
    if (not training) or p <= 0:
        return streams
    out = []
    for s in streams:
        mask = (torch.rand(s.size(0), 1, device=s.device) > p).float()
        out.append(s * mask)
    return out


def load_feature_file(path: str) -> Dict[str, Tuple[torch.Tensor, Optional[int]]]:
    out: Dict[str, Tuple[torch.Tensor, Optional[int]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            _id = str(obj["id"])
            emb = torch.tensor(obj["feature"]["user_embed"], dtype=torch.float32)
            if "label" in obj and obj["label"] is not None:
                label = int(obj["label"]) - 1
            else:
                label = None
            out[_id] = (emb, label)
    return out


def align_streams(feature_files: List[str]) -> Tuple[List[torch.Tensor], Optional[torch.Tensor], List[str]]:
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
        X = torch.stack(embs, dim=0)
        all_X.append(X)

    labels = [streams[0][_id][1] for _id in ids]
    if any(l is None for l in labels):
        y = None
    else:
        y = torch.tensor([int(l) for l in labels], dtype=torch.long)

    return all_X, y, ids


def apply_norm_with_stats(all_X: List[torch.Tensor], norm: str, stats):
    if norm == "none":
        return all_X
    out = []
    for i, X in enumerate(all_X):
        if norm == "l2":
            out.append(F.normalize(X, dim=1))
        elif norm == "zscore":
            out.append(apply_zscore(X, stats[i]))
        else:
            raise ValueError(f"Unknown norm: {norm}")
    return out


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
        self.temperature = float(temperature)
        self.use_gumbel = bool(use_gumbel)
        self.net = nn.Sequential(
            nn.Linear(embed_dim * num_streams, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_streams),
        )

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
        B, L, D = tokens.shape
        if D != self.embed_dim:
            raise ValueError(f"tokens dim D={D} != embed_dim={self.embed_dim}")
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


class BinaryLinearHead(nn.Module):
    def __init__(self, embed_dim: int, num_streams: int = 1):
        super().__init__()
        self.fc = nn.Linear(embed_dim * num_streams, 2)

    def forward(self, streams: List[torch.Tensor]):
        x = torch.cat(streams, dim=1)
        logits = self.fc(x)
        return logits, None


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
        return logits, None


class BinaryGatedHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_streams: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        gate_temperature: float = 0.7,
        gate_use_gumbel: bool = False,
    ):
        super().__init__()
        self.gate = GatingNetwork(
            embed_dim,
            num_streams,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=gate_temperature,
            use_gumbel=gate_use_gumbel,
        )
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, streams: List[torch.Tensor]):
        weights, gate_logits = self.gate(streams)
        stack = torch.stack(streams, dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * stack, 1)
        logits = self.net(fused)
        return logits, {"weights": weights, "gate_logits": gate_logits}


class BinaryBertHead(nn.Module):
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
        stream_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.stream_dropout_p = float(stream_dropout_p)

        self.gate = GatingNetwork(
            embed_dim,
            num_streams,
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
        self.cls = nn.Linear(embed_dim, 2)

    def forward(self, streams: List[torch.Tensor]):
        streams = stream_dropout(streams, p=self.stream_dropout_p, training=self.training)
        weights, gate_logits = self.gate(streams)
        stack = torch.stack(streams, dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * stack, dim=1)
        seq = torch.cat([fused.unsqueeze(1), stack], dim=1)
        pooled = self.bert(seq, pool="first")
        logits = self.cls(pooled)
        return logits, {"weights": weights, "gate_logits": gate_logits}


def build_binary_model(
    model_type: str,
    embed_dim: int,
    num_streams: int,
    hidden_dim: int,
    dropout: float,
    gate_temperature: float,
    gate_use_gumbel: bool,
    bert_layers: int = 2,
    bert_heads: int = 8,
    bert_ffn_dim: int = 1024,
    stream_dropout_p: float = 0.0,
):
    if model_type == "linear":
        return BinaryLinearHead(embed_dim, num_streams)
    elif model_type == "mlp":
        return BinaryMLPHead(embed_dim, num_streams, hidden_dim, dropout)
    elif model_type == "gated":
        return BinaryGatedHead(
            embed_dim,
            num_streams,
            hidden_dim=hidden_dim,
            dropout=dropout,
            gate_temperature=gate_temperature,
            gate_use_gumbel=gate_use_gumbel,
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


class CORALLinearHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, num_streams: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(embed_dim * num_streams, self.num_thresholds)

    def forward(self, streams: List[torch.Tensor]):
        x = torch.cat(streams, dim=1)
        logits = self.fc(x)
        return logits, None


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
        return logits, None


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
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.gate = GatingNetwork(
            embed_dim,
            num_streams,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=gate_temperature,
            use_gumbel=gate_use_gumbel,
        )
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_thresholds),
        )

    def forward(self, streams: List[torch.Tensor]):
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
        stream_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_thresholds = num_classes - 1
        self.stream_dropout_p = float(stream_dropout_p)

        self.gate = GatingNetwork(
            embed_dim,
            num_streams,
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
        seq = torch.cat([fused.unsqueeze(1), stack], dim=1)
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
    bert_layers: int = 2,
    bert_heads: int = 8,
    bert_ffn_dim: int = 1024,
    stream_dropout_p: float = 0.0,
):
    if model_type == "linear":
        return CORALLinearHead(embed_dim, num_classes, num_streams)
    elif model_type == "mlp":
        return CORALMLPHead(embed_dim, num_classes, num_streams, hidden_dim, dropout)
    elif model_type == "gated":
        return CORALGatedHead(
            embed_dim,
            num_classes,
            num_streams,
            hidden_dim=hidden_dim,
            dropout=dropout,
            gate_temperature=gate_temperature,
            gate_use_gumbel=gate_use_gumbel,
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


def coral_predict(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    passed = (probs > threshold).sum(dim=1)
    return passed.long()


class MultiStreamPredictDataset(Dataset):
    def __init__(self, streams_X: List[torch.Tensor], ids: List[str], y: Optional[torch.Tensor] = None):
        self.streams_X = streams_X
        self.ids = ids
        self.y = y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        streams = [X[i] for X in self.streams_X]
        _id = self.ids[i]
        lbl = None if self.y is None else int(self.y[i].item())
        return streams, lbl, _id


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
    return streams, labels, ids


def _extract_weights(aux_or_w: Any) -> Optional[torch.Tensor]:
    if aux_or_w is None:
        return None
    if torch.is_tensor(aux_or_w):
        return aux_or_w
    if isinstance(aux_or_w, dict) and "weights" in aux_or_w:
        return aux_or_w["weights"]
    return None


def main():
    parser = argparse.ArgumentParser(description="Hierarchical Prediction (supports bert, keep outputs unchanged)")

    parser.add_argument("--test_files", nargs="+", required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_high_ckpt", type=str, required=True)
    parser.add_argument("--stage2_low_ckpt", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--coral_threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--save_named_weights", action="store_true")
    parser.add_argument("--disable_metrics", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    stream_names = [extract_stream_name(p) for p in args.test_files]
    print("Stream names:", stream_names)

    test_X, test_y, ids = align_streams(args.test_files)
    num_streams = len(test_X)
    embed_dim = test_X[0].size(1)
    print(f"Loaded test N={len(ids)} | streams={num_streams} | D={embed_dim} | has_label={test_y is not None}")

    if len(stream_names) != num_streams:
        stream_names = [f"stream_{i}" for i in range(num_streams)]

    ckpt1 = torch.load(args.stage1_ckpt, map_location="cpu")
    norm = ckpt1.get("norm", "none")
    stats = ckpt1.get("stats", None)

    test_X_norm = apply_norm_with_stats(test_X, norm, stats)

    b_layers_1 = int(ckpt1.get("bert_layers", 2))
    b_heads_1 = int(ckpt1.get("bert_heads", 8))
    b_ffn_1 = int(ckpt1.get("bert_ffn_dim", 1024))
    sdrop_1 = float(ckpt1.get("stream_dropout_p", 0.0))

    stage1_model = build_binary_model(
        ckpt1["model_type"],
        ckpt1["embed_dim"],
        ckpt1["num_streams"],
        ckpt1["hidden_dim"],
        ckpt1["dropout"],
        gate_temperature=ckpt1.get("gate_temperature", 0.7),
        gate_use_gumbel=False,
        bert_layers=b_layers_1,
        bert_heads=b_heads_1,
        bert_ffn_dim=b_ffn_1,
        stream_dropout_p=sdrop_1,
    ).to(device)
    stage1_model.load_state_dict(ckpt1["model_state"], strict=True)
    stage1_model.eval()

    ckpt_h = torch.load(args.stage2_high_ckpt, map_location="cpu")
    b_layers_h = int(ckpt_h.get("bert_layers", 2))
    b_heads_h = int(ckpt_h.get("bert_heads", 8))
    b_ffn_h = int(ckpt_h.get("bert_ffn_dim", 1024))
    sdrop_h = float(ckpt_h.get("stream_dropout_p", 0.0))

    stage2_high_model = build_coral_model(
        ckpt_h["model_type"],
        ckpt_h["embed_dim"],
        ckpt_h["num_classes"],
        ckpt_h["num_streams"],
        ckpt_h["hidden_dim"],
        ckpt_h["dropout"],
        gate_temperature=ckpt_h.get("gate_temperature", 0.7),
        gate_use_gumbel=False,
        bert_layers=b_layers_h,
        bert_heads=b_heads_h,
        bert_ffn_dim=b_ffn_h,
        stream_dropout_p=sdrop_h,
    ).to(device)
    stage2_high_model.load_state_dict(ckpt_h["model_state"], strict=True)
    stage2_high_model.eval()

    ckpt_l = torch.load(args.stage2_low_ckpt, map_location="cpu")
    b_layers_l = int(ckpt_l.get("bert_layers", 2))
    b_heads_l = int(ckpt_l.get("bert_heads", 8))
    b_ffn_l = int(ckpt_l.get("bert_ffn_dim", 1024))
    sdrop_l = float(ckpt_l.get("stream_dropout_p", 0.0))

    stage2_low_model = build_coral_model(
        ckpt_l["model_type"],
        ckpt_l["embed_dim"],
        ckpt_l["num_classes"],
        ckpt_l["num_streams"],
        ckpt_l["hidden_dim"],
        ckpt_l["dropout"],
        gate_temperature=ckpt_l.get("gate_temperature", 0.7),
        gate_use_gumbel=False,
        bert_layers=b_layers_l,
        bert_heads=b_heads_l,
        bert_ffn_dim=b_ffn_l,
        stream_dropout_p=sdrop_l,
    ).to(device)
    stage2_low_model.load_state_dict(ckpt_l["model_state"], strict=True)
    stage2_low_model.eval()

    ds = MultiStreamPredictDataset(test_X_norm, ids, test_y)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    preds_all = []
    targets_all = []

    print("Predicting...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    with torch.no_grad(), open(args.output_path, "w", encoding="utf-8") as f:
        for streams, labels, batch_ids in dl:
            streams = [s.to(device) for s in streams]

            logits1, aux1 = stage1_model(streams)
            w1 = _extract_weights(aux1)
            prob1 = F.softmax(logits1, dim=1)
            coarse = logits1.argmax(dim=1)

            logits_h, aux_h = stage2_high_model(streams)
            wh = _extract_weights(aux_h)
            pred_h = coral_predict(logits_h, threshold=args.coral_threshold)

            logits_l, aux_l = stage2_low_model(streams)
            wl = _extract_weights(aux_l)
            pred_l = coral_predict(logits_l, threshold=args.coral_threshold) + 2

            final_pred = torch.where(coarse == 0, pred_h, pred_l)

            preds_all.append(final_pred.cpu())
            if labels[0] is not None:
                targets_all.append(torch.tensor([int(x) for x in labels], dtype=torch.long))

            coarse_list = coarse.detach().cpu().tolist()
            final_list = final_pred.detach().cpu().tolist()
            prob1_list = prob1.detach().cpu().tolist()

            w1_list = None if w1 is None else w1.detach().cpu().tolist()
            wh_list = None if wh is None else wh.detach().cpu().tolist()
            wl_list = None if wl is None else wl.detach().cpu().tolist()

            for i, _id in enumerate(batch_ids):
                used_branch = "high" if coarse_list[i] == 0 else "low"
                used_w = None
                if used_branch == "high":
                    used_w = None if wh_list is None else wh_list[i]
                else:
                    used_w = None if wl_list is None else wl_list[i]

                obj: Dict[str, Any] = {
                    "id": _id,
                    "pred_label_0to4": int(final_list[i]),
                    "pred_label_1to5": int(final_list[i]) + 1,
                    "stage1_coarse_pred": int(coarse_list[i]),
                    "stage1_coarse_prob": prob1_list[i],
                    "used_branch": used_branch,
                    "stream_weights_stage1": None if w1_list is None else w1_list[i],
                    "stream_weights_stage2_high": None if wh_list is None else wh_list[i],
                    "stream_weights_stage2_low": None if wl_list is None else wl_list[i],
                    "stream_weights_used_branch": used_w,
                }

                if labels[i] is not None:
                    obj["label_0to4"] = int(labels[i])
                    obj["label_1to5"] = int(labels[i]) + 1

                if args.save_named_weights:
                    obj["stream_names"] = stream_names
                    obj["named_weights_stage1"] = weights_to_named_dict(obj["stream_weights_stage1"], stream_names)
                    obj["named_weights_stage2_high"] = weights_to_named_dict(obj["stream_weights_stage2_high"], stream_names)
                    obj["named_weights_stage2_low"] = weights_to_named_dict(obj["stream_weights_stage2_low"], stream_names)
                    obj["named_weights_used_branch"] = weights_to_named_dict(obj["stream_weights_used_branch"], stream_names)

                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Saved predictions to: {args.output_path}")

    if args.disable_metrics:
        return
    if test_y is None:
        print("No label in test set -> skip metrics.")
        return

    preds_all = torch.cat(preds_all)
    targets_all = torch.cat(targets_all)

    acc = accuracy_score(targets_all.numpy(), preds_all.numpy())
    mae = mean_absolute_error(targets_all.numpy(), preds_all.numpy())
    print("\n[Metrics]")
    print(f"Accuracy: {acc:.4f}")
    print(f"MAE (0..4): {mae:.4f}")
    print("\nClassification report:")
    print(classification_report(targets_all.numpy(), preds_all.numpy(), digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(targets_all.numpy(), preds_all.numpy()))


if __name__ == "__main__":
    main()
