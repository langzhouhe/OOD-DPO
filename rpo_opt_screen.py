#!/usr/bin/env python3
"""Validation-only screen for a strongly optimized, matched RPO/BCE comparison.

This intentionally has no test-evaluation code path.  It restores the paper's original
scalar MLP head (d->256->128->1) and replaces the 120-update full-batch approximation used
by revision_run.py with balanced minibatches and the complete within-batch pairwise
U-statistic.  RPO and BCE receive the same feature transform, molecule-presentation budget,
optimizer, batch indices, checkpoint count, EMA, and 32-point Sobol recipe set.

The output is a screening artifact only.  Recipes must be selected from validation results,
frozen, and then rerun with new selection/final seeds before any test scores are computed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import qmc
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

import revision_run as RR


torch.set_num_threads(1)

CACHE = Path("cache/ood_dpo_cache")
SPLITS = ("train_id", "train_ood", "val_id", "val_ood")
SCREEN_BUDGET = 256_000  # total ID+OOD molecule presentations per run
N_CHECKPOINTS = 20
EMA_DECAY = 0.995
SOBOL_SEED = 20260811


class OriginalHead(nn.Module):
    """The paper head, without revision_run.py's extra d->768 projection."""

    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def stable_hash(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def recipes() -> list[dict]:
    """A fixed 32-point shared search space; neither objective gets extra trials."""
    u = qmc.Sobol(d=6, scramble=True, seed=SOBOL_SEED).random_base2(m=5)
    batches = (64, 128, 256, 512)
    wds = (0.0, 1e-6, 1e-5, 1e-4)
    drops = (0.0, 0.1, 0.2)
    out = []
    for k, z in enumerate(u):
        # Log-uniform lr in [3e-5, 3e-3].
        lr = 10 ** (math.log10(3e-5) + float(z[1]) * 2.0)
        # Reserve one eighth of the recipes for gamma=0; otherwise log U[1e-5, 1].
        gamma = 0.0 if z[2] < 0.125 else 10 ** (-5.0 + float(z[2]) * 5.0)
        out.append(
            {
                "id": k,
                "batch": batches[min(int(z[0] * len(batches)), len(batches) - 1)],
                "lr": lr,
                "gamma": gamma,
                "weight_decay": wds[min(int(z[3] * len(wds)), len(wds) - 1)],
                "dropout": drops[min(int(z[4] * len(drops)), len(drops) - 1)],
                "normalization": "zscore" if z[5] < 0.5 else "lw_whiten",
            }
        )
    return out


def load_validation(cell: str, backbone: str) -> tuple[dict[str, np.ndarray], dict]:
    """Load train/validation only; target test scores cannot be evaluated by this script."""
    base = RR.BASE[cell]
    feat = pickle.load(open(CACHE / f"{base}_{backbone}_features.pkl", "rb"))["features"]
    split = json.load(open(CACHE / f"{base}_seed42_splits.json"))["splits"]
    split, audit = RR.domain_disjoint_split(cell, split)
    data = {}
    for key in SPLITS:
        smiles = [s for s in split[key] if s in feat]
        data[key] = np.stack([feat[s] for s in smiles]).astype(np.float32)
    return data, audit


def transformed(data: dict[str, np.ndarray], kind: str) -> dict[str, np.ndarray]:
    x = data["train_id"].astype(np.float64)
    mu = x.mean(0)
    centered = x - mu
    if kind == "zscore":
        scale = x.std(0) + 1e-6
        return {k: ((v - mu) / scale).astype(np.float32) for k, v in data.items()}
    if kind != "lw_whiten":
        raise ValueError(kind)
    cov = LedoitWolf(assume_centered=True).fit(centered).covariance_
    eig, vec = np.linalg.eigh(cov)
    invsqrt = (vec * (1.0 / np.sqrt(np.maximum(eig, 1e-8)))) @ vec.T
    return {k: ((v - mu) @ invsqrt).astype(np.float32) for k, v in data.items()}


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def update_ema(ema: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    for k, v in model.state_dict().items():
        if torch.is_floating_point(v):
            ema[k].mul_(decay).add_(v, alpha=1.0 - decay)
        else:
            ema[k].copy_(v)


@torch.no_grad()
def val_auc(model: nn.Module, vid: torch.Tensor, vood: torch.Tensor) -> float:
    model.eval()
    si = model(vid).float().cpu().numpy()
    so = model(vood).float().cpu().numpy()
    return float(roc_auc_score(np.r_[np.zeros(len(si)), np.ones(len(so))], np.r_[si, so]))


def train_one(
    arrays: dict[str, np.ndarray],
    objective: str,
    recipe: dict,
    seed: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Same selected molecules for RPO/BCE at a given seed.
    rng = np.random.default_rng(seed)
    iid = rng.permutation(len(arrays["train_id"]))[: min(1500, len(arrays["train_id"]))]
    iod = rng.permutation(len(arrays["train_ood"]))[: min(2000, len(arrays["train_ood"]))]
    xid = torch.as_tensor(arrays["train_id"][iid], device=device)
    xood = torch.as_tensor(arrays["train_ood"][iod], device=device)
    vid = torch.as_tensor(arrays["val_id"], device=device)
    vood = torch.as_tensor(arrays["val_ood"], device=device)

    model = OriginalHead(xid.shape[1], float(recipe["dropout"])).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"])
    )
    batch = int(recipe["batch"])
    steps = max(N_CHECKPOINTS, SCREEN_BUDGET // (2 * batch))
    warmup = max(1, int(0.05 * steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        p = (step - warmup) / max(1, steps - warmup - 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    ema = clone_state(model)
    best_auc, best_step, best_state = -1.0, -1, None
    curve = []
    eval_every = max(1, steps // N_CHECKPOINTS)
    clipped = 0
    t0 = time.time()
    index_gen = torch.Generator(device=device.type).manual_seed(seed + 1729)

    for step in range(steps):
        model.train()
        ii = torch.randint(len(xid), (batch,), generator=index_gen, device=device)
        io = torch.randint(len(xood), (batch,), generator=index_gen, device=device)
        ei, eo = model(xid[ii]), model(xood[io])
        if objective == "rpo":
            # Complete within-batch U-statistic, beta fixed to one.
            core = F.softplus(-(eo[:, None] - ei[None, :])).mean()
        elif objective == "bce":
            core = 0.5 * F.binary_cross_entropy_with_logits(ei, torch.zeros_like(ei))
            core = core + 0.5 * F.binary_cross_entropy_with_logits(eo, torch.ones_like(eo))
        else:
            raise ValueError(objective)
        reg = float(recipe["gamma"]) * (ei.square().mean() + eo.square().mean())
        loss = core + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped += int(float(grad) > 1.0)
        opt.step()
        sched.step()
        update_ema(ema, model, EMA_DECAY)

        if (step + 1) % eval_every == 0 or step + 1 == steps:
            raw = clone_state(model)
            model.load_state_dict(ema)
            auc = val_auc(model, vid, vood)
            curve.append({"step": step + 1, "val_auroc": auc})
            if auc > best_auc:
                best_auc, best_step, best_state = auc, step + 1, clone_state(model)
            model.load_state_dict(raw)

    return {
        "val_auroc": best_auc,
        "best_step": best_step,
        "steps": steps,
        "clip_rate": clipped / steps,
        "seconds": time.time() - t0,
        "curve": curve,
        # Deliberately no test metric and no test scorer output.
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    ap.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    ap.add_argument("--objective", choices=("rpo", "bce", "both"), default="both")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    raw, audit = load_validation(args.cell, args.backbone)
    by_norm = {k: transformed(raw, k) for k in ("zscore", "lw_whiten")}
    grid = recipes()
    objectives = ("rpo", "bce") if args.objective == "both" else (args.objective,)
    out = {
        "protocol": "rpo-opt-screen-v1-validation-only",
        "cell": args.cell,
        "backbone": args.backbone,
        "seed": args.seed,
        "domain_split": audit,
        "screen_budget": SCREEN_BUDGET,
        "recipe_hash": stable_hash(grid),
        "recipes": grid,
        "results": {},
    }
    for objective in objectives:
        out["results"][objective] = {}
        for rec in grid:
            arr = by_norm[rec["normalization"]]
            result = train_one(arr, objective, rec, args.seed, device)
            out["results"][objective][str(rec["id"])] = result
            print(
                f"[{args.cell}/{args.backbone}] {objective:3s} r={rec['id']:02d} "
                f"val={result['val_auroc']:.4f} step={result['best_step']:4d}/"
                f"{result['steps']:4d} clip={result['clip_rate']:.3f} "
                f"norm={rec['normalization']} B={rec['batch']}",
                flush=True,
            )
    Path("repro").mkdir(exist_ok=True)
    target = Path("repro") / f"rpo_opt_screen_{args.cell}_{args.backbone}_s{args.seed}.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"SCREEN_DONE {target} hash={stable_hash(out)}", flush=True)


if __name__ == "__main__":
    main()
