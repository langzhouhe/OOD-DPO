#!/usr/bin/env python3
"""Frozen final-seed evaluation for RPO-Opt.

Recipes are selected deterministically from the five-seed validation artifacts produced by
rpo_opt_stage2.py.  This file was created after validation selection and before the final
seed test run.  All final seeds are reported; no test-dependent selection is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import revision_run as RR
import rpo_opt_screen as S
import rpo_opt_stage2 as P


FINAL_SEEDS = tuple(range(101, 111))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def select_recipe(stage: dict, objective: str) -> tuple[int, dict]:
    rows = []
    for rid, runs in stage["results"][objective].items():
        vals = np.array([r["val_auroc"] for r in runs], dtype=np.float64)
        # Primary mean validation AUROC; deterministic tie-break by worst seed then ID.
        rows.append((float(vals.mean()), float(vals.min()), -int(rid), int(rid)))
    best = max(rows)
    return best[3], {"mean_val": best[0], "worst_seed_val": best[1]}


def load_all(cell: str, backbone: str) -> tuple[dict[str, np.ndarray], dict]:
    base = RR.BASE[cell]
    feat = pickle.load(open(S.CACHE / f"{base}_{backbone}_features.pkl", "rb"))["features"]
    split = json.load(open(S.CACHE / f"{base}_seed42_splits.json"))["splits"]
    split, audit = RR.domain_disjoint_split(cell, split)
    keys = ("train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood")
    data = {}
    for key in keys:
        smiles = [s for s in split[key] if s in feat]
        data[key] = np.stack([feat[s] for s in smiles]).astype(np.float32)
    return data, audit


def train_final(
    arrays: dict[str, np.ndarray], objective: str, recipe: dict, seed: int, device: torch.device
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    iid = rng.permutation(len(arrays["train_id"]))[: min(1500, len(arrays["train_id"]))]
    iod = rng.permutation(len(arrays["train_ood"]))[: min(2000, len(arrays["train_ood"]))]
    xid = torch.as_tensor(arrays["train_id"][iid], device=device)
    xood = torch.as_tensor(arrays["train_ood"][iod], device=device)
    vid = torch.as_tensor(arrays["val_id"], device=device)
    vood = torch.as_tensor(arrays["val_ood"], device=device)
    tid = torch.as_tensor(arrays["test_id"], device=device)
    tood = torch.as_tensor(arrays["test_ood"], device=device)

    model = S.OriginalHead(xid.shape[1], float(recipe["dropout"])).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"])
    )
    batch = int(recipe["batch"])
    steps = max(S.N_CHECKPOINTS, P.FULL_BUDGET // (2 * batch))
    warmup = max(1, int(0.05 * steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        p = (step - warmup) / max(1, steps - warmup - 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    ema = S.clone_state(model)
    best_auc, best_step, best_state = -1.0, -1, None
    eval_every = max(1, steps // S.N_CHECKPOINTS)
    clipped = 0
    t0 = time.time()
    index_gen = torch.Generator(device=device.type).manual_seed(seed + 1729)
    for step in range(steps):
        model.train()
        ii = torch.randint(len(xid), (batch,), generator=index_gen, device=device)
        io = torch.randint(len(xood), (batch,), generator=index_gen, device=device)
        ei, eo = model(xid[ii]), model(xood[io])
        if objective == "rpo":
            core = F.softplus(-(eo[:, None] - ei[None, :])).mean()
        else:
            core = 0.5 * F.binary_cross_entropy_with_logits(ei, torch.zeros_like(ei))
            core = core + 0.5 * F.binary_cross_entropy_with_logits(eo, torch.ones_like(eo))
        loss = core + float(recipe["gamma"]) * (ei.square().mean() + eo.square().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped += int(float(grad) > 1.0)
        opt.step()
        sched.step()
        S.update_ema(ema, model, S.EMA_DECAY)
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            raw = S.clone_state(model)
            model.load_state_dict(ema)
            auc = S.val_auc(model, vid, vood)
            if auc > best_auc:
                best_auc, best_step, best_state = auc, step + 1, S.clone_state(model)
            model.load_state_dict(raw)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        sid = model(tid).float().cpu().numpy()
        sood = model(tood).float().cpu().numpy()
    auroc, aupr, fpr95 = RR.mets(sid, sood)
    return {
        "val_auroc": best_auc,
        "best_step": best_step,
        "steps": steps,
        "clip_rate": clipped / steps,
        "test_auroc": auroc,
        "test_aupr": aupr,
        "test_fpr95": fpr95,
        # Persist the final scores so paired/domain-aware uncertainty estimates and
        # any pre-registered equal-budget ensemble can be recomputed without
        # retraining or reopening model selection.
        "test_scores_id": sid.astype(np.float32).tolist(),
        "test_scores_ood": sood.astype(np.float32).tolist(),
        "seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    ap.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    stage_path = Path("repro") / f"rpo_opt_stage2_{args.cell}_{args.backbone}.json"
    stage = json.loads(stage_path.read_text())
    grid = S.recipes()
    if S.stable_hash(grid) != P.EXPECTED_RECIPE_HASH:
        raise RuntimeError("recipe drift")
    raw, audit = load_all(args.cell, args.backbone)
    device = torch.device(args.device)
    out = {
        "protocol": "rpo-opt-final-v1",
        "cell": args.cell,
        "backbone": args.backbone,
        "stage2_sha256": file_sha256(stage_path),
        "recipe_hash": P.EXPECTED_RECIPE_HASH,
        "final_seeds": list(FINAL_SEEDS),
        "domain_split": audit,
        "selection": {},
        "results": {},
    }
    for objective in ("rpo", "bce"):
        rid, sel = select_recipe(stage, objective)
        rec = grid[rid]
        out["selection"][objective] = {"recipe_id": rid, "recipe": rec, **sel}
        arrays = S.transformed(raw, rec["normalization"])
        runs = []
        for seed in FINAL_SEEDS:
            result = train_final(arrays, objective, rec, seed, device)
            runs.append(result)
            print(
                f"[{args.cell}/{args.backbone}] {objective} r={rid:02d} s={seed} "
                f"val={result['val_auroc']:.4f} test={result['test_auroc']:.4f} "
                f"FPR95={result['test_fpr95']:.4f}",
                flush=True,
            )
        out["results"][objective] = runs
    target = Path("repro") / f"rpo_opt_final_{args.cell}_{args.backbone}.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"FINAL_DONE {target} hash={S.stable_hash(out)}", flush=True)


if __name__ == "__main__":
    main()
