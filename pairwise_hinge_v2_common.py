#!/usr/bin/env python3
"""Shared implementation for the matched Pairwise-Hinge / RankSVM v2 arm."""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

import revision_run as RR
import rpo_opt_screen as S
from rpo_opt_v2_extension import extension_recipes
from rpo_opt_v2_screen import FULL_BUDGET, N_CHECKPOINTS, SELECTION_SEEDS


MARGIN = 1.0
FINAL_SEEDS = tuple(range(301, 321))
EXPECTED_BASE_RECIPE_HASH = (
    "b20fbbc9883ae9021659d8c6e4919446e8ee80d6e305d67dab8b6f6cc1632feb"
)
EXPECTED_EXTENSION_RECIPE_HASH = (
    "8044044e6b3865d538daa9834836eede06cb8c2b920850768b058c6928a6c599"
)
EXPECTED_UNION_RECIPE_HASH = (
    "20d3841c1d3800f2bf8693c36d34482f7e59aefd8c2c7bbd01b64a0ac28f28de"
)


def frozen_recipes() -> list[dict]:
    """Return the exact 48 RPO-Opt v2 recipes, failing closed on drift."""
    base = S.recipes()
    extension = extension_recipes()
    union = base + extension
    if S.stable_hash(base) != EXPECTED_BASE_RECIPE_HASH:
        raise RuntimeError("base RPO-Opt v2 recipe drift")
    if S.stable_hash(extension) != EXPECTED_EXTENSION_RECIPE_HASH:
        raise RuntimeError("extension RPO-Opt v2 recipe drift")
    if S.stable_hash(union) != EXPECTED_UNION_RECIPE_HASH:
        raise RuntimeError("union RPO-Opt v2 recipe drift")
    ids = [int(recipe["id"]) for recipe in union]
    if ids != list(range(48)):
        raise RuntimeError(f"unexpected recipe IDs: {ids}")
    return union


def pairwise_hinge(eid: torch.Tensor, eood: torch.Tensor) -> torch.Tensor:
    """Complete all-pairs unit-margin RankSVM loss; larger means more OOD."""
    return F.relu(MARGIN - (eood[:, None] - eid[None, :])).mean()


def _seed(seed: int, device: torch.device) -> np.random.Generator:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    return np.random.default_rng(seed)


def _training_tensors(
    arrays: dict[str, np.ndarray], seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = _seed(seed, device)
    iid = rng.permutation(len(arrays["train_id"]))[: min(1500, len(arrays["train_id"]))]
    iod = rng.permutation(len(arrays["train_ood"]))[: min(2000, len(arrays["train_ood"]))]
    return (
        torch.as_tensor(arrays["train_id"][iid], device=device),
        torch.as_tensor(arrays["train_ood"][iod], device=device),
        torch.as_tensor(arrays["val_id"], device=device),
        torch.as_tensor(arrays["val_ood"], device=device),
    )


def train_validation(
    arrays: dict[str, np.ndarray], recipe: dict, seed: int, device: torch.device
) -> dict:
    """Train and select a checkpoint using validation data only."""
    xid, xood, vid, vood = _training_tensors(arrays, seed, device)
    model = S.OriginalHead(xid.shape[1], float(recipe["dropout"])).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["lr"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    batch = int(recipe["batch"])
    steps = max(N_CHECKPOINTS, FULL_BUDGET // (2 * batch))
    warmup = max(1, int(0.05 * steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup - 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    ema = S.clone_state(model)
    best_auc, best_step = -1.0, -1
    curve = []
    eval_every = max(1, steps // N_CHECKPOINTS)
    clipped = 0
    started = time.time()
    index_gen = torch.Generator(device=device.type).manual_seed(seed + 1729)

    for step in range(steps):
        model.train()
        ii = torch.randint(len(xid), (batch,), generator=index_gen, device=device)
        io = torch.randint(len(xood), (batch,), generator=index_gen, device=device)
        eid, eood = model(xid[ii]), model(xood[io])
        core = pairwise_hinge(eid, eood)
        reg = float(recipe["gamma"]) * (eid.square().mean() + eood.square().mean())
        loss = core + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped += int(float(grad) > 1.0)
        opt.step()
        sched.step()
        S.update_ema(ema, model, S.EMA_DECAY)

        if (step + 1) % eval_every == 0 or step + 1 == steps:
            raw_state = S.clone_state(model)
            model.load_state_dict(ema)
            auc = S.val_auc(model, vid, vood)
            curve.append({"step": step + 1, "val_auroc": auc})
            if auc > best_auc:
                best_auc, best_step = auc, step + 1
            model.load_state_dict(raw_state)

    return {
        "val_auroc": best_auc,
        "best_step": best_step,
        "steps": steps,
        "clip_rate": clipped / steps,
        "seconds": time.time() - started,
        "curve": curve,
    }


def train_final(
    arrays: dict[str, np.ndarray], recipe: dict, seed: int, device: torch.device
) -> dict:
    """Train after frozen selection; validation chooses checkpoint, test is scored once."""
    xid, xood, vid, vood = _training_tensors(arrays, seed, device)
    tid = torch.as_tensor(arrays["test_id"], device=device)
    tood = torch.as_tensor(arrays["test_ood"], device=device)
    model = S.OriginalHead(xid.shape[1], float(recipe["dropout"])).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(recipe["lr"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    batch = int(recipe["batch"])
    steps = max(N_CHECKPOINTS, FULL_BUDGET // (2 * batch))
    warmup = max(1, int(0.05 * steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup - 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    ema = S.clone_state(model)
    best_auc, best_step, best_state = -1.0, -1, None
    eval_every = max(1, steps // N_CHECKPOINTS)
    clipped = 0
    started = time.time()
    index_gen = torch.Generator(device=device.type).manual_seed(seed + 1729)

    for step in range(steps):
        model.train()
        ii = torch.randint(len(xid), (batch,), generator=index_gen, device=device)
        io = torch.randint(len(xood), (batch,), generator=index_gen, device=device)
        eid, eood = model(xid[ii]), model(xood[io])
        core = pairwise_hinge(eid, eood)
        reg = float(recipe["gamma"]) * (eid.square().mean() + eood.square().mean())
        loss = core + reg
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped += int(float(grad) > 1.0)
        opt.step()
        sched.step()
        S.update_ema(ema, model, S.EMA_DECAY)

        if (step + 1) % eval_every == 0 or step + 1 == steps:
            raw_state = S.clone_state(model)
            model.load_state_dict(ema)
            auc = S.val_auc(model, vid, vood)
            if auc > best_auc:
                best_auc, best_step, best_state = auc, step + 1, S.clone_state(model)
            model.load_state_dict(raw_state)

    if best_state is None:
        raise RuntimeError("no validation checkpoint was evaluated")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores_id = model(tid).float().cpu().numpy()
        scores_ood = model(tood).float().cpu().numpy()
    auroc, aupr, fpr95 = RR.mets(scores_id, scores_ood)
    return {
        "val_auroc": best_auc,
        "best_step": best_step,
        "steps": steps,
        "clip_rate": clipped / steps,
        "test_auroc": auroc,
        "test_aupr": aupr,
        "test_fpr95": fpr95,
        "test_scores_id": scores_id.astype(np.float32).tolist(),
        "test_scores_ood": scores_ood.astype(np.float32).tolist(),
        "seconds": time.time() - started,
    }

