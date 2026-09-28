#!/usr/bin/env python3
"""Supervision-matched, equal-48 validation pipeline for OE baselines.

This file intentionally does not import or inspect any existing test result.  Every
method-specific grid is a deterministic function below, with exactly 48 recipes.  The
``screen`` action loads only train/validation arrays.  The ``final`` action is disabled
until a five-seed validation selection artifact has been written.

The molecule draw is byte-for-byte matched to RPO-OE v2: NumPy ``default_rng(seed)``
permutations, at most 1,500 train-ID and 2,000 auxiliary-OOD molecules.  Higher scores
always mean more OOD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
from typing import Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matched_oe as M
import revision_run as RR
import rpo_opt_screen as RPOS


torch.set_num_threads(1)

PROTOCOL = "oe-equal48-supervision-matched-v2"
GRID_PROTOCOL = "oe-equal48-fixed-grids-and-training-20260813-v2"
SELECTION_SEEDS = (21, 22, 23, 24, 25)
FINAL_SEEDS = tuple(range(301, 321))
# ``CELLS`` is the frozen four-setting primary panel consumed by the original
# paper aggregate.  ``TABLE1_CELLS`` extends the exact same protocol to every
# column in the manuscript's Table 1; keeping the two scopes explicit prevents
# an expanded run from silently changing the primary estimand.
CELLS = ("ec50_assay", "ic50_assay")
TABLE1_CELLS = (
    "ec50_scaffold",
    "ec50_size",
    "ec50_assay",
    "ic50_scaffold",
    "ic50_size",
    "ic50_assay",
    "hiv_scaffold",
    "hiv_size",
    "pcba_scaffold",
    "pcba_size",
    "zinc_scaffold",
    "zinc_size",
)
BACKBONES = ("minimol", "unimol")
METHODS = (
    "MSP-OE",
    "ODIN-OE",
    "Energy-OE",
    "OE-Mahalanobis",
    "OE-KNN",
    "OE-LOF",
)
N_ID = 1500
N_OOD = 2000
CLASSIFIER_BUDGET = 1_024_000
N_CHECKPOINTS = 100
WARMUP_FRACTION = 0.05
EMA_DECAY = 0.995
CLASSIFIER_HEAD = "d-256-128-C_dropout0.1"
SPLITS_BY_PHASE = {
    "screen": ("train_id", "train_ood", "val_id", "val_ood"),
    "final": ("train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood"),
}


def stable_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numbered(rows: list[dict]) -> list[dict]:
    return [{"id": index, **row} for index, row in enumerate(rows)]


def recipes(method: str) -> list[dict]:
    """Return a fixed, data-independent 48-point grid for one method."""
    rows: list[dict]
    if method == "MSP-OE":
        rows = [
            {
                "normalization": norm,
                "lr": lr,
                "weight_decay": wd,
                "oe_weight": lam,
            }
            for norm in ("zscore", "lw_whiten")
            for lr in (3e-3, 1e-2, 3e-2)
            for wd in (5e-4, 5e-3)
            for lam in (0.03, 0.1, 0.3, 1.0)
        ]
    elif method == "ODIN-OE":
        # Unlike matched_oe.py's nine-point grid, classifier learning rate and OE
        # weight are selected jointly with temperature.  Weight decay is fixed.
        rows = [
            {
                "normalization": norm,
                "lr": lr,
                "weight_decay": 5e-4,
                "oe_weight": lam,
                "temperature": temperature,
            }
            for norm in ("zscore", "lw_whiten")
            for lr in (3e-3, 1e-2, 3e-2)
            for lam in (0.1, 0.5)
            for temperature in (1.0, 10.0, 100.0, 1000.0)
        ]
    elif method == "Energy-OE":
        # Four ordered (m_in, m_out) pairs span the range used in the original
        # matched-OE experiment and one wider setting.  Optimizer LR and penalty
        # strength are selected jointly; weight decay is fixed.
        rows = [
            {
                "normalization": norm,
                "lr": lr,
                "weight_decay": 5e-4,
                "energy_weight": lam,
                "margin_in": margins[0],
                "margin_out": margins[1],
            }
            for norm in ("zscore", "lw_whiten")
            for lr in (3e-3, 1e-2)
            for lam in (0.03, 0.1, 0.3)
            for margins in ((-9.0, -5.0), (-7.0, -5.0), (-5.0, -3.0), (-3.0, -1.0))
        ]
    elif method == "OE-Mahalanobis":
        rows = [
            {"normalization": norm, "eps": eps, "covariance": covariance}
            for norm in ("zscore", "lw_whiten")
            for eps in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
            for covariance in ("shared", "sep", "diag")
        ]
    elif method == "OE-KNN":
        rows = [
            {"normalization": norm, "k_id": k_id, "k_ood": k_ood}
            for norm in ("zscore", "lw_whiten")
            for k_id in (1, 5, 20, 100)
            for k_ood in (1, 5, 20, 50, 100, 200)
        ]
    elif method == "OE-LOF":
        rows = [
            {"normalization": norm, "n_id": n_id, "n_ood": n_ood}
            for norm in ("zscore", "lw_whiten")
            for n_id in (5, 20, 50, 200)
            for n_ood in (5, 20, 50, 100, 200, 500)
        ]
    else:
        raise ValueError(f"unknown method: {method}")
    grid = _numbered(rows)
    if len(grid) != 48 or len({stable_hash(row) for row in grid}) != 48:
        raise RuntimeError(f"{method} grid is not 48 unique recipes")
    return grid


def grid_spec(method: str) -> dict:
    """Hash recipes together with fixed training semantics.

    A recipe-list-only hash would fail to change if checkpointing or capacity changed.
    The protocol hash therefore commits to both the 48 configurations and all shared
    optimizer/checkpoint/head constants.
    """
    return {
        "grid_protocol": GRID_PROTOCOL,
        "method": method,
        "n_id": N_ID,
        "n_ood": N_OOD,
        "classifier_budget": CLASSIFIER_BUDGET,
        "n_checkpoints": N_CHECKPOINTS,
        "warmup_fraction": WARMUP_FRACTION,
        "ema_decay": EMA_DECAY,
        "classifier_head": CLASSIFIER_HEAD,
        "recipes": recipes(method),
    }


def grid_hash(method: str) -> str:
    return stable_hash(grid_spec(method))


def all_grid_hash() -> str:
    return stable_hash({method: grid_spec(method) for method in METHODS})


def _load_phase(cell: str, backbone: str, phase: str) -> tuple[dict, dict]:
    """Load only the arrays allowed in the requested phase.

    The split manifest necessarily names all official partitions, but ``screen`` never
    materializes a test feature array and no scoring function receives one.
    """
    if phase not in SPLITS_BY_PHASE:
        raise ValueError(phase)
    base = RR.BASE[cell]
    feature_path = RPOS.CACHE / f"{base}_{backbone}_features.pkl"
    split_path = RPOS.CACHE / f"{base}_seed42_splits.json"
    with feature_path.open("rb") as handle:
        features = pickle.load(handle)["features"]
    split = json.loads(split_path.read_text())["splits"]
    split, audit = RR.domain_disjoint_split(cell, split)
    data = {}
    for key in SPLITS_BY_PHASE[phase]:
        smiles = [s for s in split[key] if s in features]
        data[key] = {
            "x": np.stack([features[s] for s in smiles]).astype(np.float32),
            "smiles": smiles,
        }
    return data, {
        "domain_split": audit,
        "feature_file": str(feature_path),
        "split_file": str(split_path),
        "loaded_splits": list(SPLITS_BY_PHASE[phase]),
    }


def _transform_phase(data: dict, normalization: str) -> dict[str, np.ndarray]:
    arrays = {key: value["x"] for key, value in data.items()}
    return RPOS.transformed(arrays, normalization)


def _draw_indices(seed: int, n_id_pool: int, n_ood_pool: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    iid = rng.permutation(n_id_pool)[: min(N_ID, n_id_pool)]
    iod = rng.permutation(n_ood_pool)[: min(N_OOD, n_ood_pool)]
    return iid, iod


def _index_audit(iid: np.ndarray, iod: np.ndarray) -> dict:
    return {
        "n_id": int(len(iid)),
        "n_ood": int(len(iod)),
        "id_index_sha256": hashlib.sha256(iid.astype("<i8").tobytes()).hexdigest(),
        "ood_index_sha256": hashlib.sha256(iod.astype("<i8").tobytes()).hexdigest(),
    }


def _contexts(data: dict, labels: np.ndarray, seed: int) -> tuple[dict[str, dict], dict]:
    iid, iod = _draw_indices(seed, len(data["train_id"]["x"]), len(data["train_ood"]["x"]))
    transformed = {
        norm: _transform_phase(data, norm) for norm in ("zscore", "lw_whiten")
    }
    by_norm = {}
    for norm, arrays in transformed.items():
        y = labels[iid]
        keep = y >= 0
        by_norm[norm] = {
            "Xid": arrays["train_id"][iid],
            "Xood": arrays["train_ood"][iod],
            "Xid_lab": arrays["train_id"][iid][keep],
            "y": y[keep],
            "arrays": arrays,
        }
    audit = _index_audit(iid, iod)
    audit["n_labelled_id"] = int((labels[iid] >= 0).sum())
    return by_norm, audit


class ClassifierHead(nn.Module):
    """Capacity-matched hidden trunk for classifier-score OE baselines."""

    def __init__(self, dimension: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _scores_from_logits(logits: torch.Tensor, method: str, recipe: dict) -> torch.Tensor:
    if method in ("MSP-OE", "ODIN-OE"):
        temperature = float(recipe.get("temperature", 1.0))
        return 1.0 - F.softmax(logits / temperature, dim=1).max(dim=1).values
    if method == "Energy-OE":
        return -torch.logsumexp(logits, dim=1)
    raise ValueError(method)


def _fit_classifier(
    ctx: dict,
    method: str,
    recipe: dict,
    seed: int,
    device: torch.device,
    *,
    budget: int = CLASSIFIER_BUDGET,
    n_checkpoints: int = N_CHECKPOINTS,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    x_id = torch.as_tensor(ctx["Xid_lab"], dtype=torch.float32, device=device)
    y_id = torch.as_tensor(ctx["y"], dtype=torch.long, device=device)
    x_ood = torch.as_tensor(ctx["Xood"], dtype=torch.float32, device=device)
    val_id = torch.as_tensor(ctx["arrays"]["val_id"], dtype=torch.float32, device=device)
    val_ood = torch.as_tensor(ctx["arrays"]["val_ood"], dtype=torch.float32, device=device)
    n_classes = int(y_id.max().item()) + 1
    model = ClassifierHead(x_id.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(recipe["lr"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    presentations_per_step = len(x_id) + len(x_ood)
    steps = max(n_checkpoints, budget // presentations_per_step)
    warmup = max(1, int(WARMUP_FRACTION * steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(1, steps - warmup - 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    ema = RPOS.clone_state(model)
    checkpoint_steps = set(
        int(value) for value in np.linspace(1, steps, num=n_checkpoints, dtype=np.int64)
    )
    if len(checkpoint_steps) != n_checkpoints:
        raise RuntimeError("checkpoint schedule did not contain exactly the requested count")
    best_auc = -1.0
    best_step = -1
    best_state = None
    best_metrics = None
    curve = []
    for step in range(steps):
        model.train()
        logits_id = model(x_id)
        logits_ood = model(x_ood)
        loss = F.cross_entropy(logits_id, y_id)
        if method in ("MSP-OE", "ODIN-OE"):
            uniform_ce = -F.log_softmax(logits_ood, dim=1).mean(dim=1).mean()
            loss = loss + float(recipe["oe_weight"]) * uniform_ce
        elif method == "Energy-OE":
            energy_id = -torch.logsumexp(logits_id, dim=1)
            energy_ood = -torch.logsumexp(logits_ood, dim=1)
            penalty = F.relu(energy_id - float(recipe["margin_in"])).square().mean()
            penalty = penalty + F.relu(float(recipe["margin_out"]) - energy_ood).square().mean()
            loss = loss + float(recipe["energy_weight"]) * penalty
        else:
            raise ValueError(method)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite {method} loss for recipe {recipe['id']}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        RPOS.update_ema(ema, model, EMA_DECAY)

        if step + 1 in checkpoint_steps:
            raw_state = RPOS.clone_state(model)
            model.load_state_dict(ema)
            model.eval()
            with torch.no_grad():
                score_id = _scores_from_logits(model(val_id), method, recipe).float().cpu().numpy()
                score_ood = _scores_from_logits(model(val_ood), method, recipe).float().cpu().numpy()
            val_tuple = RR.mets(score_id, score_ood)
            metrics = {
                "auroc": float(val_tuple[0]),
                "aupr": float(val_tuple[1]),
                "fpr95": float(val_tuple[2]),
            }
            curve.append({"step": step + 1, **metrics})
            if metrics["auroc"] > best_auc:
                best_auc = metrics["auroc"]
                best_step = step + 1
                best_state = RPOS.clone_state(model)
                best_metrics = metrics
            model.load_state_dict(raw_state)

    if best_state is None or best_metrics is None:
        raise RuntimeError("no classifier checkpoint was evaluated")
    model.load_state_dict(best_state)
    model.eval()
    return model, {
        "budget": int(budget),
        "presentations_per_step": int(presentations_per_step),
        "actual_presentations": int(steps * presentations_per_step),
        "steps": int(steps),
        "warmup_steps": int(warmup),
        "scheduler": "5%-warmup-cosine",
        "ema_decay": EMA_DECAY,
        "n_checkpoints": len(curve),
        "best_step": int(best_step),
        "best_val": best_metrics,
        "curve": curve,
    }


def _classifier_scorer(model, method: str, recipe: dict, device: torch.device) -> Callable:
    def score(array: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.as_tensor(array, dtype=torch.float32, device=device))
            values = _scores_from_logits(logits, method, recipe)
            return values.float().cpu().numpy()

    return score


def _build_scorer(
    method: str,
    recipe: dict,
    ctx: dict,
    seed: int,
    device: torch.device,
    cache: dict,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    if method in ("MSP-OE", "ODIN-OE", "Energy-OE"):
        # For a two-logit classifier, max softmax at T is sigmoid(|z1-z0|/T), a
        # strictly monotone transform of |z1-z0| for every T>0.  Thus AUROC, AUPR,
        # FPR95, and the best-AUROC checkpoint are temperature invariant.  Only in
        # that proven binary case may ODIN temperatures reuse a trajectory.
        n_classes = int(np.max(ctx["y"])) + 1
        odin_binary_invariant = method == "ODIN-OE" and n_classes == 2
        model_key = stable_hash(
            {
                key: value
                for key, value in recipe.items()
                if key != "id" and not (odin_binary_invariant and key == "temperature")
            }
        )
        if model_key not in cache:
            cache[model_key] = _fit_classifier(ctx, method, recipe, seed, device)
        model, fit_audit = cache[model_key]
        fit_audit = dict(fit_audit)
        fit_audit["odin_temperature_trajectory_reused"] = bool(odin_binary_invariant)
        if odin_binary_invariant:
            fit_audit["temperature_invariance_proof"] = (
                "For C=2, max softmax(z/T)=sigmoid(abs(z1-z0)/T), strictly monotone "
                "in abs(z1-z0) for T>0; all evaluated ranking metrics and checkpoint "
                "selection are invariant."
            )
        return _classifier_scorer(model, method, recipe, device), fit_audit
    if method == "OE-Mahalanobis":
        scorer = M.maha_oe(
            ctx["Xid"], ctx["Xood"], float(recipe["eps"]), recipe["covariance"]
        )
        return scorer, {"kind": "closed-form-no-checkpoint"}
    if method == "OE-KNN":
        scorer = M.knn_oe(
            ctx["Xid"], ctx["Xood"], int(recipe["k_id"]), int(recipe["k_ood"])
        )
        return scorer, {"kind": "closed-form-no-checkpoint"}
    if method == "OE-LOF":
        scorer = M.lof_oe(
            ctx["Xid"], ctx["Xood"], int(recipe["n_id"]), int(recipe["n_ood"])
        )
        return scorer, {"kind": "closed-form-no-checkpoint"}
    raise ValueError(method)


def _metrics(scorer: Callable, arrays: dict, split_id: str, split_ood: str) -> dict:
    score_id = np.nan_to_num(np.asarray(scorer(arrays[split_id]), dtype=np.float64))
    score_ood = np.nan_to_num(np.asarray(scorer(arrays[split_ood]), dtype=np.float64))
    auroc, aupr, fpr95 = RR.mets(score_id, score_ood)
    return {"auroc": float(auroc), "aupr": float(aupr), "fpr95": float(fpr95)}


def _labels_for_train(cell: str, train_smiles: list[str]) -> np.ndarray:
    label_lookup = M.label_map(cell)
    labels = np.asarray([label_lookup.get(smiles, -1) for smiles in train_smiles], dtype=np.int64)
    usable = labels >= 0
    if usable.sum() <= 50 or len(np.unique(labels[usable])) <= 1:
        raise RuntimeError(f"{cell} has insufficient downstream ID labels for classifier OE baselines")
    return labels


def _screen_path(cell: str, backbone: str, method: str, seed: int) -> Path:
    slug = method.lower().replace("-", "_")
    return Path("repro") / f"oe_equal48_screen_{cell}_{backbone}_{slug}_s{seed}.json"


def _selection_path(cell: str, backbone: str, method: str) -> Path:
    slug = method.lower().replace("-", "_")
    return Path("repro") / f"oe_equal48_selection_{cell}_{backbone}_{slug}.json"


def _final_path(cell: str, backbone: str, method: str, seed: int) -> Path:
    slug = method.lower().replace("-", "_")
    return Path("repro") / f"oe_equal48_final_{cell}_{backbone}_{slug}_s{seed}.json"


def screen(cell: str, backbone: str, method: str, seed: int, device: torch.device) -> Path:
    grid = recipes(method)
    data, data_audit = _load_phase(cell, backbone, "screen")
    labels = _labels_for_train(cell, data["train_id"]["smiles"])
    contexts, draw_audit = _contexts(data, labels, seed)
    result = {
        "protocol": PROTOCOL,
        "phase": "validation-only-screen",
        "grid_protocol": GRID_PROTOCOL,
        "cell": cell,
        "backbone": backbone,
        "method": method,
        "seed": seed,
        "selection_seeds": list(SELECTION_SEEDS),
        "n_recipe_trials": 48,
        "n_checkpoint_queries_per_neural_recipe": N_CHECKPOINTS,
        "recipe_hash": grid_hash(method),
        "all_grid_hash": all_grid_hash(),
        "recipes": grid,
        "data_audit": data_audit,
        "draw_audit": draw_audit,
        "results": {},
    }
    cache: dict = {}
    for recipe in grid:
        ctx = contexts[recipe["normalization"]]
        scorer, fit_audit = _build_scorer(method, recipe, ctx, seed, device, cache)
        if method in ("MSP-OE", "ODIN-OE", "Energy-OE"):
            metrics = fit_audit["best_val"]
        else:
            metrics = _metrics(scorer, ctx["arrays"], "val_id", "val_ood")
        result["results"][str(recipe["id"])] = {
            "recipe": recipe,
            "val": metrics,
            "fit": fit_audit,
        }
        print(
            f"[{cell}/{backbone}/{method}/s{seed}] r={recipe['id']:02d} "
            f"val_auroc={metrics['auroc']:.5f}",
            flush=True,
        )
    target = _screen_path(cell, backbone, method, seed)
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(f"OE_EQUAL48_SCREEN_DONE {target} hash={stable_hash(result)}", flush=True)
    return target


def select(cell: str, backbone: str, method: str) -> Path:
    grid = recipes(method)
    recipe_hash = grid_hash(method)
    documents = []
    screen_provenance = []
    for seed in SELECTION_SEEDS:
        path = _screen_path(cell, backbone, method, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        doc = json.loads(path.read_text())
        if doc["phase"] != "validation-only-screen" or doc["seed"] != seed:
            raise RuntimeError(f"invalid screen artifact: {path}")
        if doc["recipe_hash"] != recipe_hash or len(doc["results"]) != 48:
            raise RuntimeError(f"grid drift or incomplete screen: {path}")
        documents.append(doc)
        screen_provenance.append({"path": str(path), "sha256": file_sha256(path)})
    ranking = []
    for recipe in grid:
        rid = str(recipe["id"])
        values = [float(doc["results"][rid]["val"]["auroc"]) for doc in documents]
        ranking.append(
            {
                "recipe_id": recipe["id"],
                "mean_val_auroc": float(np.mean(values)),
                "worst_seed_val_auroc": float(np.min(values)),
                "per_seed_val_auroc": values,
            }
        )
    ranking.sort(
        key=lambda row: (
            row["mean_val_auroc"],
            row["worst_seed_val_auroc"],
            -row["recipe_id"],
        ),
        reverse=True,
    )
    chosen = ranking[0]
    output = {
        "protocol": PROTOCOL,
        "phase": "selection-frozen-before-final",
        "grid_protocol": GRID_PROTOCOL,
        "cell": cell,
        "backbone": backbone,
        "method": method,
        "selection_seeds": list(SELECTION_SEEDS),
        "n_recipe_trials": 48,
        "n_checkpoint_queries_per_neural_recipe": N_CHECKPOINTS,
        "selection_rule": "mean val AUROC; tie-break worst seed then lowest recipe ID",
        "recipe_hash": recipe_hash,
        "all_grid_hash": all_grid_hash(),
        "selected": {**chosen, "recipe": grid[chosen["recipe_id"]]},
        "top5": ranking[:5],
        "screen_provenance": screen_provenance,
    }
    target = _selection_path(cell, backbone, method)
    target.write_text(json.dumps(output, indent=2))
    print(
        f"OE_EQUAL48_SELECTION_FROZEN {target} r={chosen['recipe_id']} "
        f"val={chosen['mean_val_auroc']:.5f}",
        flush=True,
    )
    return target


def final(cell: str, backbone: str, method: str, seed: int, device: torch.device) -> Path:
    selection_path = _selection_path(cell, backbone, method)
    if not selection_path.exists():
        raise RuntimeError(f"selection must be frozen before final evaluation: {selection_path}")
    selection = json.loads(selection_path.read_text())
    if selection["phase"] != "selection-frozen-before-final":
        raise RuntimeError(f"selection is not frozen: {selection_path}")
    grid = recipes(method)
    if selection["recipe_hash"] != grid_hash(method):
        raise RuntimeError(f"recipe grid changed after selection: {selection_path}")
    recipe = selection["selected"]["recipe"]
    if recipe != grid[int(selection["selected"]["recipe_id"])]:
        raise RuntimeError(f"selected recipe does not match frozen grid: {selection_path}")

    data, data_audit = _load_phase(cell, backbone, "final")
    labels = _labels_for_train(cell, data["train_id"]["smiles"])
    iid, iod = _draw_indices(seed, len(data["train_id"]["x"]), len(data["train_ood"]["x"]))
    arrays = _transform_phase(data, recipe["normalization"])
    y = labels[iid]
    keep = y >= 0
    ctx = {
        "Xid": arrays["train_id"][iid],
        "Xood": arrays["train_ood"][iod],
        "Xid_lab": arrays["train_id"][iid][keep],
        "y": y[keep],
        "arrays": arrays,
    }
    scorer, fit_audit = _build_scorer(method, recipe, ctx, seed, device, {})
    metrics = _metrics(scorer, arrays, "test_id", "test_ood")
    draw_audit = _index_audit(iid, iod)
    draw_audit["n_labelled_id"] = int(keep.sum())
    output = {
        "protocol": PROTOCOL,
        "phase": "frozen-final-one-seed",
        "grid_protocol": GRID_PROTOCOL,
        "cell": cell,
        "backbone": backbone,
        "method": method,
        "seed": seed,
        "final_seeds": list(FINAL_SEEDS),
        "selection_sha256": file_sha256(selection_path),
        "recipe_hash": grid_hash(method),
        "selected_recipe_id": int(selection["selected"]["recipe_id"]),
        "selected_recipe": recipe,
        "data_audit": data_audit,
        "draw_audit": draw_audit,
        "fit": fit_audit,
        "test": metrics,
    }
    target = _final_path(cell, backbone, method, seed)
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(output, indent=2))
    print(
        f"OE_EQUAL48_FINAL_DONE {target} AUROC={metrics['auroc']:.5f} "
        f"AUPR={metrics['aupr']:.5f} FPR95={metrics['fpr95']:.5f}",
        flush=True,
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("screen", "select", "final", "show-grids", "self-test")
    )
    parser.add_argument("--cell", choices=TABLE1_CELLS)
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.action == "show-grids":
        print(
            json.dumps(
                {
                    "grid_protocol": GRID_PROTOCOL,
                    "all_grid_hash": all_grid_hash(),
                    "methods": {
                        method: {"n": len(recipes(method)), "sha256": grid_hash(method)}
                        for method in METHODS
                    },
                },
                indent=2,
            )
        )
        return
    if args.action == "self-test":
        rng = np.random.default_rng(17)
        arrays = {
            "train_id": rng.normal(0.0, 1.0, (16, 8)).astype(np.float32),
            "train_ood": rng.normal(0.6, 1.0, (20, 8)).astype(np.float32),
            "val_id": rng.normal(0.0, 1.0, (12, 8)).astype(np.float32),
            "val_ood": rng.normal(0.6, 1.0, (12, 8)).astype(np.float32),
        }
        ctx = {
            "Xid": arrays["train_id"],
            "Xood": arrays["train_ood"],
            "Xid_lab": arrays["train_id"],
            "y": np.asarray([0, 1] * 8, dtype=np.int64),
            "arrays": arrays,
        }
        recipe = recipes("ODIN-OE")[0]
        model, audit = _fit_classifier(
            ctx,
            "ODIN-OE",
            recipe,
            seed=17,
            device=torch.device("cpu"),
            budget=144,
            n_checkpoints=4,
        )
        scorer = _classifier_scorer(model, "ODIN-OE", recipe, torch.device("cpu"))
        metrics = _metrics(scorer, arrays, "val_id", "val_ood")
        assert audit["n_checkpoints"] == 4
        assert all(np.isfinite(value) for value in metrics.values())
        print(json.dumps({"self_test": "PASS", "fit": audit, "metrics": metrics}, indent=2))
        return
    if args.cell is None or args.backbone is None or args.method is None:
        parser.error("screen/select/final require --cell, --backbone, and --method")
    if args.action == "screen":
        if args.seed not in SELECTION_SEEDS:
            parser.error(f"screen seed must be one of {SELECTION_SEEDS}")
        screen(args.cell, args.backbone, args.method, args.seed, torch.device(args.device))
    elif args.action == "select":
        if args.seed is not None:
            parser.error("select does not accept --seed")
        select(args.cell, args.backbone, args.method)
    else:
        if args.seed not in FINAL_SEEDS:
            parser.error(f"final seed must be one of {FINAL_SEEDS}")
        final(args.cell, args.backbone, args.method, args.seed, torch.device(args.device))


if __name__ == "__main__":
    main()
