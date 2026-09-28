#!/usr/bin/env python3
"""Frozen-recipe limited-OE thinning gate for RPO-Opt versus balanced BCE.

This is deliberately a *thinning robustness* experiment, not a low-budget recipe search.
For each catalog seed, a single permutation of the auxiliary-OOD training pool is created;
budget K uses the first K unique molecules, so all points on a curve are nested.  RPO and
BCE receive exactly the same ID/OOD molecule sets.  Their recipes are frozen from the
five-seed, full-OE validation selection in rpo_opt_stage2.py.  Validation OOD remains the
complete domain-disjoint validation set, so K denotes the number of unique *training* OE
molecules rather than the total number of labelled OOD molecules available to the pipeline.

The gate uses three previously unused (catalog, initialization) pairs and a fixed
molecule-presentation budget.  A positive gate must be followed by an independent
validation-AULC recipe selection before making a sample-efficiency claim.
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
import rpo_opt_final as FINAL


BUDGETS = (4, 8, 16, 32, 64, 128, 256, 2000)
REPLICATES = ((7001, 211), (7003, 211), (7005, 211))
ID_CATALOG_SEED = 424_242
PRESENTATION_BUDGET = 512_000


def hash_strings(xs: list[str]) -> str:
    h = hashlib.sha256()
    for x in xs:
        h.update(x.encode())
        h.update(b"\0")
    return h.hexdigest()


def load_all_with_smiles(cell: str, backbone: str):
    base = RR.BASE[cell]
    feat = pickle.load(open(S.CACHE / f"{base}_{backbone}_features.pkl", "rb"))["features"]
    split = json.load(open(S.CACHE / f"{base}_seed42_splits.json"))["splits"]
    split, audit = RR.domain_disjoint_split(cell, split)
    keys = ("train_id", "train_ood", "val_id", "val_ood", "test_id", "test_ood")
    smiles, arrays = {}, {}
    for key in keys:
        smiles[key] = [s for s in split[key] if s in feat]
        arrays[key] = np.stack([feat[s] for s in smiles[key]]).astype(np.float32)

    raw = json.load(open(f"data/raw/{base}.json"))["split"]
    domain = {
        item["smiles"]: item.get("domain_id")
        for item in raw.get("ood_val", [])
        if item.get("smiles")
    }
    return arrays, smiles, domain, audit


def clone_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_limited(
    arrays: dict[str, np.ndarray],
    objective: str,
    recipe: dict,
    id_indices: np.ndarray,
    ood_indices: np.ndarray,
    init_seed: int,
    device: torch.device,
) -> dict:
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(init_seed)

    xid = torch.as_tensor(arrays["train_id"][id_indices], device=device)
    xood = torch.as_tensor(arrays["train_ood"][ood_indices], device=device)
    vid = torch.as_tensor(arrays["val_id"], device=device)
    vood = torch.as_tensor(arrays["val_ood"], device=device)
    tid = torch.as_tensor(arrays["test_id"], device=device)
    tood = torch.as_tensor(arrays["test_ood"], device=device)

    model = S.OriginalHead(xid.shape[1], float(recipe["dropout"])).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"])
    )
    batch = int(recipe["batch"])
    steps = max(S.N_CHECKPOINTS, PRESENTATION_BUDGET // (2 * batch))
    warmup = max(1, int(0.05 * steps))

    def lr_factor(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        p = (step - warmup) / max(1, steps - warmup - 1)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    ema = clone_state(model)
    best_auc, best_step, best_state = -1.0, -1, None
    eval_every = max(1, steps // S.N_CHECKPOINTS)
    index_gen = torch.Generator(device=device.type).manual_seed(init_seed + 1729)
    clipped = 0
    t0 = time.time()
    for step in range(steps):
        model.train()
        ii = torch.randint(len(xid), (batch,), generator=index_gen, device=device)
        io = torch.randint(len(xood), (batch,), generator=index_gen, device=device)
        ei, eo = model(xid[ii]), model(xood[io])
        if objective == "rpo":
            core = F.softplus(-(eo[:, None] - ei[None, :])).mean()
        elif objective == "bce":
            core = 0.5 * F.binary_cross_entropy_with_logits(ei, torch.zeros_like(ei))
            core += 0.5 * F.binary_cross_entropy_with_logits(eo, torch.ones_like(eo))
        else:
            raise ValueError(objective)
        loss = core + float(recipe["gamma"]) * (ei.square().mean() + eo.square().mean())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        clipped += int(float(grad) > 1.0)
        opt.step()
        sched.step()
        S.update_ema(ema, model, S.EMA_DECAY)

        if (step + 1) % eval_every == 0 or step + 1 == steps:
            raw = clone_state(model)
            model.load_state_dict(ema)
            auc = S.val_auc(model, vid, vood)
            if auc > best_auc:
                best_auc, best_step, best_state = auc, step + 1, clone_state(model)
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
        "seconds": time.time() - t0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=("ec50_assay", "ic50_assay"), required=True)
    ap.add_argument("--backbone", choices=("minimol", "unimol"), required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    ap.add_argument("--replicates", nargs="+", default=[f"{c}:{i}" for c, i in REPLICATES])
    ap.add_argument("--shared-recipe", type=int, default=None)
    args = ap.parse_args()

    reps = []
    for item in args.replicates:
        c, i = item.split(":")
        reps.append((int(c), int(i)))
    budgets = tuple(sorted(set(args.budgets)))
    stage_path = Path("repro") / f"rpo_opt_stage2_{args.cell}_{args.backbone}.json"
    stage = json.loads(stage_path.read_text())
    grid = S.recipes()
    raw, smiles, domains, audit = load_all_with_smiles(args.cell, args.backbone)
    device = torch.device(args.device)

    selections = {}
    transformed = {}
    for objective in ("rpo", "bce"):
        if args.shared_recipe is None:
            rid, sel = FINAL.select_recipe(stage, objective)
        else:
            rid = args.shared_recipe
            sel = {"mean_val": None, "worst_seed_val": None, "selection_source": "objective-blind shared-recipe crossover"}
        recipe = grid[rid]
        selections[objective] = {"recipe_id": rid, "recipe": recipe, **sel}
        transformed[objective] = S.transformed(raw, recipe["normalization"])

    out = {
        "protocol": "rpo-opt-thinning-gate-v1",
        "scope": "unique training-OE thinning; full fixed domain-disjoint OOD validation",
        "cell": args.cell,
        "backbone": args.backbone,
        "budgets_requested": list(budgets),
        "replicates": [{"catalog_seed": c, "init_seed": i} for c, i in reps],
        "id_catalog_seed": ID_CATALOG_SEED,
        "presentation_budget": PRESENTATION_BUDGET,
        "domain_split": audit,
        "selection": selections,
        "results": [],
    }

    id_rng = np.random.default_rng(ID_CATALOG_SEED)
    id_perm = id_rng.permutation(len(raw["train_id"]))
    id_idx = id_perm[: min(1500, len(id_perm))]
    for catalog_seed, init_seed in reps:
        rng = np.random.default_rng(catalog_seed)
        ood_perm = rng.permutation(len(raw["train_ood"]))
        for requested in budgets:
            k = min(requested, len(ood_perm))
            ood_idx = np.sort(ood_perm[:k])
            selected_smiles = [smiles["train_ood"][j] for j in ood_idx]
            doms = {domains[s] for s in selected_smiles if s in domains}
            catalog = {
                "catalog_seed": catalog_seed,
                "init_seed": init_seed,
                "budget_requested": requested,
                "n_unique_ood": k,
                "n_unique_id": len(id_idx),
                "n_ood_domains": len(doms),
                "id_sha256": hash_strings([smiles["train_id"][j] for j in id_idx]),
                "ood_sha256": hash_strings(selected_smiles),
                "metrics": {},
            }
            for objective in ("rpo", "bce"):
                result = train_limited(
                    transformed[objective], objective, selections[objective]["recipe"],
                    id_idx, ood_idx, init_seed, device,
                )
                catalog["metrics"][objective] = result
                print(
                    f"[{args.cell}/{args.backbone}] cat={catalog_seed} init={init_seed} "
                    f"K={k:04d} {objective} val={result['val_auroc']:.4f} "
                    f"test={result['test_auroc']:.4f}",
                    flush=True,
                )
            out["results"].append(catalog)

    suffix = "" if args.shared_recipe is None else f"_sharedr{args.shared_recipe}"
    target = Path("repro") / f"rpo_opt_thinning_{args.cell}_{args.backbone}{suffix}.json"
    target.write_text(json.dumps(out, indent=2))
    print(f"THINNING_DONE {target} hash={S.stable_hash(out)}", flush=True)


if __name__ == "__main__":
    main()

