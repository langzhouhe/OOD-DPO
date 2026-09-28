#!/usr/bin/env python3
"""Assemble every number the paper figures need into one JSON.

fig1  per-mechanism AUROC of a distance detector vs an exposure detector on unseen
      mechanisms -- the orthogonality that motivates the method
fig2  main result: delta over best_fixed per cell, per detector bank
fig3  applicability boundary vs OOD prevalence
fig4  batch-size axis
fig5  per-detector AUROC (the paper's own baseline table, measured per batch)

Usage: python figdata.py
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, re, glob, warnings, logging
import numpy as np, torch
torch.set_num_threads(8)
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
from sklearn.neighbors import NearestNeighbors

import revision_run as RR
import mepoe_gate0 as MG
from routing_gate import train_oe, N_FIT, SEEDS

CELLS = ["ec50_assay", "ic50_assay", "ki_assay", "ec50_scaffold", "ic50_scaffold"]
ASSAY = ["ec50_assay", "ic50_assay", "ki_assay"]
TAG = {"minimol": "_cv", "unimol": "_unimolcv", "both": "_bothcv"}
MAIN = "_d9"   # 10 catalog seeds, 9-detector bank
OUT = {}


def per_mechanism(cell, seed=21):
    """AUROC of Mahalanobis and of the OE head, one value per unseen mechanism."""
    d, _ = MG.setup(cell, seed); MG.G.clear(); MG.G.update(d)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(d["id_train"]))
    id_fit = d["id_train"][idx[:N_FIT]]
    Sm = d["S_mech"]; uS = np.unique(Sm); h = rng.permutation(len(uS))
    m_oe = set(uS[h[:len(uS) // 2]])
    S_oe = d["S"][torch.as_tensor(np.where(np.isin(Sm, list(m_oe)))[0])]
    tr = id_fit.numpy()
    mu = tr.mean(0); P = np.linalg.pinv(np.cov(tr, rowvar=False) + 1e-3 * np.eye(tr.shape[1]))
    md = lambda A: np.einsum("ij,jk,ik->i", A - mu, P, A - mu)
    heads = [train_oe(id_fit, S_oe, s) for s in SEEDS]
    def oe(A):
        t = torch.as_tensor(A, dtype=torch.float32)
        with torch.no_grad(): return np.mean([hh([t]).numpy() for hh in heads], 0)
    Xu, mun, off = [], [], 0
    for n in ("H1", "H2"):
        Xu.append(d[n].numpy()); mun.append(d[n + "_mech"] + off)
        off += int(d[n + "_mech"].max()) + 1
    Xu = np.concatenate(Xu); mun = np.concatenate(mun)
    te = d["id_test"].numpy()
    im, io = md(te), oe(te)
    ms, os_ = md(Xu), oe(Xu)
    out = []
    for m in np.unique(mun):
        sel = mun == m
        out.append([round(float(RR.mets(im, ms[sel])[0]), 4),
                    round(float(RR.mets(io, os_[sel])[0]), 4)])
    return out


# ---- fig 1 ----
OUT["fig1"] = {}
for c in ASSAY:
    OUT["fig1"][c] = per_mechanism(c)
    a = np.array(OUT["fig1"][c])
    print(f"[fig1] {c}: {len(a)} mechanisms, maha {a[:,0].mean():.3f} oe {a[:,1].mean():.3f}, "
          f"spearman {np.corrcoef(a[:,0].argsort().argsort(), a[:,1].argsort().argsort())[0,1]:+.3f}",
          flush=True)

# ---- fig 2 / fig 5 : from the CV-guarded runs ----
def agg(cell, bank, prev, key):
    v = []
    for f in glob.glob(f"repro/batchmulti_{cell}_s*{TAG[bank]}.json"):
        d = json.load(open(f))
        if prev in d["prevalence"]: v.append(d["prevalence"][prev][key])
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, None, 0)

# --- headline: 10 seeds x 9-detector bank, with a bootstrap-free normal CI ---
OUT["fig2b"] = {"cells": CELLS, "data": {}}
for c in CELLS:
    row = {}
    for p in ["0.5", "0.25"]:
        bf, gu, fb, orc = [], [], [], []
        for f in glob.glob(f"repro/batchmulti_{c}_s*{MAIN}.json"):
            d = json.load(open(f))
            if p not in d["prevalence"]: continue
            r = d["prevalence"][p]
            bf.append(r["best_fixed"]); gu.append(r["learned_guarded"])
            fb.append(r["dominated"]); orc.append(r["oracle"])
        if not bf: continue
        dl = np.array(gu) - np.array(bf); n = len(dl)
        hw = 1.96 * dl.std(ddof=1) / np.sqrt(n)
        row[p] = {"best_fixed": round(float(np.mean(bf)), 4),
                  "guarded": round(float(np.mean(gu)), 4),
                  "delta": round(float(dl.mean()), 4), "ci": round(float(hw), 4),
                  "oracle": round(float(np.mean(orc)), 4),
                  "fallback": round(float(np.mean(fb)), 3), "n_seeds": n}
    OUT["fig2b"]["data"][c] = row

OUT["fig2"] = {"cells": CELLS, "banks": ["minimol", "unimol", "both"], "data": {}}
for c in CELLS:
    OUT["fig2"]["data"][c] = {}
    for bank in ["minimol", "unimol", "both"]:
        row = {}
        for p in ["0.5", "0.25"]:
            bf, _, n = agg(c, bank, p, "best_fixed")
            gu, sd, _ = agg(c, bank, p, "learned_guarded")
            orc, _, _ = agg(c, bank, p, "oracle")
            fb, _, _ = agg(c, bank, p, "dominated")
            if bf is None: continue
            row[p] = {"best_fixed": round(bf, 4), "guarded": round(gu, 4),
                      "delta": round(gu - bf, 4), "sd": round(sd or 0, 4),
                      "oracle": round(orc, 4), "fallback": round(fb, 3), "n_seeds": n}
        OUT["fig2"]["data"][c][bank] = row

OUT["fig5"] = {}
for c in CELLS:
    fs = glob.glob(f"repro/batchmulti_{c}_s*{MAIN}.json")
    if not fs: continue
    dets = json.load(open(fs[0]))["detectors"]
    acc = {dn: [] for dn in dets}
    for f in fs:
        d = json.load(open(f))
        if "0.5" not in d["prevalence"]: continue
        for dn, v in d["prevalence"]["0.5"]["per_detector"].items():
            acc.setdefault(dn, []).append(v)
    OUT["fig5"][c] = {dn: round(float(np.mean(v)), 4) for dn, v in acc.items() if v}

# ---- fig 3 : prevalence sweep, parsed from the fine-grained logs ----
SC = "/tmp/claude-0/-root-autodl-tmp/042e0c6a-a135-4a5e-be7b-9d6cb3cf24a4/scratchpad"
prev_acc = {}
for f in glob.glob(f"{SC}/pv_*.log"):
    cell = prev = None
    for ln in open(f):
        m = re.search(r"\[(\w+)/s\d+\] prevalence (\d+)%", ln)
        if m: cell, prev = m.group(1), int(m.group(2)); continue
        m = re.search(r"best_fixed ([\d.]+) \| learned ([\d.]+) \(([+-][\d.]+)\)", ln)
        if m and cell: prev_acc.setdefault((cell, prev), []).append(float(m.group(3)))
OUT["fig3"] = {}
for (c, p), v in prev_acc.items():
    OUT["fig3"].setdefault(c, {})[p] = round(float(np.mean(v)), 4)

# ---- fig 4 : batch-size axis ----
OUT["fig4"] = {}
for c in ASSAY:
    OUT["fig4"][c] = {}
    for k in [4, 6, 8]:
        suf = "" if k == 8 else f"_nood{k}"
        for p in ["0.5", "0.25"]:
            v = []
            for f in glob.glob(f"repro/batchmulti_{c}_s*{suf}.json"):
                if any(t in f for t in ("_cv", "_unimol", "_both", "_smoke", "_regress")): continue
                d = json.load(open(f))
                if p in d["prevalence"]:
                    v.append(d["prevalence"][p].get("delta_learned_guarded",
                                                    d["prevalence"][p]["delta_learned"]))
            if v: OUT["fig4"][c].setdefault(p, {})[k] = round(float(np.mean(v)), 4)

json.dump(OUT, open("repro/figdata.json", "w"), indent=1)
print("wrote repro/figdata.json", flush=True)
for k in OUT: print(f"  {k}: {len(OUT[k])} entries", flush=True)
