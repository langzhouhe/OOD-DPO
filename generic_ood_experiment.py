#!/usr/bin/env python3
"""
Direction (3): does the detector need the benchmark's CURATED OOD, or can a
GENERIC background (random ZINC molecules) replace it? If generic-OOD training
approx matches curated-OOD training on the real test OOD, the method escapes
the "you peeked at the benchmark's OOD type" confound (reviewer C1) and
validates the paper's own claim that Dout can come from public libraries.

Protocol: train RPO head on <target ID> vs <train_ood source>, best-val
checkpoint, test on the target benchmark's real (test_id, test_ood).
train_ood source in {curated (benchmark), generic (random ZINC), mix}.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
import json, pickle, warnings, logging
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
from pathlib import Path
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

CACHE = Path("cache/ood_dpo_cache")
FEAT = {
    "ec50_assay":    "lbap_general_ec50_assay_minimol_features.pkl",
    "ec50_scaffold": "lbap_general_ec50_scaffold_minimol_features.pkl",
    "ic50_assay":    "lbap_general_ic50_assay_minimol_features.pkl",
    "hiv_scaffold":  "good_hiv_scaffold_covariate_minimol_features.pkl",
}
SPL = {
    "ec50_assay":    "lbap_general_ec50_assay_seed42_splits.json",
    "ec50_scaffold": "lbap_general_ec50_scaffold_seed42_splits.json",
    "ic50_assay":    "lbap_general_ic50_assay_seed42_splits.json",
    "hiv_scaffold":  "good_hiv_scaffold_covariate_seed42_splits.json",
}
# generic background: random ZINC molecules (a public library — NOT the benchmark's OOD)
ZINC_FEAT = "good_zinc_size_covariate_minimol_features.pkl"
ZINC_SPL = "good_zinc_size_covariate_seed42_splits.json"


class Head(nn.Module):
    def __init__(s, d=512, h=256):
        super().__init__(); s.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h // 2, 1))
    def forward(s, x): return s.net(x).squeeze(-1)


def feats_of(cell):
    f = pickle.load(open(CACHE / FEAT[cell], "rb"))["features"]
    sp = json.load(open(CACHE / SPL[cell], "rb"))["splits"]
    return f, sp


def mat(f, smis): return np.stack([f[s] for s in smis if s in f]).astype(np.float32)


def auroc(a, b): return roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b])


def train_eval(Xid, Xood, val_id, val_ood, test_id, test_ood, seed=1, epochs=300):
    torch.manual_seed(seed); np.random.seed(seed)
    Xid, Xood = torch.tensor(Xid), torch.tensor(Xood)
    vid, vood = torch.tensor(val_id), torch.tensor(val_ood)
    head = Head(); opt = torch.optim.AdamW(head.parameters(), 1e-4, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.StepLR(opt, 10, 0.9); best, bs = -1, None
    for ep in range(epochs):
        head.train(); opt.zero_grad(); Eid, Eood = head(Xid), head(Xood)
        loss = F.softplus(-0.1 * (Eood.unsqueeze(0) - Eid.unsqueeze(1))).mean() + 0.01 * (Eid.pow(2).mean() + Eood.pow(2).mean())
        loss.backward(); nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sch.step()
        if (ep + 1) % 15 == 0:
            head.eval()
            with torch.no_grad(): v = auroc(head(vid).numpy(), head(vood).numpy())
            if v > best: best, bs = v, {k: t.clone() for k, t in head.state_dict().items()}
    head.load_state_dict(bs); head.eval()
    with torch.no_grad(): return auroc(head(torch.tensor(test_id)).numpy(), head(torch.tensor(test_ood)).numpy())


zf = pickle.load(open(CACHE / ZINC_FEAT, "rb"))["features"]
zsp = json.load(open(CACHE / ZINC_SPL, "rb"))["splits"]
zinc_all = mat(zf, zsp["train_id"] + zsp["train_ood"])            # generic background pool
rng = np.random.default_rng(0)

print(f"{'cell':14}{'curated':>9}{'generic':>9}{'mix':>9}   (test on real benchmark OOD)")
print("-" * 56)
res = {}
for cell in FEAT:
    f, sp = feats_of(cell)
    Xid = mat(f, sp["train_id"])[:2000]
    cur_ood = mat(f, sp["train_ood"])[:2000]
    gen_ood = zinc_all[rng.choice(len(zinc_all), min(2000, len(zinc_all)), replace=False)]
    vid = mat(f, sp["val_id"]); vood = mat(f, sp["val_ood"])
    vgen = zinc_all[rng.choice(len(zinc_all), min(600, len(zinc_all)), replace=False)]
    tid = mat(f, sp["test_id"]); tood = mat(f, sp["test_ood"])
    a_cur = train_eval(Xid, cur_ood, vid, vood, tid, tood)
    a_gen = train_eval(Xid, gen_ood, vid, vgen, tid, tood)          # val OOD is generic too (no benchmark OOD peeked)
    mix = np.concatenate([cur_ood[:1000], gen_ood[:1000]])
    a_mix = train_eval(Xid, mix, vid, vood, tid, tood)
    res[cell] = {"curated": a_cur, "generic": a_gen, "mix": a_mix}
    print(f"{cell:14}{a_cur:>9.3f}{a_gen:>9.3f}{a_mix:>9.3f}", flush=True)
json.dump(res, open("repro/generic_ood_results.json", "w"), indent=2)
print("GENERIC_DONE")
