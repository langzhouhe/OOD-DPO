#!/usr/bin/env python3
"""Context-Matched RPO substrate -- FINAL frozen construction.

Architecture untouched: frozen encoder, the same scoring MLP, the same pairwise logistic
loss.  Only the pairing distribution changes.

Mechanism.  If a score decomposes as s(x) = a_g + r(x) for a context g (protein target,
potency regime, chemotype), within-context differences cancel a_g, so a pairwise loss never
has to fit it while a pointwise loss must absorb it.  This is the first construction here
that matches DPO's premise -- DPO compares chosen/rejected completions OF THE SAME PROMPT,
whereas random ID/OOD pairs share no context at all.

PROTOCOL REVISION LOG.  This is a revised pre-registration, not an unmodified one.  Every
decision below was taken before any AUROC was computed, and this is the LAST revision:
after the structural audit passes, splits, source weights, calipers, the pair ratio and the
control constructions are frozen.

 R1  Configuration B (protein-exact, same activity class, same 0.5 pChEMBL bin, Tanimoto
     >= 0.65, calipers |dHA|<=3 |drings|<=1 |dlogP|<=1.0 |dTPSA|<=25 |dMW|<=12%) replaced
     configuration A, whose descriptor audit showed mean |dHA| 3.2-6.7 and mean |dTPSA|
     17.8-43.8 -- the size/polarity axes this project's shortcut audit (report E18) showed
     drive these benchmarks.
 R2  Protein-family fallback dropped: under an identical block cap it LOWERS yield.
 R3  1-1 greedy matching replaced by degree-capped b-matching (max degree 3, coverage
     first): 1-1 gives a relational effective sample size of ~200 however many anchors are
     added.
 R4  SOURCE WEIGHTS ARE NATURAL FREQUENCY, w_e = 1/|E|, for every endpoint.  A 50/50
     re-weighting was rejected: on Ki it puts half the pairwise gradient on 13 edges
     (edge-ESS 49 of 214), and an endpoint-specific "switch to natural below 15%" rule
     would be a discontinuous objective chosen after seeing Ki.  The 50/50 variant is kept
     as an appendix sensitivity only.  Anchor pools remain exactly 1000 per source.
 R5  DECONTAMINATION IS SYMMETRIC AND LABEL-AGNOSTIC.  One identity blacklist of canonical
     SMILES and InChIKeys is built from EVERY target split with the split labels discarded,
     and molecules on it are removed from the external auxiliary pool before edges or
     anchors are built.  Correct wording for the paper:
       "Target test molecular identities were used solely for a prespecified,
        label-agnostic exact-compound decontamination step.  No target OOD labels, domains,
        embeddings, validation metrics or model scores were used for training or selection."
     Do NOT write "test data were never accessed".
 R6  Auxiliary assays are partitioned by a DETERMINISTIC FIVE-FOLD BALANCED split on
     degree-capped edge potential y_a = sum_{x in a} min(3, deg_legal(x)), greedily
     balancing potential, assay count and molecule count, ties broken by SHA256 (Python's
     `hash` is not stable across processes).  Fold 0 is aux-val, folds 1-4 aux-train.  No
     search over split seeds.  Train and validation are assay-disjoint AND molecule-disjoint.
 R7  The context-broken control is rebuilt as its own constrained matching rather than a
     permutation of the matched edges.  Permuting partners inside cost buckets does NOT
     preserve cost -- cost is a property of the pair -- and the measured result was
     Tanimoto 0.73-0.80 -> 0.13-0.34 with only 6-10% of calipers still satisfied, i.e. it
     destroyed chemistry as well as context.  The new control uses the SAME nodes, the
     EXACT same left and right degrees, the same source, activity class and pChEMBL bin,
     Tanimoto >= 0.65 and all configuration-B calipers, a DIFFERENT protein, and a
     min-cost matching whose per-node cost is |Tanimoto(candidate) - Tanimoto(matched)| so
     the similarity profile tracks the matched graph.  If exact degrees are infeasible,
     BOTH graphs are cut to the largest common feasible support; chemistry windows are
     never relaxed and the molecule set is never changed.

TEST SETS.  A fixed random subsample of N_TEST molecules per side is drawn once with seed
42 from each official test split, because encoding every official test molecule with
Uni-Mol is not affordable here (ic50 alone is 194k molecules).  The paper must say
"a fixed random subsample of the official test split", not "the official test split".

TRAINING DISTRIBUTION (frozen)
  P_CM(i,j) = 0.75 * w_ij + 0.25/(N_I*N_O) with w_e = 1/|E| (natural)
  mu_i = 0.75*sum_{e~i} w_e + 0.25/N_I ,  nu_j = 0.75*sum_{e~j} w_e + 0.25/N_O

Usage: python cmrpo_data.py --target ec50
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json, zipfile, io, argparse, warnings, logging, random, hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, inchi
RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)

ZIP = Path("data/archives/drugood_all.zip")
RAW = Path("data/raw")
OUT = Path("cache/cmrpo"); OUT.mkdir(parents=True, exist_ok=True)
ENDPOINTS = ["ec50", "ic50", "ki"]
SEED = 42

N_ID_ANCHOR, N_OOD_PER_SOURCE = 1500, 1000
N_VAL_ID, N_VAL_OOD = 600, 600
N_TEST = 5000
NFOLD = 5
REG_BIN = 0.5
MAX_DEGREE = 3
PROT_CAP, BLOCK_CAP = 1200, 600
TANIMOTO_MIN = 0.65
CAL = dict(ha=3, rings=1, logp=1.0, tpsa=25.0, mw_frac=0.12)


def sha(s):
    return hashlib.sha256(str(s).encode()).hexdigest()


def protein_maps():
    z = zipfile.ZipFile(ZIP)
    a2p = {}
    for t in ENDPOINTS:
        d = json.load(io.TextIOWrapper(z.open(f"drugood_all/sbap_general_{t}_assay.json")))
        m = {}
        for items in d["split"].values():
            for it in items:
                a, p = it.get("assay_id"), it.get("protein")
                if a is not None and p: m.setdefault(str(a), p)
        a2p[t] = m
    return a2p


def featurise(smis):
    out = {}
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        try: key = inchi.MolToInchiKey(m)
        except Exception: key = None
        out[s] = {"can": Chem.MolToSmiles(m), "ik": key,
                  "fp": AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048),
                  "ha": Descriptors.HeavyAtomCount(m),
                  "rings": rdMolDescriptors.CalcNumRings(m),
                  "mw": Descriptors.MolWt(m), "logp": Descriptors.MolLogP(m),
                  "tpsa": Descriptors.TPSA(m)}
    return out


def calipers_ok(a, b):
    return (abs(a["ha"] - b["ha"]) <= CAL["ha"]
            and abs(a["rings"] - b["rings"]) <= CAL["rings"]
            and abs(a["logp"] - b["logp"]) <= CAL["logp"]
            and abs(a["tpsa"] - b["tpsa"]) <= CAL["tpsa"]
            and abs(a["mw"] - b["mw"]) <= CAL["mw_frac"] * max(a["mw"], b["mw"]))


def deltas(a, b):
    return {"d_ha": abs(a["ha"] - b["ha"]), "d_rings": abs(a["rings"] - b["rings"]),
            "d_mw_frac": abs(a["mw"] - b["mw"]) / max(a["mw"], b["mw"]),
            "d_logp": abs(a["logp"] - b["logp"]), "d_tpsa": abs(a["tpsa"] - b["tpsa"])}


def dist_stats(v):
    v = list(v)
    if not v: return None
    return {"mean": float(np.mean(v)), "p10": float(np.quantile(v, .1)),
            "p50": float(np.quantile(v, .5)), "p90": float(np.quantile(v, .9))}


def degree_hist(pairs, side):
    d = defaultdict(int)
    for e in pairs: d[e[side]] += 1
    if not d: return {}
    v, c = np.unique(list(d.values()), return_counts=True)
    return {str(int(x)): int(y) for x, y in zip(v, c)}


def five_fold(stats):
    """Deterministic balanced assignment of assays to NFOLD folds.  Descending edge
    potential, each assay to the currently lightest fold, SHA256 tie-break."""
    order = sorted(stats, key=lambda a: (-stats[a][0], -stats[a][1], sha(a)))
    folds = [{"pot": 0.0, "na": 0, "nm": 0, "assays": []} for _ in range(NFOLD)]
    for a in order:
        pot, nm = stats[a]
        k = min(range(NFOLD),
                key=lambda f: (folds[f]["pot"], folds[f]["na"], folds[f]["nm"], f))
        folds[k]["pot"] += pot; folds[k]["na"] += 1; folds[k]["nm"] += nm
        folds[k]["assays"].append(a)
    return folds


def b_match(edges, max_deg=MAX_DEGREE):
    edges = sorted(edges, key=lambda e: (e["cost"], e["i_smiles"], e["o_smiles"]))
    di, do, chosen, taken = defaultdict(int), defaultdict(int), [], set()
    def add(k, e):
        chosen.append(e); di[e["i_smiles"]] += 1; do[e["o_smiles"]] += 1; taken.add(k)
    for k, e in enumerate(edges):
        if di[e["i_smiles"]] == 0 and do[e["o_smiles"]] == 0: add(k, e)
    for k, e in enumerate(edges):
        if k in taken: continue
        if (di[e["i_smiles"]] == 0) != (do[e["o_smiles"]] == 0):
            if di[e["i_smiles"]] < max_deg and do[e["o_smiles"]] < max_deg: add(k, e)
    for k, e in enumerate(edges):
        if k in taken: continue
        if di[e["i_smiles"]] < max_deg and do[e["o_smiles"]] < max_deg: add(k, e)
    return chosen


def broken_graph(cm, F, ctx_of):
    """Exact-degree, chemistry-preserving, protein-breaking control (R7).

    Each matched edge becomes one slot on the ID side and one on the auxiliary side, so a
    perfect slot matching reproduces both degree sequences exactly.  A slot pair is feasible
    only if the two molecules share activity class and pChEMBL bin, sit on DIFFERENT
    proteins, reach Tanimoto >= 0.65 and pass every configuration-B caliper.  Cost is
    |Tanimoto(candidate) - Tanimoto(the matched edge at that slot)| so the similarity
    profile follows the matched graph node by node.  Maximum flow is solved first, so if no
    perfect matching exists the largest common feasible support is returned and the caller
    cuts BOTH graphs to it."""
    import networkx as nx
    n = len(cm)
    ladder = {"cross_protein_same_ctx": 0, "plus_tanimoto": 0, "plus_calipers": 0}
    G = nx.DiGraph()
    G.add_node("s", demand=-n); G.add_node("t", demand=n)
    for k in range(n):
        G.add_edge("s", f"L{k}", capacity=1, weight=0)
        G.add_edge(f"R{k}", "t", capacity=1, weight=0)
    for k, ek in enumerate(cm):
        fa = F[ek["i_smiles"]]
        for m, em in enumerate(cm):
            fb = F[em["o_smiles"]]
            if ek["cls"] != em["cls"] or ek["bin"] != em["bin"]: continue
            if ek["protein"] == em["protein"]: continue
            ladder["cross_protein_same_ctx"] += 1
            sim = DataStructs.TanimotoSimilarity(fa["fp"], fb["fp"])
            if sim < TANIMOTO_MIN: continue
            ladder["plus_tanimoto"] += 1
            if not calipers_ok(fa, fb): continue
            ladder["plus_calipers"] += 1
            G.add_edge(f"L{k}", f"R{m}", capacity=1,
                       weight=int(round(1000 * abs(sim - ek["tanimoto"]))))
    if ladder["plus_calipers"] == 0: return [], ladder, 0
    flow = nx.max_flow_min_cost(G, "s", "t")
    out = []
    for k in range(n):
        for m, f in flow.get(f"L{k}", {}).items():
            if f > 0 and m.startswith("R"):
                j = int(m[1:])
                fa, fb = F[cm[k]["i_smiles"]], F[cm[j]["o_smiles"]]
                sim = float(DataStructs.TanimotoSimilarity(fa["fp"], fb["fp"]))
                out.append({"slot": k, "i_smiles": cm[k]["i_smiles"],
                            "o_smiles": cm[j]["o_smiles"], "src": cm[j]["src"],
                            "tanimoto": sim, "cost": 1.0 - sim,
                            "cross_protein": cm[k]["protein"] != cm[j]["protein"],
                            "calipers_ok": calipers_ok(fa, fb),
                            "cls": cm[k]["cls"], "bin": cm[k]["bin"],
                            "protein": cm[j]["protein"], **deltas(fa, fb)})
    return out, ladder, len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, choices=ENDPOINTS)
    a = ap.parse_args()
    T = a.target
    others = [t for t in ENDPOINTS if t != T]
    rng = random.Random(SEED)
    a2p = protein_maps()
    L = {t: json.load(open(RAW / f"lbap_general_{t}_assay.json"))["split"] for t in ENDPOINTS}
    tgt_ids = {x["smiles"] for p in L[T] for x in L[T][p]}

    id_prot = defaultdict(list)
    for x in L[T]["train"]:
        p = a2p[T].get(str(x["assay_id"]))
        if p: id_prot[p].append(x)
    aux_prot = defaultdict(list)
    for t in others:
        for x in L[t]["train"]:
            p = a2p[t].get(str(x["assay_id"]))
            if p and p in id_prot: aux_prot[p].append({**x, "src": t})
    for d in (id_prot, aux_prot):
        for p in list(d): rng.shuffle(d[p]); d[p] = d[p][:PROT_CAP]

    need = sorted({x["smiles"] for v in id_prot.values() for x in v}
                  | {x["smiles"] for v in aux_prot.values() for x in v} | tgt_ids)
    print(f"[{T}] featurising {len(need)} molecules ...", flush=True)
    F = featurise(need)
    BLACK = {F[s]["can"] for s in tgt_ids if s in F}
    BLACK_IK = {F[s]["ik"] for s in tgt_ids if s in F and F[s]["ik"]}
    def bad(s):
        f = F.get(s)
        return (f is None) or (f["can"] in BLACK) or (f["ik"] and f["ik"] in BLACK_IK)

    n0 = sum(len(v) for v in aux_prot.values())
    for p in list(aux_prot): aux_prot[p] = [x for x in aux_prot[p] if not bad(x["smiles"])]
    n1 = sum(len(v) for v in aux_prot.values())
    print(f"[{T}] decontamination removed {n0-n1} of {n0} auxiliary candidates", flush=True)

    # ---- all legal edges, over ALL auxiliary assays (needed for edge potential) -------
    legal, n_cross = [], 0
    for p in sorted(id_prot):
        blocks = defaultdict(lambda: ([], []))
        for x in id_prot[p]:
            f = F.get(x["smiles"])
            if f: blocks[(int(x["cls_label"]), int(float(x["reg_label"]) / REG_BIN))][0].append((x, f))
        for x in aux_prot.get(p, []):
            f = F.get(x["smiles"])
            if f: blocks[(int(x["cls_label"]), int(float(x["reg_label"]) / REG_BIN))][1].append((x, f))
        for k, (A, B) in blocks.items():
            A, B = A[:BLOCK_CAP], B[:BLOCK_CAP]
            if not A or not B: continue
            n_cross += len(A) * len(B)
            for xa, fa in A:
                sims = DataStructs.BulkTanimotoSimilarity(fa["fp"], [fb["fp"] for _, fb in B])
                for (xb, fb), sim in zip(B, sims):
                    if sim >= TANIMOTO_MIN and calipers_ok(fa, fb):
                        legal.append({"i_smiles": xa["smiles"], "o_smiles": xb["smiles"],
                                      "protein": p, "src": xb["src"], "tanimoto": float(sim),
                                      "cost": 1.0 - float(sim), "cls": int(k[0]),
                                      "bin": int(k[1]), "o_assay": str(xb["assay_id"])})
    print(f"[{T}] cross candidates {n_cross:,} -> legal edges {len(legal):,}", flush=True)

    # ---- R6: deterministic five-fold balanced assay split on edge potential -----------
    deg_legal = defaultdict(int)
    for e in legal: deg_legal[e["o_smiles"]] += 1
    assay_src, assay_mols = {}, defaultdict(set)
    for t in others:
        for x in L[t]["train"]:
            assay_src[str(x["assay_id"])] = t
            assay_mols[str(x["assay_id"])].add(x["smiles"])
    folds_by_src = {}
    for t in others:
        st = {a: (float(sum(min(MAX_DEGREE, deg_legal.get(s, 0)) for s in ms)), len(ms))
              for a, ms in assay_mols.items() if assay_src[a] == t}
        folds_by_src[t] = five_fold(st)
    val_assays, tr_assays = set(), set()
    for t in others:
        val_assays |= set(folds_by_src[t][0]["assays"])
        for f in folds_by_src[t][1:]: tr_assays |= set(f["assays"])

    edges = b_match([e for e in legal if e["o_assay"] in tr_assays])
    if not edges:
        print(f"[{T}] NO EDGES", flush=True); return
    ctx_of = {}
    for e in edges: ctx_of.setdefault(e["i_smiles"], []).append((e["cls"], e["bin"]))

    print(f"[{T}] b-matching: {len(edges)} edges; building the exact-degree broken control ...",
          flush=True)
    broken, ladder, n_broken = broken_graph(edges, F, ctx_of)
    n_cm0 = len(edges)
    if n_broken == n_cm0:
        broken_status = "exact_degree"
    elif n_broken >= 0.5 * n_cm0:
        keep = {b["slot"] for b in broken}          # cut BOTH to the common support
        edges = [e for k, e in enumerate(edges) if k in keep]
        broken_status = f"common_support_{n_broken}_of_{n_cm0}"
    else:
        # R7 fallback: the CM graph is NEVER destroyed.  The broken control is demoted to
        # exploratory and marginal-product RPO becomes the primary coupling control.
        broken_status = f"EXPLORATORY_only_{n_broken}_of_{n_cm0}_feasible"
    print(f"[{T}] broken control: {n_broken}/{n_cm0} slots feasible -> {broken_status}; "
          f"cross-protein ladder {ladder}", flush=True)

    di = defaultdict(int); do = defaultdict(int)
    for e in edges: di[e["i_smiles"]] += 1; do[e["o_smiles"]] += 1
    id_nodes, ood_nodes = sorted(di), sorted(do)
    n_src = {t: sum(1 for e in edges if e["src"] == t) for t in others}
    print(f"[{T}] final: {len(edges)} edges, {len(id_nodes)} ID + {len(ood_nodes)} aux "
          f"molecules, source counts {n_src}", flush=True)

    # ---- anchors: edge nodes first, then exactly 1000 auxiliary per source ------------
    id_pool = list(id_nodes)
    rest = [x["smiles"] for x in L[T]["train"] if x["smiles"] not in di]
    rng.shuffle(rest); id_pool += rest[:max(0, N_ID_ANCHOR - len(id_pool))]

    node_src = {e["o_smiles"]: e["src"] for e in edges}
    node_assay = {e["o_smiles"]: e["o_assay"] for e in edges}
    ood_pool, ood_src, ood_assay, seen = [], [], [], set()
    for t in others:
        take = [s for s in ood_nodes if node_src[s] == t]
        for s in take: seen.add(F[s]["can"])
        cand = [x for x in L[t]["train"] if str(x["assay_id"]) in tr_assays
                and x["smiles"] not in do]
        rng.shuffle(cand)
        for x in cand:
            if len(take) >= N_OOD_PER_SOURCE: break
            s = x["smiles"]
            if bad(s) or F[s]["can"] in seen: continue
            seen.add(F[s]["can"]); take.append(s)
            node_src[s] = t; node_assay[s] = str(x["assay_id"])
        take = take[:N_OOD_PER_SOURCE]
        ood_pool += take; ood_src += [t] * len(take)
        ood_assay += [node_assay[s] for s in take]

    # ---- auxiliary validation: assay-disjoint AND molecule-disjoint -------------------
    train_can = {F[s]["can"] for s in ood_pool}
    aux_val, aux_val_src, seenv = [], [], set()
    for t in others:
        cand = [x for x in L[t]["train"] if str(x["assay_id"]) in val_assays]
        rng.shuffle(cand); k = 0
        for x in cand:
            if k >= N_VAL_OOD // 2: break
            s = x["smiles"]
            if bad(s): continue
            c = F.get(s, {}).get("can")
            if c is None or c in train_can or c in seenv: continue
            seenv.add(c); aux_val.append(s); aux_val_src.append(t); k += 1

    ipos = {s: k for k, s in enumerate(id_pool)}
    opos = {s: k for k, s in enumerate(ood_pool)}
    prot_ix = {p: k for k, p in enumerate(sorted({e["protein"] for e in edges}))}
    src_ix = {t: k for k, t in enumerate(others)}

    def pack(es):
        out = []
        for e in es:
            if e["i_smiles"] not in ipos or e["o_smiles"] not in opos: continue
            out.append([ipos[e["i_smiles"]], opos[e["o_smiles"]], src_ix[e["src"]],
                        prot_ix.get(e["protein"], -1), int(e["cls"]), int(e["bin"]),
                        float(e["tanimoto"])])
        return out
    E, EB = pack(edges), pack(broken)

    lab = {}
    for x in L[T]["train"]:
        if x.get("cls_label") is not None: lab.setdefault(x["smiles"], int(x["cls_label"]))

    def take_n(items, n, off):
        r = np.random.default_rng(SEED + off)
        idx = r.choice(len(items), size=min(n, len(items)), replace=False)
        return [items[i]["smiles"] for i in idx]

    def canon(ss):
        out = set()
        for s in ss:
            f = F.get(s)
            if f: out.add(f["can"])
            else:
                m = Chem.MolFromSmiles(s)
                if m is not None: out.add(Chem.MolToSmiles(m))
        return out
    aux_can = canon(ood_pool) | canon(aux_val)
    tgt_by_split = {p: canon({x["smiles"] for x in L[T][p]}) for p in L[T]}
    n_i, n_o, m = len(id_pool), len(ood_pool), len(E)
    mu = np.zeros(n_i); nu = np.zeros(n_o)
    for i, j, *_ in E: mu[i] += 1.0 / m; nu[j] += 1.0 / m
    mu = 0.75 * mu + 0.25 / n_i; nu = 0.75 * nu + 0.25 / n_o

    rec = {
        "target": T, "sources": others, "seed": SEED, "config": "B",
        "id_train": id_pool, "ood_train": ood_pool,
        "ood_src": ood_src, "ood_assay": ood_assay,
        "id_labels": [lab.get(s, -1) for s in id_pool],
        "edges": E, "edges_broken": EB,
        "edge_fields": ["i", "j", "src_ix", "protein_ix", "cls", "bin", "tanimoto"],
        "mu": mu.tolist(), "nu": nu.tolist(),
        "val_id": take_n(L[T]["iid_val"], N_VAL_ID, 2), "val_ood": aux_val[:N_VAL_OOD],
        "val_ood_src": aux_val_src[:N_VAL_OOD],
        "test_id": take_n(L[T]["iid_test"], N_TEST, 4),
        "test_ood": take_n(L[T]["ood_test"], N_TEST, 5),
        "audit": {
            "cross_candidates": int(n_cross), "legal_edges": len(legal),
            "decontaminated_aux": int(n0 - n1), "aux_candidates": int(n1),
            "edges": len(E), "edges_broken": len(EB),
            "broken_status": broken_status,
            "broken_cross_protein_ladder": ladder,
            "broken_exact_degree": bool(len(EB) == len(E)),
            "broken_cross_protein_frac": float(np.mean([b["cross_protein"] for b in broken]))
                                          if broken else None,
            "broken_calipers_ok_frac": float(np.mean([b["calipers_ok"] for b in broken]))
                                        if broken else None,
            "id_nodes": len(id_nodes), "ood_nodes": len(ood_nodes),
            "n_proteins": len({e["protein"] for e in edges}),
            "edge_counts_by_source": n_src,
            "loss_mass_by_source": {t: float(n_src[t] / max(len(edges), 1)) for t in others},
            "degree_id": degree_hist(E, 0), "degree_ood": degree_hist(E, 1),
            "degree_id_broken": degree_hist(EB, 0), "degree_ood_broken": degree_hist(EB, 1),
            "id_pool": n_i, "ood_pool": n_o,
            "ood_pool_by_source": {t: int(sum(1 for s in ood_src if s == t)) for t in others},
            "id_labelled": int(sum(1 for s in id_pool if lab.get(s, -1) >= 0)),
            "id_label_classes": sorted({lab[s] for s in id_pool if s in lab}),
            "assay_overlap_train_val": len(tr_assays & val_assays),
            "anchor_assay_in_val": int(sum(1 for a in ood_assay if a in val_assays)),
            "aux_val_molecule_overlap": len(canon(ood_pool) & canon(aux_val)),
            "aux_vs_target_identity_overlap": {p: len(aux_can & v) for p, v in tgt_by_split.items()},
            "fold_balance": {t: [{"pot": round(f["pot"], 1), "assays": f["na"],
                                  "mols": f["nm"]} for f in folds_by_src[t]] for t in others},
            "matched_tanimoto": dist_stats([e["tanimoto"] for e in edges]),
            "broken_tanimoto": dist_stats([b["tanimoto"] for b in broken]),
            "matched_deltas": {k: dist_stats([deltas(F[e["i_smiles"]], F[e["o_smiles"]])[k]
                                              for e in edges])
                               for k in ("d_ha", "d_rings", "d_mw_frac", "d_logp", "d_tpsa")},
            "broken_deltas": {k: dist_stats([b[k] for b in broken])
                              for k in ("d_ha", "d_rings", "d_mw_frac", "d_logp", "d_tpsa")},
            "test_subsample": {"per_side": N_TEST,
                               "note": "fixed random subsample of the official test split"},
            "source_weighting": "natural frequency w_e = 1/|E| (primary); 50/50 appendix only",
        },
    }
    json.dump(rec, open(OUT / f"{T}_cmrpo.json", "w"))
    print(f"[{T}] audit {json.dumps(rec['audit'])}", flush=True)
    print(f"CMRPO_DATA_DONE {T}", flush=True)


if __name__ == "__main__":
    main()
