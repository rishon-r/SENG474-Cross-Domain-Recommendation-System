"""Hyperparameter study (RQ2): n-fold cross-validation on the DEVELOPMENT users only.

    python -m src.tune --config configs/default.yaml

Development users = the training overlap users of make_split (the final test users are never
touched here). Each fold holds out a group of development users as validation users: their book
ratings are removed before the book embeddings are trained (so they are cold-start, exactly like
the final test users), and the mapping is trained on the remaining development users. The folds
are identical for every strategy and every configuration. Hyperparameters are varied one at a
time around the values in the config; the best configuration per strategy is the one with the
highest mean validation NDCG@n. Output (in evaluation results dir):
    tune_results.csv   every configuration, fold and strategy
    tune_summary.csv   mean/std over folds
    tuned.json         chosen embedding size and per-strategy mapping settings (read by src.main)
    charts/tune_*.png  NDCG@n against each hyperparameter
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from src.cmn.config import load_config
from src.cmn.seed import set_seed
from src.data.preprocess import config_hash, make_split, parse_and_cache
from src.embeddings.mf import train_mf
from src.evaluation.metrics import build_truth, evaluate_at_n
from src.evaluation.recommend import history_matrix, make_score_fn
from src.mappings.train import STRATEGIES

# which hyperparameters each strategy uses (embedding size "k" is shared)
USES = {"shared": ["k", "hidden", "lr"], "cluster": ["k", "hidden", "K", "lr"], "specific": ["k", "meta_hidden", "lr"]}


def log(msg):
    print(msg, flush=True)


def make_folds(train_users, n_folds, seed):
    """Disjoint, seed-dependent folds of the development users."""
    perm = np.random.default_rng(seed + 1000).permutation(train_users)
    return [np.sort(f) for f in np.array_split(perm, n_folds)]


def fold_embeddings(cfg, data, split, val_users, k, fold, seed, cache_dir, log=log):
    """Movie embeddings (all movie ratings; they do not depend on the fold, so cached per k) and
    book embeddings trained WITHOUT the validation users' book ratings (cached per k and fold)."""
    e = dict(cfg["embedding"], k=k)
    bu, bi, br = split["b_train"]
    keep = ~np.isin(bu, val_users)
    m_path = os.path.join(cache_dir, f"tune_movie_{config_hash(dict(data=data['cache_key'], embedding=e, seed=seed))}.npz")
    key = config_hash(dict(data=data["cache_key"], embedding=e, seed=seed, fold=fold, n_val=len(val_users)))
    b_path = os.path.join(cache_dir, f"tune_book_{key}.npz")
    if os.path.exists(m_path):
        U_m = np.load(m_path)["U_movie"]
    else:
        log(f"  training movie embeddings k={k}")
        U_m, _ = train_mf(data["m_user"], data["m_item"], data["m_rating"], int(data["n_users"]), int(data["n_movies"]),
                          k, e["lam"], e["lr"], e["batch_size"], e["epochs"], seed, log)
        np.savez(m_path, U_movie=U_m)
    if os.path.exists(b_path):
        z = np.load(b_path)
        U_b, V_b = z["U_book"], z["V_book"]
    else:
        log(f"  training book embeddings k={k} fold={fold}")
        U_b, V_b = train_mf(bu[keep], bi[keep], br[keep], int(data["n_users"]), int(data["n_books"]),
                            k, e["lam"], e["lr"], e["batch_size"], e["epochs"], seed + 1, log)
        np.savez(b_path, U_book=U_b, V_book=V_b)
    return U_m, U_b, V_b, (bu[keep], bi[keep], br[keep])


def configs(cfg):
    """Default configuration plus one-at-a-time variations from cfg['tune']['grid'].
    Returns (config, strategies that use the varied hyperparameter). The 'hidden' grid sets the
    hidden width of every strategy (MLP width for shared/cluster, meta-network width for specific)."""
    base = dict(k=cfg["embedding"]["k"], **{h: cfg["mapping"][h] for h in ("hidden", "meta_hidden", "K", "lr")})
    out = [(dict(base), list(STRATEGIES))]
    for name, values in cfg["tune"]["grid"].items():
        for v in values:
            if v == base[name]:
                continue
            c = dict(base, **{name: v})
            if name == "hidden":
                c["meta_hidden"] = v
            users = [st for st in STRATEGIES if name in USES[st] or (name == "hidden" and st == "specific")]
            out.append((c, users))
    return out


def run(cfg):
    d, s, ev, tn = cfg["data"], cfg["split"], cfg["eval"], cfg["tune"]
    res_dir, out_dir = cfg["output"]["results_dir"], cfg["output"]["dir"]
    os.makedirs(os.path.join(res_dir, "charts"), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    data = parse_and_cache(d["movies_path"], d["books_path"], d["preprocessed_dir"], d.get("user_frac"), False, log)
    seed = tn.get("seed", cfg["seeds"][0])
    set_seed(seed)
    split = make_split(data, seed, s["test_frac"], tuple(s["fractions"]))
    dev = split["train_users"]
    folds = make_folds(dev, tn["folds"], seed)
    n_users, n_books = int(data["n_users"]), int(data["n_books"])
    cfgs = configs(cfg)
    log(f"{len(dev)} development users, {tn['folds']} folds, {len(cfgs)} configurations, "
        f"{sum(len(x[1]) for x in cfgs)} strategy runs per fold")
    rows = []
    for fold, val in enumerate(folds):
        pool = np.setdiff1d(dev, val)            # users the mapping is trained on (100% of the rest)
        # truth for validation users = their book ratings from the development data
        bu, bi, br = split["b_train"]
        is_val = np.isin(bu, val)
        truth = build_truth((bu[is_val], bi[is_val], br[is_val]), val, n_books)
        for c, strategies in cfgs:
            t0 = time.time()
            U_m, U_b, V_b, (fu, fi, _) = fold_embeddings(cfg, data, split, val, c["k"], fold, seed, out_dir)
            hist = np.bincount(fu, minlength=n_users)
            history = history_matrix(fu, fi, n_users, n_books)
            assert (hist[val] < ev["T"]).all(), "validation users must be cold-start"
            for name in strategies:
                cls = STRATEGIES[name]
                mcfg = dict(cfg["mapping"], **{h: c[h] for h in ("hidden", "meta_hidden", "K", "lr")})
                mapper = cls(c["k"], mcfg).fit(U_m[pool], U_b[pool], seed)
                v, rc = evaluate_at_n(make_score_fn(mapper, U_m, U_b, V_b, hist, ev["T"], history), val, truth,
                                      ev["n"], ev["user_batch"])
                rows.append(dict(fold=fold, strategy=name, **c, ndcg=v, recall=rc))
                log(f"fold {fold} {name:8s} {c} NDCG@{ev['n']}={v:.4f}")
            log(f"  config done in {time.time() - t0:.0f}s")
    summarize(rows, cfg, res_dir)


def summarize(rows, cfg, res_dir):
    n = cfg["eval"]["n"]
    with open(os.path.join(res_dir, "tune_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    groups = {}
    for r in rows:
        key = (r["strategy"],) + tuple((h, r[h]) for h in USES[r["strategy"]])
        groups.setdefault(key, []).append(r)
    best, summary = {}, []
    for key, rs in groups.items():
        m = float(np.mean([r["ndcg"] for r in rs]))
        sd = float(np.std([r["ndcg"] for r in rs]))
        summary.append(dict(strategy=key[0], **dict(key[1:]), ndcg_mean=round(m, 8), ndcg_std=round(sd, 8),
                            recall_mean=round(float(np.mean([r["recall"] for r in rs])), 8), n_folds=len(rs)))
        if key[0] not in best or m > best[key[0]][0]:
            best[key[0]] = (m, dict(key[1:]))
    cols = ["strategy", "k", "hidden", "meta_hidden", "K", "lr", "ndcg_mean", "ndcg_std", "recall_mean", "n_folds"]
    with open(os.path.join(res_dir, "tune_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary:
            w.writerow({c: r.get(c, "") for c in cols})
    # one embedding size for all strategies: the k of the strategy-wise winners with the best mean
    k_best = max((v[1]["k"] for v in best.values()), key=lambda k: np.mean([v[0] for v in best.values() if v[1]["k"] == k]))
    tuned = {"k": int(k_best), "mapping": {}, "val_ndcg": {s: round(v[0], 8) for s, v in best.items()}}
    for s, (_, h) in best.items():
        tuned["mapping"][s] = dict(cfg["mapping"], **{x: h[x] for x in h if x != "k"})
    with open(os.path.join(res_dir, "tuned.json"), "w") as f:
        json.dump(tuned, f, indent=2)
    log(f"Best per strategy (validation NDCG@{n}): {tuned['val_ndcg']}; tuned.json written")
    plot(summary, cfg, res_dir)


def plot(summary, cfg, res_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    base = dict(k=cfg["embedding"]["k"], **{h: cfg["mapping"][h] for h in ("hidden", "meta_hidden", "K", "lr")})
    params = list(cfg["tune"]["grid"].keys())
    fig, axes = plt.subplots(1, len(params), figsize=(3.0 * len(params), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, params):
        for st in STRATEGIES:
            q = "meta_hidden" if (p == "hidden" and st == "specific") else p
            if q not in USES[st]:
                continue
            pts = []
            for r in summary:
                if r["strategy"] != st:
                    continue
                if all(r[h] == base[h] for h in USES[st] if h != q):   # others at default
                    pts.append((r[q], r["ndcg_mean"], r["ndcg_std"]))
            pts.sort()
            ax.errorbar([x[0] for x in pts], [x[1] for x in pts], yerr=[x[2] for x in pts], marker="o", capsize=2, label=st)
        ax.set_xlabel(p)
        if p == "lr":
            ax.set_xscale("log")
        ax.legend(fontsize=6)
    axes[0].set_ylabel(f"validation NDCG@{cfg['eval']['n']}")
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, "charts", "tune_hyperparameters.png"), dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    run(load_config(a.config))
