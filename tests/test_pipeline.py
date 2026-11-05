"""Smoke / correctness tests. Run with:  python -m pytest -q"""
import copy
import gzip
import json
import os

import numpy as np
import torch

from src.cmn import synthetic
from src.cmn.config import load_config
from src.data.preprocess import keep_user, make_split, parse_and_cache, read_reviews
from src.embeddings.mf import train_mf
from src.evaluation.metrics import build_truth, ndcg_at_n
from src.main import run


def _data(tmp_path):
    raw = tmp_path / "raw"
    synthetic.make(str(raw), n_users=400, n_items=120, per_user=15, seed=1)
    return parse_and_cache(str(raw / "Movies_and_TV_5.json.gz"), str(raw / "Books_5.json.gz"), str(tmp_path / "pre"))


def test_split_has_no_leakage(tmp_path):
    data = _data(tmp_path)
    sp = make_split(data, seed=0)
    assert not np.isin(sp["b_train"][0], sp["test_users"]).any()          # no test-user book rating in training
    assert np.isin(sp["truth"][0], sp["test_users"]).all()                # ground truth only for test users
    assert not np.isin(sp["subsets"][1.0], sp["test_users"]).any()        # O_test never used for mappings
    assert len(sp["test_users"]) == round(0.2 * len(sp["overlap"]))
    assert [len(sp["subsets"][f]) for f in (0.1, 0.2, 0.5, 1.0)] == \
           [int(np.floor(f * len(sp["train_users"]))) for f in (0.1, 0.2, 0.5, 1.0)]


def test_mf_reduces_error(tmp_path):
    data = _data(tmp_path)
    P, Q = train_mf(data["m_user"], data["m_item"], data["m_rating"], int(data["n_users"]), int(data["n_movies"]),
                    k=8, lam=0.02, lr=20.0, batch_size=256, epochs=15, seed=0)
    pred = (P[data["m_user"]] * Q[data["m_item"]]).sum(1)
    rmse = np.sqrt(np.mean((pred - data["m_rating"]) ** 2))
    assert rmse < np.std(data["m_rating"])          # better than predicting the mean


def test_ndcg_known_value():
    # one test user, 4 books, held-out ratings: book0=5, book2=3
    truth = build_truth((np.array([7, 7]), np.array([0, 2]), np.array([5.0, 3.0])), np.array([7]), 4)
    perfect = lambda u: torch.tensor([[9.0, 1.0, 5.0, 0.0]])        # ranks book0, book2 first
    assert abs(ndcg_at_n(perfect, np.array([7]), truth, 2) - 1.0) < 1e-6
    swapped = lambda u: torch.tensor([[5.0, 1.0, 9.0, 0.0]])        # book2 first, then book0
    expected = (3 / np.log2(2) + 5 / np.log2(3)) / (5 / np.log2(2) + 3 / np.log2(3))
    assert abs(ndcg_at_n(swapped, np.array([7]), truth, 2) - expected) < 1e-6


def test_same_seed_same_results(tmp_path):
    raw = tmp_path / "raw"
    synthetic.make(str(raw), n_users=400, n_items=120, per_user=15, seed=1)
    cfg = load_config("configs/quickstart.yaml")
    cfg["data"].update(movies_path=str(raw / "Movies_and_TV_5.json.gz"), books_path=str(raw / "Books_5.json.gz"),
                       preprocessed_dir=str(tmp_path / "pre"))
    cfg["embedding"].update(epochs=3)
    cfg["mapping"].update(epochs=3)
    cfg["split"]["fractions"] = [0.5, 1.0]
    outs = []
    for i in range(2):
        c = copy.deepcopy(cfg)
        c["output"] = dict(dir=str(tmp_path / f"o{i}"), results_dir=str(tmp_path / f"r{i}"),
                           ground_truth_dir=str(tmp_path / f"g{i}"))
        outs.append([r["ndcg"] for r in run(c, force=True)])
    assert outs[0] == outs[1]


def _write(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for u, i, r, t in rows:
            f.write(json.dumps({"reviewerID": u, "asin": i, "overall": r, "unixReviewTime": t}) + "\n")


def test_parser_keeps_latest_duplicate(tmp_path):
    p = tmp_path / "m.json.gz"
    _write(p, [("A", "x", 1.0, 10), ("B", "y", 4.0, 5), ("A", "x", 5.0, 20), ("A", "z", 3.0, 1), ("A", "x", 2.0, 15)])
    users = {}
    u, i, r, n_items = read_reviews(str(p), users)
    assert users == {"A": 0, "B": 1} and n_items == 3
    # one row per (user, item), in first-appearance order; A-x keeps the latest review (t=20, rating 5)
    assert u.tolist() == [0, 1, 0] and i.tolist() == [0, 1, 2] and r.tolist() == [5.0, 4.0, 3.0]


def test_user_sample_is_consistent_across_files(tmp_path):
    raw = tmp_path / "raw"
    synthetic.make(str(raw), n_users=400, n_items=120, per_user=15, seed=1)
    m, b = str(raw / "Movies_and_TV_5.json.gz"), str(raw / "Books_5.json.gz")
    full = parse_and_cache(m, b, str(tmp_path / "pre"))
    half = parse_and_cache(m, b, str(tmp_path / "pre"), user_frac=0.5)
    assert half["cache_key"] != full["cache_key"]                     # separate cache files
    assert 0 < len(half["m_rating"]) < len(full["m_rating"])
    n_overlap = lambda d: len(np.intersect1d(d["m_user"], d["b_user"]))
    ratio = n_overlap(half) / n_overlap(full)
    assert 0.35 < ratio < 0.65                                         # overlap users survive sampling
    assert keep_user("A123", 0.5) == keep_user("A123", 0.5)           # deterministic


def test_cache_is_not_reused_after_settings_change(tmp_path):
    raw = tmp_path / "raw"
    synthetic.make(str(raw), n_users=400, n_items=120, per_user=15, seed=1)
    m, b, pre = str(raw / "Movies_and_TV_5.json.gz"), str(raw / "Books_5.json.gz"), str(tmp_path / "pre")
    parse_and_cache(m, b, pre, user_frac=0.3)
    full = parse_and_cache(m, b, pre)                                  # must NOT load the 30% cache
    assert int(full["n_users"]) > 300
    assert len([f for f in os.listdir(pre) if f.startswith("ratings_")]) == 2
    synthetic.make(str(raw), n_users=200, n_items=120, per_user=15, seed=2)   # "re-downloaded" file
    os.utime(m, ns=(1, 1)); os.utime(b, ns=(1, 1))
    assert int(parse_and_cache(m, b, pre)["n_users"]) <= 200         # re-parsed, not the old cache


def test_recommend_excludes_history():
    from src.evaluation.recommend import history_matrix, recommend

    class Identity:
        def map(self, X):
            return X
    V = np.eye(3, dtype=np.float32)                       # book b scores e[b]
    U = np.array([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]], dtype=np.float32)
    hist_counts = np.array([1, 0])                         # user 0 has book 0 in H_b(u); user 1 is cold-start
    H = history_matrix(np.array([0]), np.array([0]), 2, 3)
    assert recommend(0, Identity(), U, U, V, hist_counts, 1, 2, H).tolist() == [1, 2]   # book 0 excluded
    assert recommend(1, Identity(), U, U, V, hist_counts, 1, 2, H).tolist() == [0, 1]   # nothing to exclude
