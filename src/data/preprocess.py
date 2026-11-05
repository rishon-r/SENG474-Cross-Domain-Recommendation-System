"""Data layer: parse the Amazon 2018 5-core files and build the seed-dependent split.

parse_and_cache()  -> seed-independent: ratings of both domains with one shared user index.
make_split()       -> seed-dependent: overlap set O, held-out O_test (20% of O, book ratings
                      removed BEFORE any training), O_train and the overlap subsets O_rho.
"""
import gzip
import hashlib
import json
import os
import zlib
from array import array

import numpy as np

# Bump when the parsed format or parsing rules change, so old caches are not reused.
PARSE_VERSION = 2


def config_hash(obj) -> str:
    """Short, stable hash of a JSON-serialisable object (used to name cache files)."""
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def keep_user(reviewer_id, user_frac):
    """Deterministic user sample for debug runs.

    The decision depends only on the reviewerID (CRC32, not Python's salted hash), so the
    same users are kept in both files and on every machine, and overlap users stay overlap
    users. user_frac=None keeps everyone.
    """
    if user_frac is None or user_frac >= 1:
        return True
    return zlib.crc32(reviewer_id.encode("utf-8")) % 1_000_000 < user_frac * 1_000_000


def read_reviews(path, users, user_frac=None):
    """Stream one gzipped JSON-lines file into integer arrays.

    users: dict reviewerID -> index, shared across both domains and extended in place.
    Returns (user, item, rating, n_items). Each rating is held as integers while reading
    (~28 bytes) instead of as a dict entry with string keys (~400 bytes), so the 27M-rating
    Books file fits in memory. If a (user, item) pair occurs more than once, the latest
    review is kept (ties: the later line). Output rows keep the order in which each pair
    first appears in the file.
    """
    items = {}
    u, it, r, t = array("q"), array("q"), array("f"), array("q")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rid = rec["reviewerID"]
            if not keep_user(rid, user_frac):
                continue
            u.append(users.setdefault(rid, len(users)))
            it.append(items.setdefault(rec["asin"], len(items)))
            r.append(float(rec["overall"]))
            t.append(int(rec.get("unixReviewTime", 0)))
    u, it, t = (np.frombuffer(a, dtype=np.int64) for a in (u, it, t))
    r = np.frombuffer(r, dtype=np.float32)
    n = len(u)
    if n == 0:
        return u.copy(), it.copy(), r.copy(), 0
    # de-duplicate (user, item): sort by pair, then time, then line number; keep the last of each group
    key = u * len(items) + it
    order = np.lexsort((np.arange(n), t, key))
    k_sorted = key[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    chosen = order[np.r_[starts[1:], n] - 1]               # latest review of each pair
    first_seen = np.minimum.reduceat(order, starts)        # first line of each pair
    keep = chosen[np.argsort(first_seen, kind="stable")]
    u, it, r = u[keep], it[keep], r[keep]
    assert ((r >= 1.0) & (r <= 5.0)).all(), "rating outside 1-5"
    return u, it, r, len(items)


def parse_and_cache(movies_path, books_path, out_dir, user_frac=None, force=False, log=print):
    """Returns dict with arrays m_user,m_item,m_rating,b_user,b_item,b_rating, the counts
    n_users,n_movies,n_books, and 'cache_key' (identifies the raw files + parse settings).

    The cache file name includes a hash of the raw files' size and modification time and of
    user_frac, so a debug parse is never silently reused for a full run (or vice versa), and
    a re-downloaded file is re-parsed.
    """
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for p in (movies_path, books_path):
        st = os.stat(p)
        files.append([os.path.basename(p), st.st_size, st.st_mtime_ns])
    key = config_hash(dict(version=PARSE_VERSION, files=files, user_frac=user_frac))
    cache = os.path.join(out_dir, f"ratings_{key}.npz")
    if os.path.exists(cache) and not force:
        log(f"Loading parsed ratings from cache {cache}")
        z = np.load(cache)
        data = {k: z[k] for k in z.files}
    else:
        users = {}
        log(f"Parsing {movies_path} (user_frac={user_frac})")
        mu, mi, mr, n_movies = read_reviews(movies_path, users, user_frac)
        log(f"Parsing {books_path} (user_frac={user_frac})")
        bu, bi, br, n_books = read_reviews(books_path, users, user_frac)
        data = dict(m_user=mu, m_item=mi, m_rating=mr, b_user=bu, b_item=bi, b_rating=br,
                    n_users=np.array(len(users)), n_movies=np.array(n_movies), n_books=np.array(n_books))
        del users
        np.savez(cache, **data)
        log(f"Parsed {len(mr):,} movie and {len(br):,} book ratings; cached to {cache}")
    data["cache_key"] = key
    return data


def make_split(data, seed, test_frac=0.2, fractions=(0.1, 0.2, 0.5, 1.0)):
    """Seed-dependent split. Book ratings of O_test users are removed from the training books."""
    rng = np.random.default_rng(seed)
    overlap = np.intersect1d(np.unique(data["m_user"]), np.unique(data["b_user"]))
    shuffled = rng.permutation(overlap)
    n_test = int(round(test_frac * len(shuffled)))
    test_users, train_users = np.sort(shuffled[:n_test]), np.sort(shuffled[n_test:])
    is_test = np.isin(data["b_user"], test_users)
    subsets = {}
    for f in fractions:
        size = int(np.floor(f * len(train_users)))
        subsets[f] = np.sort(rng.choice(train_users, size=size, replace=False))
    return dict(
        overlap=overlap, rho_true=len(overlap) / int(data["n_users"]),
        test_users=test_users, train_users=train_users, subsets=subsets,
        b_train=(data["b_user"][~is_test], data["b_item"][~is_test], data["b_rating"][~is_test]),
        truth=(data["b_user"][is_test], data["b_item"][is_test], data["b_rating"][is_test]),
    )
