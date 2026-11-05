"""Scoring and recommendation (Eq. 6 / Algorithm 3).

f(u,b) = g(f_m(u)) . v_b   if |H_b(u)| < T   (cold start: map the movie embedding)
       = f_b(u)    . v_b   otherwise
"""
import numpy as np
import torch


def make_score_fn(mapper, U_movie, U_book, V_book, hist_counts, T):
    V = torch.as_tensor(V_book, dtype=torch.float32)

    def score_fn(users):
        cold = hist_counts[users] < T
        E = np.where(cold[:, None], mapper.map(U_movie[users]), U_book[users]).astype(np.float32)
        return torch.as_tensor(E) @ V.T

    return score_fn


def recommend(user, mapper, U_movie, U_book, V_book, hist_counts, T, n):
    """Top-n book indices for a single user (descending score)."""
    s = make_score_fn(mapper, U_movie, U_book, V_book, hist_counts, T)(np.array([user]))[0]
    return torch.topk(s, k=min(n, len(s))).indices.numpy()


def popularity_score_fn(train_items, n_books):
    pop = torch.as_tensor(np.bincount(train_items, minlength=n_books), dtype=torch.float32)
    return lambda users: pop[None, :].expand(len(users), -1)


def random_score_fn(n_books, seed):
    gen = torch.Generator().manual_seed(seed)
    return lambda users: torch.rand(len(users), n_books, generator=gen)
