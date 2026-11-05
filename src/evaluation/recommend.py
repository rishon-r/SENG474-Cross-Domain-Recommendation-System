"""Scoring and recommendation (Eq. 4 / Algorithm 3).

f(u,b) = g(f_m(u)) . v_b   if |H_b(u)| < T   (cold start: map the movie embedding)
       = f_b(u)    . v_b   otherwise
Books already in the user's training book history H_b(u) are not candidates (Algorithm 3,
line 6). Test users have no book history after the split, so for them every book is a candidate.
"""
import numpy as np
import scipy.sparse as sp
import torch


def history_matrix(b_user, b_item, n_users, n_books):
    """CSR matrix (n_users x n_books) with a 1 for every book in a user's training history H_b(u)."""
    return sp.csr_matrix((np.ones(len(b_user), dtype=np.float32), (b_user, b_item)), shape=(n_users, n_books))


def make_score_fn(mapper, U_movie, U_book, V_book, hist_counts, T, history=None):
    V = torch.as_tensor(V_book, dtype=torch.float32)

    def score_fn(users):
        cold = hist_counts[users] < T
        E = np.where(cold[:, None], mapper.map(U_movie[users]), U_book[users]).astype(np.float32)
        scores = torch.as_tensor(E) @ V.T
        if history is not None:                        # exclude b in H_b(u)
            rows, cols = history[users].nonzero()
            scores[torch.as_tensor(rows), torch.as_tensor(cols)] = float("-inf")
        return scores

    return score_fn


def recommend(user, mapper, U_movie, U_book, V_book, hist_counts, T, n, history=None):
    """Top-n book indices for a single user (descending score), excluding books in H_b(u)."""
    s = make_score_fn(mapper, U_movie, U_book, V_book, hist_counts, T, history)(np.array([user]))[0]
    return torch.topk(s, k=min(n, len(s))).indices.numpy()


def popularity_score_fn(train_items, n_books):
    pop = torch.as_tensor(np.bincount(train_items, minlength=n_books), dtype=torch.float32)
    return lambda users: pop[None, :].expand(len(users), -1)


def random_score_fn(n_books, seed):
    gen = torch.Generator().manual_seed(seed)
    return lambda users: torch.rand(len(users), n_books, generator=gen)
