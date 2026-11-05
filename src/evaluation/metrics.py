"""Cold-start ranking evaluation: NDCG@n with the held-out rating as graded relevance."""
import numpy as np
import scipy.sparse as sp
import torch


def build_truth(truth, test_users, n_books):
    """CSR matrix (len(test_users) x n_books) of held-out ratings; row i = test_users[i]."""
    tu, ti, tr = truth
    row = np.searchsorted(test_users, tu)
    return sp.csr_matrix((tr, (row, ti)), shape=(len(test_users), n_books))


def ndcg_at_n(score_fn, test_users, truth_csr, n, user_batch=64):
    """Mean NDCG@n over test users. score_fn(user_ids) -> tensor (B, n_books).
    Candidates are all books (test users have no book history after the split).
    Gain = rating of the held-out book (0 if unrated); discount = 1/log2(rank+1)."""
    disc = 1.0 / np.log2(np.arange(2, n + 2))
    total, count = 0.0, 0
    for s in range(0, len(test_users), user_batch):
        rows = np.arange(s, min(s + user_batch, len(test_users)))
        scores = score_fn(test_users[rows])
        top = torch.topk(scores, k=min(n, scores.shape[1]), dim=1).indices.numpy()
        T = truth_csr[rows]
        gains = np.asarray(T[np.repeat(np.arange(len(rows)), top.shape[1]), top.reshape(-1)]).reshape(top.shape)
        dcg = (gains * disc[:top.shape[1]]).sum(1)
        for j in range(len(rows)):
            ideal = np.sort(T[j].data)[::-1][:n]
            idcg = (ideal * disc[:len(ideal)]).sum()
            if idcg > 0:
                total += dcg[j] / idcg
                count += 1
    return total / max(count, 1)
