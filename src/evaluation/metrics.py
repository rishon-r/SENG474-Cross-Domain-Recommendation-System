"""Cold-start ranking evaluation: NDCG@n with the held-out rating as graded relevance."""
import numpy as np
import scipy.sparse as sp
import torch


def build_truth(truth, test_users, n_books):
    """CSR matrix (len(test_users) x n_books) of held-out ratings; row i = test_users[i]."""
    tu, ti, tr = truth
    row = np.searchsorted(test_users, tu)
    return sp.csr_matrix((tr, (row, ti)), shape=(len(test_users), n_books))


def evaluate_at_n(score_fn, test_users, truth_csr, n, user_batch=64):
    """Returns (mean NDCG@n, mean Recall@n) over users with at least one held-out book.
    NDCG: gain = held-out rating (0 if unrated), discount = 1/log2(rank+1).
    Recall@n: held-out books found in the top n / number of held-out books.
    score_fn(user_ids) -> tensor (B, n_books); candidates are all books."""
    disc = 1.0 / np.log2(np.arange(2, n + 2))
    nd_total, rc_total, count = 0.0, 0.0, 0
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
                nd_total += dcg[j] / idcg
                rc_total += (gains[j] > 0).sum() / T[j].nnz
                count += 1
    count = max(count, 1)
    return nd_total / count, rc_total / count


def ndcg_at_n(score_fn, test_users, truth_csr, n, user_batch=64):
    """Mean NDCG@n (kept for the tests and older callers)."""
    return evaluate_at_n(score_fn, test_users, truth_csr, n, user_batch)[0]
