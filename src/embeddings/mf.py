"""Matrix factorization by mini-batch gradient descent (Algorithm 1 in the paper).

R ~ P Q^T with P: users x k, Q: items x k, trained on observed ratings only with the
regularized squared error
    l(u,i,r) = (r - P[u].Q[i])^2 + lam (||P[u]||^2 + ||Q[i]||^2).
The constant 2 of the gradient is absorbed into the step size alpha (= lr). As in the
pseudocode, the accumulated update of a batch is divided by the batch size |S|, so lr is the
step on the batch-MEAN gradient (it is therefore larger than a per-rating SGD step).
"""
import numpy as np
import torch


def train_mf(user, item, rating, n_users, n_items, k, lam, lr, batch_size, epochs, seed, log=None):
    gen = torch.Generator().manual_seed(seed)
    P = torch.randn(n_users, k, generator=gen) * 0.01        # N(0, 0.01^2)
    Q = torch.randn(n_items, k, generator=gen) * 0.01
    u, i, r = (torch.as_tensor(a) for a in (user, item, rating))
    n = len(u)
    for epoch in range(epochs):
        perm = torch.randperm(n, generator=gen)
        sq = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            bu, bi, br = u[idx], i[idx], r[idx]
            pu, qi = P[bu], Q[bi]
            err = br - (pu * qi).sum(1)
            sq += float((err ** 2).sum())
            B = len(idx)
            # accumulate per-rating updates, then apply alpha * (sum of updates) / |S|
            P.index_add_(0, bu, err[:, None] * qi - lam * pu, alpha=lr / B)
            Q.index_add_(0, bi, err[:, None] * pu - lam * qi, alpha=lr / B)
        if log:
            log(f"  epoch {epoch + 1}/{epochs}  train RMSE {np.sqrt(sq / n):.4f}")
    return P.numpy(), Q.numpy()
