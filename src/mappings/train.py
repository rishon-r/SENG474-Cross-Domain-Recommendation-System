"""Training of the three mapping strategies on an overlap subset (Algorithm 2: Fit)."""
import numpy as np
import torch
from sklearn.cluster import KMeans

from .models import MLP, MetaNet


def fit(model, X, Y, lr, batch_size, epochs, seed, optimizer="sgd"):
    """Gradient descent on L_g = mean_batch ||g(x) - y||^2 (Eq. 3)."""
    gen = torch.Generator().manual_seed(seed)
    X, Y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)
    opt = (torch.optim.SGD(model.parameters(), lr=lr) if optimizer == "sgd"
           else torch.optim.Adam(model.parameters(), lr=lr))
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            loss = ((model(X[idx]) - Y[idx]) ** 2).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def _apply(model, X):
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).numpy()


class SharedMapper:
    """One MLP for all users."""
    name = "shared"

    def __init__(self, k, cfg):
        self.k, self.cfg = k, cfg

    def fit(self, X, Y, seed):
        c = self.cfg
        torch.manual_seed(seed)
        self.model = fit(MLP(self.k, c["hidden"]), X, Y, c["lr"], c["batch_size"], c["epochs"], seed, c["optimizer"])
        return self

    def map(self, X):
        return _apply(self.model, X)


class ClusterMapper:
    """K-means on movie embeddings; one MLP per cluster; nearest centroid at inference."""
    name = "cluster"

    def __init__(self, k, cfg):
        self.k, self.cfg = k, cfg

    def fit(self, X, Y, seed):
        c = self.cfg
        K = max(1, min(c["K"], len(X)))
        self.km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(X)
        labels = self.km.labels_
        self.models = []
        for j in range(K):
            torch.manual_seed(seed + j)
            m = fit(MLP(self.k, c["hidden"]), X[labels == j], Y[labels == j],
                    c["lr"], c["batch_size"], c["epochs"], seed + j, c["optimizer"])
            self.models.append(m)
        return self

    def map(self, X):
        labels = self.km.predict(X)
        out = np.zeros_like(X, dtype=np.float32)
        for j, m in enumerate(self.models):
            sel = labels == j
            if sel.any():
                out[sel] = _apply(m, X[sel])
        return out


class SpecificMapper:
    """Meta-network m_phi generating a user-specific matrix W_u; g(x) = W_u x."""
    name = "specific"

    def __init__(self, k, cfg):
        self.k, self.cfg = k, cfg

    def fit(self, X, Y, seed):
        c = self.cfg
        torch.manual_seed(seed)
        self.model = fit(MetaNet(self.k, c["meta_hidden"]), X, Y, c["lr"], c["batch_size"], c["epochs"], seed, c["optimizer"])
        return self

    def map(self, X):
        return _apply(self.model, X)


STRATEGIES = {"shared": SharedMapper, "cluster": ClusterMapper, "specific": SpecificMapper}
