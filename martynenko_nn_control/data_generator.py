import numpy as np
from config import RANDOM_SEED, N_SAMPLES, TEST_RATIO, NOISE

def make_two_moons(n_samples=N_SAMPLES, noise=NOISE, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n1 = n_samples // 2
    n2 = n_samples - n1
    t1 = rng.uniform(0, np.pi, n1)
    x1 = np.c_[np.cos(t1), np.sin(t1)]
    t2 = rng.uniform(0, np.pi, n2)
    x2 = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    X = np.vstack([x1, x2])
    y = np.r_[np.zeros(n1), np.ones(n2)].reshape(-1, 1)
    X += rng.normal(0, noise, X.shape)
    idx = rng.permutation(n_samples)
    return X[idx], y[idx]

def train_test_split(X, y, test_ratio=TEST_RATIO, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    test_idx = idx[:test_size]
    train_idx = idx[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

if __name__ == "__main__":
    X, y = make_two_moons()
    print(X.shape, y.shape)
