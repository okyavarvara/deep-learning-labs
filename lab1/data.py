"""
Завантаження Iris, стратифікований поділ 70/30 і стандартизація ознак.
"""
import numpy as np
from sklearn.datasets import load_iris


def load_and_split(seed: int = 0):
    """
    Повертає X_train, y_train, X_test, y_test (уже стандартизовані).

    Поділ: для кожного класу (у порядку 0, 1, 2) перемішуємо індекси
    цього класу генератором np.random.default_rng(seed) і беремо перші
    35 обʼєктів у навчальну вибірку, решту 15 - у тестову.
    Стандартизація (mean/std, ddof=0) обчислюється лише на train.
    """
    iris = load_iris()
    X = iris.data.astype(np.float64)
    y = iris.target.astype(np.int64)

    rng = np.random.default_rng(seed)

    train_idx = []
    test_idx = []
    for c in (0, 1, 2):
        class_idx = np.flatnonzero(y == c)
        shuffled = rng.permutation(class_idx)
        train_idx.append(shuffled[:35])
        test_idx.append(shuffled[35:])

    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)

    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, y_train, X_test_std, y_test, mean, std
