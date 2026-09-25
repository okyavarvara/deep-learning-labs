"""
Мережа X -> Linear(4, 8) -> ReLU -> Linear(8, 3) реалізована вручну в NumPy,
без автоматичного диференціювання. Ваги мають форму (in, out), тобто
Z = A @ W + b, що відповідає вимозі завдання ЛР1.
"""
import numpy as np


def init_params(seed: int = 0):
    """
    He-ініціалізація W1 (4x8): N(0, sqrt(2/4)), b1 = 0.
    Xavier-ініціалізація W2 (8x3): N(0, sqrt(2/(8+3))), b2 = 0.
    Один rng = np.random.default_rng(seed): спочатку генерується W1, потім W2.
    """
    rng = np.random.default_rng(seed)
    std1 = np.sqrt(2.0 / 4.0)
    W1 = rng.normal(0.0, std1, size=(4, 8))
    b1 = np.zeros(8)
    std2 = np.sqrt(2.0 / (8.0 + 3.0))
    W2 = rng.normal(0.0, std2, size=(8, 3))
    b2 = np.zeros(3)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def log_softmax(Z: np.ndarray) -> np.ndarray:
    """Чисельно стабільний log-softmax зі зсувом на максимум (по рядках)."""
    z_shift = Z - Z.max(axis=1, keepdims=True)
    log_sum_exp = np.log(np.exp(z_shift).sum(axis=1, keepdims=True))
    return z_shift - log_sum_exp


def forward(params: dict, X: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Прямий прохід. Повертає логіти Z2 (N,3) і кеш проміжних значень,
    потрібних для зворотного проходу: X, Z1, A1.
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    Z1 = X @ W1 + b1          # (N, 8)
    A1 = np.maximum(0.0, Z1)  # (N, 8)   ReLU
    Z2 = A1 @ W2 + b2         # (N, 3)   логіти

    cache = {"X": X, "Z1": Z1, "A1": A1}
    return Z2, cache


def cross_entropy_loss(Z2: np.ndarray, y: np.ndarray) -> float:
    """Крос-ентропія, усереднена за обʼєктами, через log-softmax."""
    N = Z2.shape[0]
    logp = log_softmax(Z2)
    loss = -logp[np.arange(N), y].mean()
    return float(loss)


def backward(params: dict, cache: dict, Z2: np.ndarray, y: np.ndarray,
             fix_division: bool = True) -> dict:
    """
    Ручний зворотний прохід.

    fix_division=True  - правильна реалізація: dZ2 = (softmax(Z2) - onehot(y)) / N
    fix_division=False - навмисна помилка з п.4: ділення на N прибрано,
                          функція втрат (для звірки значення) лишається незмінною.
    """
    W2 = params["W2"]
    X, Z1, A1 = cache["X"], cache["Z1"], cache["A1"]
    N = Z2.shape[0]

    # softmax через ті самі стабільні обчислення
    logp = log_softmax(Z2)
    P = np.exp(logp)                       # (N, 3) softmax-ймовірності

    Y_onehot = np.zeros_like(P)
    Y_onehot[np.arange(N), y] = 1.0

    dZ2 = (P - Y_onehot)                   # (N, 3)
    if fix_division:
        dZ2 = dZ2 / N

    dW2 = A1.T @ dZ2                       # (8, 3)
    db2 = dZ2.sum(axis=0)                  # (3,)

    dA1 = dZ2 @ W2.T                       # (N, 8)
    dZ1 = dA1 * (Z1 > 0).astype(dZ2.dtype)  # (N, 8)  маска ReLU

    dW1 = X.T @ dZ1                        # (4, 8)
    db1 = dZ1.sum(axis=0)                  # (8,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
