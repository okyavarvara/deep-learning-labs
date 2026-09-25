"""
Чисельна перевірка окремих елементів градієнта центральною різницею.
"""
import copy
import numpy as np

from model import forward, cross_entropy_loss


def numeric_grad_entry(params: dict, X: np.ndarray, y: np.ndarray,
                        param_name: str, index: tuple, eps: float = 1e-6) -> float:
    """
    Обчислює чисельну похідну втрати за params[param_name][index]
    центральною різницею з кроком eps. Параметри тимчасово змінюються
    і відновлюються до початкового значення.
    """
    p = copy.deepcopy(params)

    theta0 = p[param_name][index]

    p[param_name][index] = theta0 + eps
    Z2_plus, _ = forward(p, X)
    L_plus = cross_entropy_loss(Z2_plus, y)

    p[param_name][index] = theta0 - eps
    Z2_minus, _ = forward(p, X)
    L_minus = cross_entropy_loss(Z2_minus, y)

    p[param_name][index] = theta0  # відновлення (для чистоти; p - копія)

    return (L_plus - L_minus) / (2 * eps)
