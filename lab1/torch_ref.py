"""
Еталонна реалізація тієї самої мережі в PyTorch для звірки з ручним NumPy
backprop. Автоматичне диференціювання (autograd) використовується ЛИШЕ тут.

Важливо: torch.nn.Linear зберігає ваги у формі (out, in), тоді як у
NumPy-реалізації ваги мають форму (in, out). Тому при копіюванні
параметрів матриці ваг транспонуються.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_torch_model(params: dict) -> nn.Module:
    """Створює nn.Sequential(Linear(4,8), ReLU, Linear(8,3)) і копіює ваги."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 3),
    )
    # ВАЖЛИВО: спершу перевести модель у float64, і лише потім копіювати
    # float64-ваги. Якщо копіювати до .double(), значення спершу зрізаються
    # до float32 (втрата точності ~1e-7), що й давало хибну розбіжність
    # зі звіркою на рівні 1e-8 замість очікуваних ~1e-15.
    model = model.double()
    with torch.no_grad():
        # NumPy W1: (4,8) -> torch weight (8,4): транспонування
        model[0].weight.copy_(torch.tensor(params["W1"].T, dtype=torch.float64))
        model[0].bias.copy_(torch.tensor(params["b1"], dtype=torch.float64))
        model[2].weight.copy_(torch.tensor(params["W2"].T, dtype=torch.float64))
        model[2].bias.copy_(torch.tensor(params["b2"], dtype=torch.float64))
    return model


def torch_loss_and_grads(params: dict, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Прямий+зворотний прохід у PyTorch на тих самих X, y і параметрах.
    Повертає втрату (float) і градієнти чотирьох параметрів у формі,
    сумісній з NumPy-реалізацією (ваги транспоновано назад до (in, out)).
    """
    model = build_torch_model(params)
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.long)

    logits = model(Xt)
    loss = F.cross_entropy(logits, yt)  # усереднена крос-ентропія, очікує логіти

    model.zero_grad()
    loss.backward()

    grads = {
        "W1": model[0].weight.grad.detach().numpy().T.copy(),  # (8,4)->(4,8)
        "b1": model[0].bias.grad.detach().numpy().copy(),
        "W2": model[2].weight.grad.detach().numpy().T.copy(),  # (3,8)->(8,3)
        "b2": model[2].bias.grad.detach().numpy().copy(),
    }
    return {"loss": float(loss.item()), "grads": grads}
