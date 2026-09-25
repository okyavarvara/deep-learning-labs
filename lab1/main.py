"""
Головний скрипт ЛР1: прямий/зворотний прохід у NumPy, звірка з PyTorch,
чисельна перевірка градієнтів і дослід із навмисною помилкою.

Запуск (через uv):
    uv run python main.py                 # правильна реалізація (за замовчуванням)
    uv run python main.py --mode buggy     # дослід із навмисною помилкою (п.4)
    uv run python main.py --mode both      # обидва запуски поспіль

Результати друкуються в консоль і зберігаються в results.json.
"""
import argparse
import json

import numpy as np

from data import load_and_split
from model import init_params, forward, cross_entropy_loss, backward

try:
    from torch_ref import torch_loss_and_grads
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from numcheck import numeric_grad_entry

PARAM_ENTRIES = [
    ("W1", (0, 0)),
    ("b1", (0,)),
    ("W2", (0, 0)),
    ("b2", (0,)),
]

EPS_NUM = 1e-6
TOL_NUM = 1e-7
TOL_TORCH = 1e-12


def run_numpy_pass(params, X, y, fix_division: bool):
    Z2, cache = forward(params, X)
    loss = cross_entropy_loss(Z2, y)  # функція втрат НЕ змінюється в п.4
    grads = backward(params, cache, Z2, y, fix_division=fix_division)
    return loss, grads


def compare_with_torch(params, X, y, loss_np, grads_np):
    if not TORCH_AVAILABLE:
        print("[!] PyTorch недоступний у цьому середовищі - розділ пропущено. "
              "Запустіть `uv run python main.py`, маючи встановлений torch.")
        return None

    ref = torch_loss_and_grads(params, X, y)
    loss_torch = ref["loss"]
    grads_torch = ref["grads"]

    rows = []
    diff_loss = abs(loss_np - loss_torch)
    rows.append({
        "величина": "Втрата",
        "max_abs_diff": diff_loss,
        "passed": bool(diff_loss <= TOL_TORCH),
    })
    for name in ("W1", "b1", "W2", "b2"):
        diff = float(np.max(np.abs(grads_np[name] - grads_torch[name])))
        rows.append({
            "величина": f"Градієнт {name}",
            "max_abs_diff": diff,
            "passed": bool(diff <= TOL_TORCH),
        })

    return {
        "loss_numpy": loss_np,
        "loss_torch": loss_torch,
        "rows": rows,
    }


def run_numeric_check(params, X, y, grads_manual):
    rows = []
    for name, idx in PARAM_ENTRIES:
        g_num = numeric_grad_entry(params, X, y, name, idx, eps=EPS_NUM)
        g_manual = grads_manual[name][idx]
        diff = abs(g_num - g_manual)
        rows.append({
            "параметр": f"{name}{list(idx)}",
            "backward": float(g_manual),
            "numeric": float(g_num),
            "abs_diff": float(diff),
            "passed": bool(diff <= TOL_NUM),
        })
    return rows


def print_table(rows, keys, headers):
    widths = [max(len(h), max(len(f"{r[k]}") for r in rows)) for h, k in zip(headers, keys)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(line)
    print("-" * len(line))
    for r in rows:
        print(" | ".join(f"{r[k]}".ljust(w) for k, w in zip(keys, widths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["correct", "buggy", "both"], default="correct")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test, mean, std = load_and_split(seed=args.seed)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    results = {}

    modes = ["correct", "buggy"] if args.mode == "both" else [args.mode]

    for mode in modes:
        fix_division = (mode == "correct")
        print(f"\n===== Режим: {mode} (fix_division={fix_division}) =====")

        params = init_params(seed=0)  # ті самі початкові ваги для обох режимів
        loss_np, grads_np = run_numpy_pass(params, X_train, y_train, fix_division)
        print(f"Втрата (NumPy): {loss_np:.10f}")

        torch_cmp = compare_with_torch(params, X_train, y_train, loss_np, grads_np)
        if torch_cmp is not None:
            print("\nЗвірка з PyTorch:")
            print_table(torch_cmp["rows"], ["величина", "max_abs_diff", "passed"],
                        ["Величина", "Макс. абс. різниця", "Перевірку пройдено"])
            print(f"Втрата NumPy = {torch_cmp['loss_numpy']:.10f}, "
                  f"Втрата PyTorch = {torch_cmp['loss_torch']:.10f}")

        numeric_rows = run_numeric_check(params, X_train, y_train, grads_np)
        print("\nЧисельна перевірка градієнтів:")
        print_table(numeric_rows, ["параметр", "backward", "numeric", "abs_diff", "passed"],
                    ["Параметр", "Градієнт backward()", "Чисельна похідна", "Абс. різниця", "Перевірку пройдено"])

        results[mode] = {
            "loss_numpy": loss_np,
            "torch_comparison": torch_cmp,
            "numeric_check": numeric_rows,
        }

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nРезультати збережено у results.json")


if __name__ == "__main__":
    main()
