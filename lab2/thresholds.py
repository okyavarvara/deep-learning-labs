"""Підрахунок матриці плутанини, вартості та пошук порога.

Тут лише логіка, без читання файлів і побудови графіків. Використовується в analyze.py
та test_thresholds.py.

Правило рішення: транзакція вважається шахрайською, якщо s >= t.
Вартість: C(t) = 10 * FN(t) + FP(t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COST_FN = 10
COST_FP = 1
COLUMNS = ["threshold", "TP", "FP", "FN", "TN", "C"]


# ------------------------------------------------------------------ метрики
def confusion(y: np.ndarray, s: np.ndarray, t: float) -> tuple[int, int, int, int]:
    """TP, FP, FN, TN для порога t за правилом s >= t. Підходить і для t = +inf."""
    y = np.asarray(y)
    predicted = np.asarray(s, dtype=np.float64) >= t
    actual = y == 1
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = int(np.count_nonzero(~predicted & ~actual))
    return tp, fp, fn, tn


def cost(fn: int, fp: int) -> int:
    return COST_FN * fn + COST_FP * fp


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float | None, float | None]:
    """(precision, recall). Якщо знаменник нульовий, метрика не визначена: None."""
    precision = tp / (tp + fp) if tp + fp > 0 else None
    recall = tp / (tp + fn) if tp + fn > 0 else None
    return precision, recall


def false_positive_rate(fp: int, tn: int) -> float | None:
    return fp / (fp + tn) if fp + tn > 0 else None


# ----------------------------------------- швидкий пошук: сортування + cumsum
def candidate_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Таблиця всіх кандидатів за O(n log n).

    Кандидати: +inf та кожна унікальна оцінка. Рядки йдуть за спаданням порога.

    Оцінки сортуються за спаданням. Для порога t, що дорівнює k-й оцінці, правило
    s >= t охоплює всі об'єкти з оцінкою не меншою за неї, тому беруться кумулятивні суми
    в КІНЦІ групи рівних оцінок. Так рівні оцінки завжди отримують однаковий прогноз.
    """
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    positives = int(np.count_nonzero(y == 1))
    negatives = n - positives

    order = np.argsort(-s, kind="stable")
    s_sorted, y_sorted = s[order], y[order]
    cum_tp = np.cumsum(y_sorted == 1)  # позитивних серед k найбільших оцінок
    cum_predicted = np.arange(1, n + 1)  # усього об'єктів серед k найбільших оцінок
    group_end = np.concatenate([s_sorted[1:] != s_sorted[:-1], [True]])

    threshold = np.concatenate([[np.inf], s_sorted[group_end]])
    tp = np.concatenate([[0], cum_tp[group_end]])
    predicted = np.concatenate([[0], cum_predicted[group_end]])
    fp = predicted - tp
    fn = positives - tp
    tn = negatives - fp
    return pd.DataFrame(
        {
            "threshold": threshold,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "C": COST_FN * fn + COST_FP * fp,
        }
    )


def pick_threshold(table: pd.DataFrame) -> pd.Series:
    """Рядок із мінімальною вартістю; за рівної вартості - з найбільшим порогом."""
    minimum = table["C"].min()
    tied = table[table["C"] == minimum]
    return tied.loc[tied["threshold"].idxmax()]


def near_optimal(table: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """k найкращих кандидатів за вартістю (за рівної вартості - більший поріг першим)."""
    ranked = table.sort_values(["C", "threshold"], ascending=[True, False])
    return ranked.head(k).reset_index(drop=True)


# --------------------------------------- повільний незалежний прямий перебір
def brute_force_table(y: list[int], s: list[float]) -> pd.DataFrame:
    """Лише для перевірки на малих наборах (п. 3 завдання).

    Не користується швидким алгоритмом: список кандидатів будується окремо, а матриця
    плутанини для кожного кандидата рахується прямим застосуванням правила s_i >= t
    у звичайному циклі.
    """
    candidates = [float("inf")] + sorted(set(s), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for label, score in zip(y, s):
            predicted = score >= t
            if predicted and label == 1:
                tp += 1
            elif predicted and label == 0:
                fp += 1
            elif not predicted and label == 1:
                fn += 1
            else:
                tn += 1
        rows.append((t, tp, fp, fn, tn, COST_FN * fn + COST_FP * fp))
    return pd.DataFrame(rows, columns=COLUMNS)


def brute_force_best(table: pd.DataFrame) -> tuple[float, int]:
    """Послідовний вибір найкращого рядка; за рівної вартості перемагає більший поріг."""
    best_t, best_c = None, None
    for t, c in zip(table["threshold"], table["C"]):
        if best_c is None or c < best_c or (c == best_c and t > best_t):
            best_t, best_c = float(t), int(c)
    return best_t, best_c


# --------------------------------------------------- контрольні набори A/B/C
# (оцінки, мітки, очікуваний поріг, очікувана вартість)
CONTROL_SETS: dict[str, tuple[list[float], list[int], float, int]] = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0], 0.4, 2),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0], float("inf"), 0),
    # одинадцять оцінок 0.5 підряд: перша з міткою 1, решта десять з міткою 0
    "C": ([0.8] + [0.5] * 11 + [0.2], [1, 1] + [0] * 10 + [0], 0.8, 10),
}
