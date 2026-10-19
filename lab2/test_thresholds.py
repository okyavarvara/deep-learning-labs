"""Самоперевірка thresholds.py без навченої моделі.

Запуск: uv run python test_thresholds.py

1. Контрольні набори A/B/C з умови: швидкий і повільний алгоритми збігаються й дають
   очікуваний поріг та вартість.
2. Багато випадкових малих наборів із великою кількістю рівних оцінок і без позитивів:
   таблиці й вибраний поріг швидкого та повільного алгоритмів збігаються точно.
"""

from __future__ import annotations

import numpy as np

from thresholds import (
    CONTROL_SETS,
    brute_force_best,
    brute_force_table,
    candidate_table,
    confusion,
    pick_threshold,
)


def same_tables(fast, slow) -> bool:
    return all(
        np.array_equal(fast[col].to_numpy(), slow[col].to_numpy())
        for col in ("threshold", "TP", "FP", "FN", "TN", "C")
    )


def check_control_sets() -> None:
    for name, (s, y, expected_t, expected_c) in CONTROL_SETS.items():
        fast = candidate_table(np.array(y), np.array(s))
        slow = brute_force_table(y, s)
        assert same_tables(fast, slow), f"набір {name}: таблиці різняться"
        best = pick_threshold(fast)
        slow_t, slow_c = brute_force_best(slow)
        assert float(best["threshold"]) == slow_t == expected_t, f"набір {name}: поріг"
        assert int(best["C"]) == slow_c == expected_c, f"набір {name}: вартість"
    print("контрольні набори A/B/C: OK")


def check_random(trials: int = 2000) -> None:
    rng = np.random.default_rng(12345)
    for trial in range(trials):
        n = int(rng.integers(1, 40))
        # малий алфавіт оцінок => багато рівних значень
        alphabet = rng.random(int(rng.integers(1, 8)))
        s = rng.choice(alphabet, size=n)
        p_positive = rng.choice([0.0, 0.1, 0.5, 1.0])
        y = (rng.random(n) < p_positive).astype(int)
        fast = candidate_table(y, s)
        slow = brute_force_table(y.tolist(), s.tolist())
        assert same_tables(fast, slow), f"випадковий набір {trial}: таблиці різняться"
        assert float(pick_threshold(fast)["threshold"]) == brute_force_best(slow)[0]
        # кожен рядок таблиці має збігатися з прямим застосуванням confusion()
        for _, row in fast.iterrows():
            tp, fp, fn, tn = confusion(y, s, row["threshold"])
            assert (tp, fp, fn, tn) == (row["TP"], row["FP"], row["FN"], row["TN"])
        assert (fast["TP"] + fast["FN"] == int(np.count_nonzero(y == 1))).all()
        assert (fast["FP"] + fast["TN"] == int(np.count_nonzero(y == 0))).all()
    print(f"{trials} випадкових наборів: OK")


if __name__ == "__main__":
    check_control_sets()
    check_random()
