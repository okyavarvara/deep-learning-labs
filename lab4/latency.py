"""Пункт 5: вимірювання затримки за протоколом з умови.

Обидві сесії створюються заздалегідь; вхід один і той самий (перше зображення test, 1x3x160x160);
3 раунди, у кожному спершу FP32, потім INT8; для кожного варіанта 20 прогрівальних викликів без запису
і 100 вимірів; вимір = один sess.run між двома зчитуваннями time.perf_counter(), і нічого більше.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import common as C


def measure(sess, x: np.ndarray, n_warmup: int, n_measure: int) -> list[float]:
    feed = {C.INPUT_NAME: x}
    for _ in range(n_warmup):
        sess.run(None, feed)
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        sess.run(None, feed)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)       # мс
    return times


def run_protocol(paths: dict[str, Path], x1: np.ndarray) -> dict:
    assert x1.shape == (1, 3, C.IMG_SIZE, C.IMG_SIZE) and x1.dtype == np.float32
    sessions = {v: C.make_session(p) for v, p in paths.items()}          # 1: сесії створено заздалегідь
    raw = {v: [] for v in paths}                                          # v -> список раундів
    for rnd in range(1, C.LAT_ROUNDS + 1):                                # 3: раунди; FP32, потім INT8
        for v in paths:
            raw[v].append(measure(sessions[v], x1, C.LAT_WARMUP, C.LAT_MEASURES))
        print(f"   раунд {rnd}/{C.LAT_ROUNDS}: " + ", ".join(
            f"{v} p50 = {np.percentile(raw[v][-1], 50):.3f} мс" for v in paths))
    res = {}
    for v, p in paths.items():
        allv = np.array([t for rnd in raw[v] for t in rnd])
        p50_rounds = [float(np.percentile(r, 50)) for r in raw[v]]
        res[v] = {"file": p.name, "file_bytes": p.stat().st_size, "file_mb": p.stat().st_size / 1e6,
                  "p50_ms": float(np.percentile(allv, 50)), "p95_ms": float(np.percentile(allv, 95)),
                  "mean_ms": float(allv.mean()), "min_ms": float(allv.min()), "max_ms": float(allv.max()),
                  "p50_by_round_ms": p50_rounds,
                  "p95_by_round_ms": [float(np.percentile(r, 95)) for r in raw[v]],
                  "p50_round_min": min(p50_rounds), "p50_round_max": max(p50_rounds),
                  "n_measurements": int(allv.size), "all_measurements_ms": raw[v]}
    f, q = res["fp32"], res["int8"]
    disjoint = f["p50_round_max"] < q["p50_round_min"] or q["p50_round_max"] < f["p50_round_min"]
    res["comparison"] = {
        "size_ratio_fp32_over_int8": f["file_bytes"] / q["file_bytes"],
        "speedup_p50": f["p50_ms"] / q["p50_ms"],
        "speedup_p95": f["p95_ms"] / q["p95_ms"],
        "p50_intervals_disjoint": bool(disjoint),
        "verdict": ("зміну затримки виміряно: відрізки [min, max] значень p50 за раундами не перетинаються"
                    if disjoint else
                    "зміну затримки не виміряно: відрізки [min, max] значень p50 за раундами перетинаються")}
    res["protocol"] = {"batch": C.LAT_BATCH, "threads": C.LAT_THREADS, "rounds": C.LAT_ROUNDS,
                       "warmup": C.LAT_WARMUP, "measures": C.LAT_MEASURES,
                       "input": "перше зображення test за порядком split.csv, float32 1x3x160x160",
                       "order_in_round": list(paths)}
    return res
