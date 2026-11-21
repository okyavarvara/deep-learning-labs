"""Пункт 1: експорт моделі в ONNX, підрахунок вузлів, діапазони ваг згорток (пункт 4)."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

import common as C


def export_fp32(model, path: Path) -> float:
    """torch.onnx.export: dynamo=False, opset 17, вхід input, вихід logits, змінний пакет."""
    model.eval()
    dummy = torch.zeros(1, 3, C.IMG_SIZE, C.IMG_SIZE, dtype=torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(model, dummy, str(path), dynamo=False, opset_version=C.OPSET,
                          input_names=[C.INPUT_NAME], output_names=[C.OUTPUT_NAME],
                          dynamic_axes={C.INPUT_NAME: {0: "batch"}, C.OUTPUT_NAME: {0: "batch"}})
    return time.perf_counter() - t0


def node_counts(path: Path) -> dict:
    g = onnx.load(str(path)).graph
    cnt = Counter(n.op_type for n in g.node)
    return {"Conv": cnt.get("Conv", 0), "BatchNormalization": cnt.get("BatchNormalization", 0),
            "all_ops": dict(sorted(cnt.items()))}


def conv_ranges(path: Path) -> dict:
    """Для кожного вузла Conv: alpha_c = max|w| за елементами вихідного каналу c, r = max_c alpha / median_c alpha.
    Ваги беруться з graph.initializer за іменем другого входу вузла (форма Cout x Cin x kH x kW)."""
    g = onnx.load(str(path)).graph
    inits = {t.name: numpy_helper.to_array(t) for t in g.initializer}
    rows = []
    for node in g.node:
        if node.op_type != "Conv":
            continue
        w = inits[node.input[1]]
        assert w.ndim == 4, (node.name, w.shape)
        alpha = np.abs(w.astype(np.float64)).reshape(w.shape[0], -1).max(axis=1)
        r = float(alpha.max() / np.median(alpha))
        rows.append({"node": node.name, "weight": node.input[1], "shape": list(w.shape),
                     "alpha_max": float(alpha.max()), "alpha_median": float(np.median(alpha)),
                     "alpha_min": float(alpha.min()), "r": r})
    rs = np.array([x["r"] for x in rows])
    top3 = sorted(rows, key=lambda x: -x["r"])[:3]
    r_med = float(np.median(rs))
    return {"convs": rows, "n_convs": len(rows), "top3": top3, "r_median": r_med,
            "median_cost_bits": float(np.log2(r_med)),
            "top1_cost_bits": float(np.log2(top3[0]["r"]))}
