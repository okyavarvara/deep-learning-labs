"""Порівняння двох запусків навчання: чи збігаються поділ і статистики стандартизації.

    uv run python lab2_starter.py --data creditcard.csv --out results_rerun
    uv run python compare_runs.py --a results --b results_rerun

Індекси, мітки, Amount і параметри scaler мають збігатися точно. Оцінки моделі можуть
відрізнятися в молодших розрядах (залежать від версії PyTorch і платформи), тому для них
виводиться максимальна абсолютна різниця.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    args = parser.parse_args()

    ok = True
    with np.load(args.a / "predictions.npz") as pa, np.load(args.b / "predictions.npz") as pb:
        for name in pa.files:
            if name.endswith("_scores"):
                diff = float(np.max(np.abs(pa[name].astype(np.float64) - pb[name])))
                print(f"{name}: макс. |різниця| = {diff:.3e}")
            else:
                same = np.array_equal(pa[name], pb[name])
                ok &= same
                print(f"{name}: {'збігається' if same else 'РІЗНИТЬСЯ'}")
    with np.load(args.a / "preprocessing.npz") as ra, np.load(args.b / "preprocessing.npz") as rb:
        for name in ("mean", "scale", "var", "n_samples_seen"):
            same = np.array_equal(ra[name], rb[name])
            ok &= same
            print(f"scaler.{name}: {'збігається' if same else 'РІЗНИТЬСЯ'}")
    print("ПІДСУМОК:", "поділ і статистики збігаються" if ok else "є розбіжності")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
