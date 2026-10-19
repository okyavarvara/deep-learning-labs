"""ЛР2, пункти 2-4: аналіз порогів зі збережених прогнозів (без навчання і без CSV).

    uv run python analyze.py --saved results --out analysis

Порядок: validation (криві, пошук t*) -> перевірки A/B/C -> test.
Test оцінюється лише після того, як усі три перевірки пройдено.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from thresholds import (
    CONTROL_SETS,
    brute_force_best,
    brute_force_table,
    candidate_table,
    confusion,
    cost,
    false_positive_rate,
    near_optimal,
    pick_threshold,
    precision_recall,
)


# --------------------------------------------------------------- форматування
def fmt_threshold(t: float) -> str:
    return "+inf" if np.isinf(t) else f"{t:.6g}"


def md_table(df: pd.DataFrame, digits: int = 6) -> str:
    """Markdown-таблиця; числа округлюються лише для відображення."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    lines = [head, sep]
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if value is None or (isinstance(value, float) and np.isnan(value)):
                cells.append("не визначено")
            elif isinstance(value, (float, np.floating)):
                cells.append("+inf" if np.isinf(value) else f"{value:.{digits}g}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load_predictions(directory: Path) -> dict[str, np.ndarray]:
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


# ------------------------------------------------------------ validation (п. 2)
def rule_row(label: str, y: np.ndarray, s: np.ndarray, t: float) -> dict:
    tp, fp, fn, tn = confusion(y, s, t)
    precision, recall = precision_recall(tp, fp, fn)
    return {
        "Правило": label,
        "Поріг": fmt_threshold(t),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "C": cost(fn, fp),
        "precision": precision,
        "recall": recall,
        "FPR": false_positive_rate(fp, tn),
    }


def plot_curves(y: np.ndarray, s: np.ndarray, t_star: float, out: Path) -> float:
    """PR- та ROC-криві на validation з позначеними порогами 0.5 та t*."""
    auc = float(roc_auc_score(y, s))
    marks = [(0.5, "t = 0.5", "tab:red"), (t_star, f"t* = {t_star:.4g}", "tab:green")]

    precision, recall, _ = precision_recall_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="tab:blue")
    for t, label, color in marks:
        if np.isinf(t):
            continue
        tp, fp, fn, _ = confusion(y, s, t)
        p, r = precision_recall(tp, fp, fn)
        if p is not None and r is not None:
            ax.scatter([r], [p], color=color, zorder=3, label=label)
    ax.axhline(np.mean(y), ls="--", color="gray", lw=1, label="частка шахрайських")
    ax.set(xlabel="Recall", ylabel="Precision", title="PR-крива (validation)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pr_curve.png", dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y, s)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, zoom in zip(axes, (False, True)):
        ax.plot(fpr, tpr, color="tab:blue", label=f"ROC-AUC = {auc:.4f}")
        if not zoom:
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        for t, label, color in marks:
            if np.isinf(t):
                continue
            tp, fp, fn, tn = confusion(y, s, t)
            _, recall_t = precision_recall(tp, fp, fn)
            fpr_t = false_positive_rate(fp, tn)
            ax.scatter([fpr_t], [recall_t], color=color, zorder=3, label=label)
        if zoom:
            ax.set_xlim(0, 0.01)
            ax.set_title("ROC, збільшено FPR ≤ 0.01")
        else:
            ax.set_title("ROC-крива (validation)")
        ax.set(xlabel="FPR", ylabel="TPR")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "roc_curve.png", dpi=150)
    plt.close(fig)
    return auc


# ----------------------------------------------------------- перевірки (п. 3)
def run_checks(out: Path) -> bool:
    summary_rows = []
    markdown = []
    for name, (s, y, expected_t, expected_c) in CONTROL_SETS.items():
        fast = candidate_table(np.array(y), np.array(s))
        slow = brute_force_table(y, s)
        tables_equal = all(
            np.array_equal(fast[col].to_numpy(), slow[col].to_numpy())
            for col in ("threshold", "TP", "FP", "FN", "TN", "C")
        )
        best = pick_threshold(fast)
        slow_t, slow_c = brute_force_best(slow)
        fast_t, fast_c = float(best["threshold"]), int(best["C"])
        passed = (
            tables_equal
            and fast_t == slow_t == expected_t
            and fast_c == slow_c == expected_c
        )
        summary_rows.append(
            {
                "set": name,
                "tables_equal": tables_equal,
                "t_fast": fmt_threshold(fast_t),
                "t_brute_force": fmt_threshold(slow_t),
                "t_expected": fmt_threshold(expected_t),
                "C_fast": fast_c,
                "C_brute_force": slow_c,
                "C_expected": expected_c,
                "passed": passed,
            }
        )
        slow.to_csv(out / f"check_{name}.csv", index=False, float_format="%.17g")
        markdown += [
            f"### Набір {name}: {'пройдено' if passed else 'НЕ ПРОЙДЕНО'}",
            f"Обраний поріг {fmt_threshold(slow_t)}, C = {slow_c}.",
            "",
            md_table(slow),
            "",
        ]
        print(
            f"перевірка {name}: {'OK' if passed else 'ПОМИЛКА'}  "
            f"t*={fmt_threshold(slow_t)}  C={slow_c}"
        )
    pd.DataFrame(summary_rows).to_csv(out / "checks_summary.csv", index=False)
    (out / "checks.md").write_text("\n".join(markdown), encoding="utf-8")
    return all(row["passed"] for row in summary_rows)


# --------------------------------------------------------------- test (п. 4)
def error_examples(data: dict, t: float, k: int = 5) -> pd.DataFrame:
    """Перші k FP і перші k FN за номером рядка CSV (нумерація від 0)."""
    y = data["test_y"]
    s = data["test_scores"].astype(np.float64)  # той самий тип, що й під час вибору t*
    row_ids, amount = data["test_row_ids"], data["test_amount"]
    predicted = s >= t
    parts = []
    for kind, mask in (("FP", predicted & (y == 0)), ("FN", ~predicted & (y == 1))):
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(row_ids[idx], kind="stable")][:k]
        for i in idx:
            parts.append(
                {
                    "row": int(row_ids[i]),
                    "type": kind,
                    "label": int(y[i]),
                    "score": float(s[i]),
                    "amount": float(amount[i]),
                    "score_minus_t": float(s[i] - t),
                }
            )
    return pd.DataFrame(
        parts, columns=["row", "type", "label", "score", "amount", "score_minus_t"]
    )


# ----------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved", type=Path, required=True, help="тека з predictions.npz")
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    data = load_predictions(args.saved)

    # --- п. 2: validation
    y_val = data["validation_y"]
    s_val = data["validation_scores"].astype(np.float64)
    table = candidate_table(y_val, s_val)
    table.to_csv(out / "validation_candidates.csv", index=False, float_format="%.17g")
    best = pick_threshold(table)
    t_star = float(best["threshold"])
    auc = plot_curves(y_val, s_val, t_star, out)

    summary_val = pd.DataFrame(
        [
            rule_row("Стандартний поріг", y_val, s_val, 0.5),
            rule_row("Мінімум вартості", y_val, s_val, t_star),
            rule_row("Усі негативні", y_val, s_val, float("inf")),
        ]
    )
    summary_val.to_csv(out / "validation_summary.csv", index=False)
    near = near_optimal(table)
    near.to_csv(out / "near_optimal.csv", index=False, float_format="%.17g")

    n_unique = int(len(np.unique(s_val)))
    threshold_info = {
        "t_star": "inf" if np.isinf(t_star) else t_star,
        "validation_cost": int(best["C"]),
        "validation_roc_auc": auc,
        "n_validation": int(len(s_val)),
        "n_unique_scores": n_unique,
        "n_candidates": int(len(table)),
        "n_repeated_scores": int(len(s_val) - n_unique),
    }
    (out / "threshold.json").write_text(
        json.dumps(threshold_info, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )

    print(f"validation ROC-AUC = {auc:.6f}")
    print(f"t* = {t_star!r}  C(t*) = {int(best['C'])}")
    print(
        f"унікальних оцінок {n_unique}, кандидатів {len(table)}, "
        f"повторних оцінок {len(s_val) - n_unique}"
    )
    print(summary_val.to_string(index=False))
    print("найближчі за вартістю кандидати:")
    print(near.to_string(index=False))

    # --- п. 3: перевірки до test
    if not run_checks(out):
        print("Перевірки не пройдено: test не оцінюється.", file=sys.stderr)
        return 1

    # --- п. 4: test із зафіксованим t*
    y_test = data["test_y"]
    s_test = data["test_scores"].astype(np.float64)
    tp, fp, fn, tn = confusion(y_test, s_test, t_star)
    precision, recall = precision_recall(tp, fp, fn)
    _, fp_05, fn_05, _ = confusion(y_test, s_test, 0.5)
    n_pos = int(np.count_nonzero(y_test == 1))
    test_metrics = {
        "t_star": t_star,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "C": cost(fn, fp),
        "precision": precision,
        "recall": recall,
        "C_at_0.5": cost(fn_05, fp_05),
        "C_all_negative": cost(n_pos, 0),
    }
    (out / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print("test:", json.dumps(test_metrics, ensure_ascii=False))

    errors = error_examples(data, t_star)
    errors.to_csv(out / "test_errors.csv", index=False, float_format="%.17g")
    print(errors.to_string(index=False))
    if len(errors):
        grouped = errors.groupby("type")
        print("медіана Amount:", grouped["amount"].median().to_dict())
        print(
            "середня |score - t*|:",
            grouped["score_minus_t"].apply(lambda v: v.abs().mean()).to_dict(),
        )

    markdown = [
        f"ROC-AUC (validation) = {auc:.6f}; t* = {t_star!r}",
        "",
        "## Validation",
        md_table(summary_val),
        "",
        "## Test (t* зафіксовано на validation)",
        md_table(pd.DataFrame([test_metrics])),
        "",
        "## Перші п'ять FP і FN на test",
        md_table(errors) if len(errors) else "помилок немає",
    ]
    (out / "summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
