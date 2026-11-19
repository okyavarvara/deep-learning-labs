"""Команда аналізу ЛР3: перевірка хешу split.csv у журналах; таблиця, криві й ефекти зі збережених
журналів; перевірка правила вибору; одноразова тестова оцінка lab3_model.pt на CPU,
матриця плутанини та п'ять характерних помилок.

    uv run python analyze.py --data-dir <шлях до теки, де лежать seg_train і seg_test>

Код, що завантажує test, викликається лише звідси. Результати test ніде не використовуються
для вибору режиму, запуску, гіперпараметрів чи кількості епох.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import common as C

RUN_ORDER = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
             ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]
COLORS = {"scratch": "#444444", "frozen": "#0072B2", "unfreeze": "#D55E00"}
STYLES = {1: "-", 2: "--", 3: ":"}
MARKERS = {1: "o", 2: "s", 3: "^"}


def load_runs(out: Path, split_hash: str) -> list[dict]:
    runs = []
    for regime, seed in RUN_ORDER:
        p = out / "runs" / f"{regime}_seed{seed}" / "run.json"
        if not p.exists():
            raise SystemExit(f"Немає журналу {p}: потрібні рівно сім запусків.")
        runs.append(json.loads(p.read_text(encoding="utf-8")))
    extra = sorted(d.name for d in (out / "runs").iterdir() if d.is_dir()) if (out / "runs").exists() else []
    expected = sorted(f"{r}_seed{s}" for r, s in RUN_ORDER)
    if extra != expected:
        raise SystemExit(f"У runs/ мають бути рівно сім запусків {expected}, знайдено {extra}.")
    hashes = {r["split_sha256"] for r in runs}
    if len(hashes) != 1 or hashes != {split_hash}:
        raise SystemExit("SHA-256 split.csv у журналах відрізняється між запусками або від "
                         f"поточного split.csv ({split_hash}). Запуски на різних поділах "
                         "порівнювати не можна - перенавчіть усі сім.")
    return runs


def plot_curves(runs, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)
    for r in runs:
        ep = [e["epoch"] for e in r["epochs"]]
        kw = dict(color=COLORS[r["regime"]], ls=STYLES[r["seed"]], marker=MARKERS[r["seed"]],
                  ms=4, lw=1.6, label=f"{r['regime']} (seed {r['seed']})")
        axes[0].plot(ep, [e["train_loss"] for e in r["epochs"]], **kw)
        axes[1].plot(ep, [e["val_macro_f1"] for e in r["epochs"]], **kw)
    axes[0].set(title="Втрата на train", xlabel="Епоха", ylabel="Втрата (cross-entropy)")
    axes[1].set(title="macro-F1 на validation", xlabel="Епоха", ylabel="macro-F1")
    for ax in axes:
        ax.set_xticks(range(1, 9))
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def effects(runs) -> dict:
    F = {(r["regime"], r["seed"]): r["final"]["val_macro_f1"] for r in runs}
    fro = [F[("frozen", s)] for s in (1, 2, 3)]
    unf = [F[("unfreeze", s)] for s in (1, 2, 3)]
    e = {"F_scratch": F[("scratch", 1)], "F_frozen_mean": float(np.mean(fro)),
         "R_frozen": max(fro) - min(fro), "F_unfreeze_mean": float(np.mean(unf)),
         "R_unfreeze": max(unf) - min(unf)}
    e["R"] = max(e["R_frozen"], e["R_unfreeze"])
    e["delta_pre"] = e["F_frozen_mean"] - e["F_scratch"]
    e["delta_unf"] = e["F_unfreeze_mean"] - e["F_frozen_mean"]
    e["delta_total"] = e["delta_pre"] + e["delta_unf"]
    for k in ("pre", "unf"):
        e[f"verdict_{k}"] = ("модуль різниці ≤ R: різниця не відрізняється від розкиду, результату немає"
                             if abs(e[f"delta_{k}"]) <= e["R"] else
                             "модуль різниці > R: ефект вартий перевірки, але не доведений")
    return e


def plot_confusion(cm, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), C.CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(6), C.CLASSES)
    ax.set(xlabel="Передбачений клас", ylabel="Справжній клас", title="Матриця плутанини (test)")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pick_errors(test_items, y_true, y_pred, probs):
    """П'ять характерних помилок за правилом з умови."""
    pairs = Counter((int(t), int(p)) for t, p in zip(y_true, y_pred) if t != p)
    ordered = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    names = [f for f, _ in test_items]
    files_of = {pr: sorted(names[i] for i in range(len(names))
                           if y_true[i] == pr[0] and y_pred[i] == pr[1]) for pr, _ in ordered}
    chosen: list[tuple[tuple[int, int], int, str]] = []
    for pr, _ in ordered[:5]:
        chosen.append((pr, pairs[pr], files_of[pr][0]))
    rnd = 1
    while len(chosen) < 5 and ordered:
        added = False
        for pr, _ in ordered:
            if len(chosen) < 5 and len(files_of[pr]) > rnd:
                chosen.append((pr, pairs[pr], files_of[pr][rnd]))
                added = True
        if not added:
            break
        rnd += 1
    idx_of = {f: i for i, f in enumerate(names)}
    return [dict(file=f, true=C.CLASSES[pr[0]], pred=C.CLASSES[pr[1]],
                 prob=float(probs[idx_of[f], pr[1]]), count=cnt) for pr, cnt, f in chosen]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="тека, у якій лежать seg_train/seg_train і seg_test/seg_test")
    ap.add_argument("--out-dir", type=Path, default=C.HERE)
    args = ap.parse_args()
    out, data_dir = args.out_dir.resolve(), args.data_dir.resolve()
    fig_dir, res_dir = out / "figures", out / "results"
    fig_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)

    # 1. хеш split.csv
    split_hash = C.sha256_file(out / "split.csv")
    runs = load_runs(out, split_hash)
    print(f"SHA-256 split.csv однаковий у всіх 7 журналах і збігається з поточним: {split_hash}")
    split = C.read_split(out / "split.csv")

    # 2. таблиця семи запусків
    md = ["## Таблиця семи запусків\n",
          "| # | Режим | Зерно | Втрата train, епоха 8 | macro-F1 validation, епоха 8 | Час навчання, с |",
          "|---|---|---|---|---|---|"]
    for i, r in enumerate(runs, 1):
        md.append(f"| {i} | {r['regime']} | {r['seed']} | {r['final']['train_loss']:.4f} | "
                  f"{r['final']['val_macro_f1']:.4f} | {r['train_time_sec']:.0f} |")
    plot_curves(runs, fig_dir / "curves.png")

    # 3. розклад ефектів
    e = effects(runs)
    md += ["\n## Розклад ефектів\n", "| Величина | Значення |", "|---|---|",
           f"| F̄_frozen, R_frozen | {e['F_frozen_mean']:.4f}, {e['R_frozen']:.4f} |",
           f"| F̄_unfreeze, R_unfreeze | {e['F_unfreeze_mean']:.4f}, {e['R_unfreeze']:.4f} |",
           f"| F_scratch,1 | {e['F_scratch']:.4f} |",
           f"| R | {e['R']:.4f} |",
           f"| Δ_pre | {e['delta_pre']:+.4f} ({e['verdict_pre']}) |",
           f"| Δ_unf | {e['delta_unf']:+.4f} ({e['verdict_unf']}) |",
           f"| Δ_pre + Δ_unf | {e['delta_total']:+.4f} |"]

    # 4. перевірка правила вибору
    results = [{"regime": r["regime"], "seed": r["seed"], "f1": r["final"]["val_macro_f1"]} for r in runs]
    regime, seed, means = C.select_best(results)
    ckpt = torch.load(out / "lab3_model.pt", map_location="cpu", weights_only=True)
    if (ckpt["regime"], ckpt["seed"]) != (regime, seed):
        raise SystemExit(f"lab3_model.pt ({ckpt['regime']}, seed {ckpt['seed']}) не відповідає "
                         f"правилу вибору ({regime}, seed {seed}).")
    if ckpt["classes"] != C.CLASSES or ckpt["img_size"] != C.IMG_SIZE:
        raise SystemExit("Невірні classes/img_size у lab3_model.pt")
    sel_f = next(x["f1"] for x in results if x["regime"] == regime and x["seed"] == seed)
    md += ["\n## Вибір конфігурації (за validation)\n",
           f"Середні F̄ за режимами: " + ", ".join(f"{k} = {v:.4f}" for k, v in means.items()),
           f"\nВибрано: режим **{regime}**, зерно **{seed}**, F = {sel_f:.4f}; "
           f"lab3_model.pt відповідає правилу вибору: так.",
           f"\n|Δ_unf| ≤ R? {'так - вибір між frozen і unfreeze визначив шум' if abs(e['delta_unf']) <= e['R'] else 'ні'}."]

    # 5. тестова оцінка (один раз, лише тут, на CPU)
    device = torch.device("cpu")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # контроль: логіти validation з lab3_model.pt на CPU ~ збережені після навчання
    val_ds = C.SplitDataset(split["validation"], "validation", data_dir, train=False)
    val_logits = C.predict_logits(model, C.eval_loader(val_ds), device)
    saved = np.load(out / "runs" / f"{regime}_seed{seed}" / "val_logits.npy")
    max_diff = float(np.abs(val_logits - saved).max())
    f1_cpu = C.macro_f1(val_ds.targets, val_logits)
    md.append(f"\nКонтроль: макс. |різниця| логітів validation (CPU проти збережених) = {max_diff:.2e}; "
              f"macro-F1 на CPU = {f1_cpu:.4f}.")

    test_ds = C.SplitDataset(split["test"], "test", data_dir, train=False)
    logits = C.predict_logits(model, C.eval_loader(test_ds), device)
    y_true = test_ds.targets
    y_pred = logits.argmax(1)
    probs = torch.softmax(torch.from_numpy(logits), 1).numpy()
    f1_test = C.macro_f1(y_true, logits)
    acc = float((y_pred == y_true).mean())
    n = len(y_true)
    se = math.sqrt(acc * (1 - acc) / n)
    cm = np.zeros((6, 6), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    np.save(res_dir / "test_logits.npy", logits.astype(np.float32))
    plot_confusion(cm, fig_dir / "confusion_matrix.png")

    md += ["\n## Тестова оцінка\n",
           f"macro-F1 на test = **{f1_test:.4f}**; accuracy = {acc:.4f}; n = {n}.",
           f"Вибраний запуск: F на validation = {sel_f:.4f}; різниця test − validation = "
           f"{f1_test - sel_f:+.4f}.",
           f"Стандартна похибка частки правильних відповідей √(p(1−p)/n) = {se:.4f}.",
           "\n### Матриця плутанини (рядки - справжній клас, стовпці - передбачений)\n",
           "| справжній \\ передбачений | " + " | ".join(C.CLASSES) + " |",
           "|---|" + "---|" * 6]
    for i, c in enumerate(C.CLASSES):
        md.append(f"| **{c}** | " + " | ".join(str(int(v)) for v in cm[i]) + " |")

    # 6. п'ять помилок
    errs = pick_errors(split["test"], y_true, y_pred, probs)
    md += ["\n### П'ять характерних помилок\n",
           "| Файл | Справжній клас | Передбачений клас | Імовірність передбаченого класу | "
           "Кількість таких помилок у test |", "|---|---|---|---|---|"]
    for r in errs:
        md.append(f"| {r['file']} | {r['true']} | {r['pred']} | {r['prob']:.4f} | {r['count']} |")
    if errs:
        fig, axes = plt.subplots(1, len(errs), figsize=(3.3 * len(errs), 3.9))
        by_name = {f: i for i, (f, _) in enumerate(split["test"])}
        for ax, r in zip(np.atleast_1d(axes), errs):
            ax.imshow(test_ds.image(by_name[r["file"]]))
            ax.set_title(f"{r['file']}\nсправжній: {r['true']}\nпередбачено: {r['pred']} "
                         f"(p={r['prob']:.2f})", fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(fig_dir / "top5_errors.png", dpi=140, bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)

    text = "\n".join(md) + "\n"
    (res_dir / "analysis.md").write_text(text, encoding="utf-8")
    C.dump_json({"runs": results, "effects": e, "selected": {"regime": regime, "seed": seed},
                 "means": means, "test": {"macro_f1": f1_test, "accuracy": acc, "n": n, "se": se,
                                          "confusion": cm.tolist()}, "errors": errs,
                 "val_check_max_abs_diff": max_diff}, res_dir / "analysis.json")
    print(text)


if __name__ == "__main__":
    main()
