"""Перевірки пунктів 1-2: поділ, пакет після перетворень, початкова втрата,
параметри/групи оптимізатора, що змінюється під час навчання, дослід із BN."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CLASSES, EXPECTED_SPLIT, EXPECTED_TOTAL_PARAMS, EXPECTED_TOTALS, MEAN, PARTS,
                    REGIMES, STD, TOP_MODULES, SplitDataset, build, count_params, set_train_mode,
                    split_control, train_loader)

# Прогноз для досліду з навмисною помилкою, записаний ДО запуску досліду.
BN_PREDICTION = (
    "Прогноз (записано до запуску): у frozen з model.train() параметри кістяка НЕ зміняться "
    "(requires_grad=False, в оптимізаторі їх немає), але ковзні статистики running_mean, "
    "running_var і лічильник num_batches_tracked УСІХ BN-шарів кістяка оновляться, бо BN у режимі "
    "train оновлює їх незалежно від requires_grad (з momentum=0.1 за 5 кроків ~41% старої "
    "статистики замінюється статистикою нових даних). Голова fc навчатиметься як і раніше. "
    "На кривій втрати на train це майже не буде помітно: у режимі train BN нормує за статистикою "
    "самого пакета, тому втрата на train лишається схожою на правильну; наслідок проявиться лише "
    "в eval(), де використовуються вже зміщені ковзні статистики."
)


# ------------------------------------------------------------ 1. поділ
def check_split(split) -> dict:
    table = split_control(split)
    cells, all_ok = [], True
    for part in PARTS:
        for label in CLASSES:
            got, exp = table[(label, part)], EXPECTED_SPLIT[(label, part)]
            ok = got == exp
            all_ok &= ok
            cells.append(dict(cls=label, part=part, got=list(got), expected=list(exp), ok=ok))
    totals = {}
    for part in PARTS:
        got = (sum(table[(l, part)][0] for l in CLASSES), sum(table[(l, part)][1] for l in CLASSES))
        ok = got == EXPECTED_TOTALS[part]
        all_ok &= ok
        totals[part] = dict(got=list(got), expected=list(EXPECTED_TOTALS[part]), ok=ok)
    names = {p: {f for f, _ in split[p]} for p in PARTS}
    overlaps = {f"{a}&{b}": sorted(names[a] & names[b])
                for i, a in enumerate(PARTS) for b in PARTS[i + 1:]}
    n_rows = sum(len(split[p]) for p in PARTS)
    n_unique = len(set().union(*names.values()))
    disjoint = all(not v for v in overlaps.values()) and n_rows == n_unique
    return dict(cells=cells, totals=totals, all_match=all_ok, disjoint=disjoint,
                n_rows=n_rows, n_unique_names=n_unique)


# ------------------------------------------------------- 1. пакет після перетворень
def save_batch_figure(split, data_dir: Path, path: Path) -> None:
    ds = SplitDataset(split["train"], "train", data_dir, train=True)
    torch.manual_seed(1)
    x, y = next(iter(train_loader(ds)))
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    img = (x[:16] * std + mean).clamp(0, 1)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.8))
    for ax, im, lab in zip(axes.ravel(), img, y[:16]):
        ax.imshow(im.permute(1, 2, 0).numpy())
        ax.set_title(CLASSES[int(lab)], fontsize=10)
        ax.axis("off")
    fig.suptitle("Перші 16 зображень навчального пакета після аугментацій (зерно 1)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------ 1. початкова втрата
@torch.no_grad()
def initial_loss(regime: str, split, data_dir: Path, device) -> float:
    model, _ = build(regime, 1, device)
    ds = SplitDataset(split["train"], "train", data_dir, train=True)
    loader = train_loader(ds)
    set_train_mode(model, regime)
    x, y = next(iter(loader))
    loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
    return float(loss.detach())


# ------------------------------------------------------------ 2. параметри
def check_params(regime: str, device) -> dict:
    model, opt = build(regime, 1, device)
    total, trainable = count_params(model)
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    frozen_in_opt = [n for n, p in model.named_parameters()
                     if not p.requires_grad and id(p) in in_opt]
    trainable_not_in_opt = [n for n, p in model.named_parameters()
                            if p.requires_grad and id(p) not in in_opt]
    groups = [dict(n_params=sum(p.numel() for p in g["params"]), lr=g["lr"],
                   weight_decay=g["weight_decay"]) for g in opt.param_groups]
    exp = REGIMES[regime]["expected_trainable"]
    return dict(regime=regime, total=total, trainable=trainable, groups=groups,
                expected_total=EXPECTED_TOTAL_PARAMS, expected_trainable=exp,
                counts_match=(total == EXPECTED_TOTAL_PARAMS and trainable == exp),
                frozen_params_in_optimizer=frozen_in_opt,
                trainable_params_missing_in_optimizer=trainable_not_in_opt,
                groups_match=(sum(g["n_params"] for g in groups) == exp))


# ------------------------------------------------------ 2. що змінюється під час навчання
def state_change_check(regime: str, split, data_dir: Path, device, correct_mode: bool,
                       n_steps: int = 5) -> dict:
    """5 кроків на перших 5 пакетах. correct_mode=True - set_train_mode, False - model.train()."""
    model, opt = build(regime, 1, device)
    ds = SplitDataset(split["train"], "train", data_dir, train=True)
    loader = train_loader(ds)
    before = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    if correct_mode:
        set_train_mode(model, regime)
    else:
        model.train()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i == n_steps:
            break
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    after = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    param_names = {n for n, _ in model.named_parameters()}
    expected_frozen = set(REGIMES[regime]["eval_modules"])
    rows, all_ok = [], True
    for top in TOP_MODULES:
        pk = [k for k in after if k.split(".")[0] == top and k in param_names]
        sk = [k for k in after if k.split(".")[0] == top and k not in param_names]
        p_changed = any(not torch.equal(before[k], after[k]) for k in pk)
        has_bn = len(sk) > 0
        s_changed = any(not torch.equal(before[k], after[k]) for k in sk) if has_bn else None
        if top in expected_frozen:
            ok = (not p_changed) and (not s_changed)
        else:
            ok = p_changed and (s_changed is None or s_changed)
        all_ok &= ok
        rows.append(dict(module=top, params_changed=p_changed, stats_changed=s_changed,
                         has_bn=has_bn, as_expected=ok))

    key = "layer1.0.bn1"
    d_mean = (after[f"{key}.running_mean"] - before[f"{key}.running_mean"]).abs().max().item()
    d_var = (after[f"{key}.running_var"] - before[f"{key}.running_var"]).abs().max().item()
    nbt = int(after[f"{key}.num_batches_tracked"])
    return dict(regime=regime, correct_mode=correct_mode, rows=rows, all_as_expected=all_ok,
                losses=losses, layer1_0_bn1_max_abs_d_running_mean=d_mean,
                layer1_0_bn1_max_abs_d_running_var=d_var, layer1_0_bn1_num_batches_tracked=nbt)


def run_all_checks(split, data_dir: Path, device, out_dir: Path, allow_mismatch: bool) -> dict:
    fig_dir, chk_dir = out_dir / "figures", out_dir / "checks"
    fig_dir.mkdir(exist_ok=True)
    chk_dir.mkdir(exist_ok=True)
    res: dict = {}

    print("== Перевірка 1: поділ ==")
    res["split"] = check_split(split)
    print(f"   усі 18 клітинок + підсумки збігаються: {res['split']['all_match']}; "
          f"частини не перетинаються: {res['split']['disjoint']}")
    if not (res["split"]["all_match"] and res["split"]["disjoint"]) and not allow_mismatch:
        raise SystemExit("Поділ не збігається з контрольною таблицею - це помилка реалізації. "
                         "Зупиняюсь (для налагодження на чужих даних є --smoke).")

    print("== Перевірка 1: пакет після перетворень ==")
    save_batch_figure(split, data_dir, fig_dir / "train_batch.png")

    print("== Перевірка 1: початкова втрата L0 ==")
    ln6 = math.log(6)
    res["initial_loss"] = {"ln6": ln6}
    for regime in ("scratch", "frozen"):
        l0 = initial_loss(regime, split, data_dir, device)
        res["initial_loss"][regime] = l0
        print(f"   {regime}: L0 = {l0:.4f} (ln 6 = {ln6:.4f})")

    print("== Перевірка 2: параметри і групи оптимізатора ==")
    res["params"] = {}
    for regime in REGIMES:
        r = check_params(regime, device)
        res["params"][regime] = r
        print(f"   {regime}: всього {r['total']}, requires_grad {r['trainable']}, групи "
              f"{[(g['n_params'], g['lr']) for g in r['groups']]}, збігається: {r['counts_match']}, "
              f"заморожених в оптимізаторі: {len(r['frozen_params_in_optimizer'])}")
        if not (r["counts_match"] and r["groups_match"] and not r["frozen_params_in_optimizer"]
                and not r["trainable_params_missing_in_optimizer"]) and not allow_mismatch:
            raise SystemExit(f"Перевірка параметрів режиму {regime} не пройшла.")

    print("== Перевірка 2: що змінюється під час навчання (set_train_mode) ==")
    res["state_change"] = {}
    for regime in ("frozen", "unfreeze"):
        r = state_change_check(regime, split, data_dir, device, correct_mode=True)
        res["state_change"][regime] = r
        print(f"   {regime}: усі модулі як очікувано: {r['all_as_expected']}")
        if not r["all_as_expected"] and not allow_mismatch:
            raise SystemExit(f"Перевірка станів режиму {regime} не пройшла.")

    print("== Дослід із навмисною помилкою (frozen, model.train()) ==")
    print("   " + BN_PREDICTION)
    res["bn_prediction"] = BN_PREDICTION
    bug = state_change_check("frozen", split, data_dir, device, correct_mode=False)
    res["bn_bug"] = bug
    for r in bug["rows"]:
        print(f"   {r['module']:7s} params_changed={r['params_changed']} "
              f"stats_changed={r['stats_changed']}")
    print(f"   layer1.0.bn1: max|d running_mean| = {bug['layer1_0_bn1_max_abs_d_running_mean']:.3e}, "
          f"max|d running_var| = {bug['layer1_0_bn1_max_abs_d_running_var']:.3e}")
    print(f"   втрати 5 кроків (помилка): {np.round(bug['losses'], 4).tolist()}")
    print(f"   втрати 5 кроків (правильно): "
          f"{np.round(res['state_change']['frozen']['losses'], 4).tolist()}")

    from common import dump_json
    dump_json(res, chk_dir / "checks.json")
    (chk_dir / "checks.md").write_text(render_md(res), encoding="utf-8")
    return res


def _yn(v):
    return "немає BN" if v is None else ("так" if v else "ні")


def render_md(res: dict) -> str:
    L = ["# Результати перевірок (пункти 1–2)\n", "## Поділ\n",
         "| Клас | train | validation | test |", "|---|---|---|---|"]
    cell = {(c["cls"], c["part"]): c for c in res["split"]["cells"]}
    for cl in CLASSES:
        L.append(f"| {cl} | " + " | ".join(
            f"{cell[(cl, p)]['got'][0]} / {cell[(cl, p)]['got'][1]:,}".replace(",", " ")
            + (" ✓" if cell[(cl, p)]["ok"] else " ✗") for p in PARTS) + " |")
    L.append("| **Разом** | " + " | ".join(
        f"{res['split']['totals'][p]['got'][0]} / {res['split']['totals'][p]['got'][1]:,}".replace(",", " ")
        + (" ✓" if res["split"]["totals"][p]["ok"] else " ✗") for p in PARTS) + " |")
    L.append(f"\nУсі числа збігаються: **{res['split']['all_match']}**; "
             f"частини не перетинаються: **{res['split']['disjoint']}**.\n")
    il = res["initial_loss"]
    L += ["## Початкова втрата\n", f"ln 6 = {il['ln6']:.4f}; scratch: {il['scratch']:.4f}; "
          f"frozen: {il['frozen']:.4f}\n", "## Параметри\n",
          "| Режим | Усього | requires_grad=True | Групи (параметрів, lr) | Заморожених в оптимізаторі |",
          "|---|---|---|---|---|"]
    for r, p in res["params"].items():
        gs = "; ".join(f"{g['n_params']} @ {g['lr']:g}" for g in p["groups"])
        L.append(f"| {r} | {p['total']} | {p['trainable']} | {gs} | "
                 f"{len(p['frozen_params_in_optimizer'])} |")
    for title, items in (("Що змінюється (set_train_mode)", res["state_change"]),
                         ("Дослід із помилкою (frozen, model.train())", {"frozen": res["bn_bug"]})):
        L += [f"\n## {title}\n", "| Режим | Модуль | Параметри змінилися | Ковзні статистики BN змінилися | Збігається з очікуваним |",
              "|---|---|---|---|---|"]
        for reg, r in items.items():
            for m in r["rows"]:
                ok = m["as_expected"] if r["correct_mode"] else "—"
                L.append(f"| {reg} | {m['module']} | {_yn(m['params_changed'])} | "
                         f"{_yn(m['stats_changed'])} | {_yn(ok) if ok != '—' else '—'} |")
    b = res["bn_bug"]
    L += [f"\nlayer1.0.bn1 після 5 кроків: max|Δ running_mean| = {b['layer1_0_bn1_max_abs_d_running_mean']:.4e}, "
          f"max|Δ running_var| = {b['layer1_0_bn1_max_abs_d_running_var']:.4e}, "
          f"num_batches_tracked = {b['layer1_0_bn1_num_batches_tracked']}\n",
          f"Втрати 5 кроків: з помилкою {np.round(b['losses'], 4).tolist()}, "
          f"правильно {np.round(res['state_change']['frozen']['losses'], 4).tolist()}\n"]
    return "\n".join(L)
