"""Команда навчання ЛР3: поділ, перевірки пунктів 1-2, сім запусків, журнали,
вибір конфігурації й збереження lab3_model.pt. Тестова вибірка тут НЕ використовується.

    uv run python train.py --data-dir <шлях до теки, де лежать seg_train і seg_test>
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

import common as C
import checks

RUNS = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
        ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]


def describe_settings(regime: str, model, opt) -> dict:
    cfg = C.REGIMES[regime]
    total, trainable = C.count_params(model)
    defaults = {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt.defaults.items()}
    return {
        "transforms": {
            "train": ["RandomResizedCrop(160, scale=(0.7, 1.0))", "RandomHorizontalFlip()",
                      "ToTensor()", "Normalize(mean, std)"],
            "validation_test": ["Resize((160, 160))", "ToTensor()", "Normalize(mean, std)"],
            "img_size": C.IMG_SIZE, "mean": C.MEAN, "std": C.STD,
            "image_open": 'PIL.Image.open(...).convert("RGB")',
        },
        "model": {
            "arch": "torchvision.models.resnet18",
            "weights": "ResNet18_Weights.IMAGENET1K_V1" if cfg["pretrained"] else None,
            "fc": "torch.nn.Linear(512, 6), стандартна ініціалізація",
            "bn_momentum": 0.1, "bn_eps": 1e-5, "dtype": "float32",
            "total_params": total, "trainable_params": trainable,
        },
        "regime": {
            "name": regime, "trainable_modules": cfg["trainable"],
            "eval_modules_during_training": cfg["eval_modules"],
            "param_groups": [{"modules": m, "lr": lr, "n_params": sum(
                p.numel() for mm in m for p in getattr(model, mm).parameters())}
                for m, lr in cfg["groups"]],
        },
        "optimizer": {"name": "torch.optim.AdamW", "weight_decay": C.WEIGHT_DECAY,
                      "defaults_after_construction": defaults},
        "loss": "CrossEntropyLoss(), середнє за пакетом",
        "loaders": {
            "train": {"batch_size": C.TRAIN_BS, "shuffle": True, "drop_last": False,
                      "num_workers": 0, "generator": None, "sampler": None},
            "validation": {"batch_size": C.EVAL_BS, "shuffle": False, "num_workers": 0},
        },
        "budget": {"epochs": C.EPOCHS, "steps_per_epoch": 38, "total_steps": C.EPOCHS * 38},
        "not_used": ["lr scheduler", "gradient clipping", "mixed precision", "label smoothing",
                     "class weights", "early stopping", "checkpoint selection by validation"],
        "metric": "sklearn.metrics.f1_score(average='macro') на validation після епохи 8",
    }


def train_run(regime, seed, split, data_dir, device, out_dir, split_hash, epochs, force) -> dict:
    run_dir = out_dir / "runs" / f"{regime}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    jpath = run_dir / "run.json"
    if (not force and jpath.exists() and (run_dir / "weights.pt").exists()
            and (run_dir / "val_logits.npy").exists()):
        old = json.loads(jpath.read_text(encoding="utf-8"))
        if old["split_sha256"] == split_hash and len(old["epochs"]) == epochs:
            print(f"[{regime} seed {seed}] журнал уже є - пропускаю (використайте --force)")
            return old

    print(f"[{regime} seed {seed}] старт на {device}")
    model, opt = C.build(regime, seed, device)
    settings = describe_settings(regime, model, opt)
    train_ds = C.SplitDataset(split["train"], "train", data_dir, train=True)
    val_ds = C.SplitDataset(split["validation"], "validation", data_dir, train=False)
    tl, vl = C.train_loader(train_ds), C.eval_loader(val_ds)
    criterion = torch.nn.CrossEntropyLoss()

    log, steps, t_train = [], 0, 0.0
    t_all = time.perf_counter()
    logits = None
    for ep in range(1, epochs + 1):
        C.sync(device)
        t0 = time.perf_counter()
        C.set_train_mode(model, regime)
        running = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)   # втрата пакета до кроку оптимізатора
            steps += 1
        C.sync(device)
        t_train += time.perf_counter() - t0
        train_loss = running / len(train_ds)
        logits = C.predict_logits(model, vl, device)
        f1 = C.macro_f1(val_ds.targets, logits)
        log.append({"epoch": ep, "train_loss": train_loss, "val_macro_f1": f1})
        print(f"[{regime} seed {seed}] епоха {ep}/{epochs}: train loss {train_loss:.4f}, "
              f"val macro-F1 {f1:.4f}, {time.perf_counter() - t_all:.0f} c")
    total_time = time.perf_counter() - t_all
    if epochs == C.EPOCHS:
        assert steps == 304, steps

    np.save(run_dir / "val_logits.npy", logits.astype(np.float32))
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run_dir / "weights.pt")
    rec = {"regime": regime, "seed": seed, "settings": settings, "optimizer_steps": steps,
           "git": C.git_info(), "split_sha256": split_hash, "versions": C.versions(),
           "device": C.device_name(device), "epochs": log,
           "final": {"train_loss": log[-1]["train_loss"], "val_macro_f1": log[-1]["val_macro_f1"]},
           "train_time_sec": t_train, "total_time_sec_with_validation": total_time}
    C.dump_json(rec, jpath)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="тека, у якій лежать seg_train/seg_train і seg_test/seg_test")
    ap.add_argument("--out-dir", type=Path, default=C.HERE, help="куди писати результати")
    ap.add_argument("--force", action="store_true", help="перенавчити, навіть якщо журнали є")
    ap.add_argument("--smoke", action="store_true",
                    help="лише налагодження конвеєра (1 епоха, без жорсткої перевірки поділу); "
                         "результати НЕ для звіту")
    args = ap.parse_args()
    out, data_dir = args.out_dir.resolve(), args.data_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    epochs = 1 if args.smoke else C.EPOCHS
    device = C.get_device()
    print(f"Пристрій: {C.device_name(device)}; torch {torch.__version__}")

    # --- поділ
    rows = C.build_split(data_dir)
    split_path = out / "split.csv"
    C.write_split(rows, split_path)
    split_hash = C.sha256_file(split_path)
    split = C.read_split(split_path)
    print(f"split.csv: {len(rows)} рядків, sha256 = {split_hash}")

    # --- перевірки пунктів 1-2
    checks.run_all_checks(split, data_dir, device, out, allow_mismatch=args.smoke)

    # --- сім запусків
    results = []
    for regime, seed in RUNS:
        rec = train_run(regime, seed, split, data_dir, device, out, split_hash, epochs, args.force)
        results.append({"regime": regime, "seed": seed, "f1": rec["final"]["val_macro_f1"]})

    # --- вибір за validation і збереження lab3_model.pt
    regime, seed, means = C.select_best(results)
    print(f"Середні validation macro-F1 за режимами: {means}")
    print(f"Вибрано: {regime}, зерно {seed}")
    sd = torch.load(out / "runs" / f"{regime}_seed{seed}" / "weights.pt", map_location="cpu")
    torch.save({"state_dict": sd, "regime": regime, "seed": seed, "classes": C.CLASSES,
                "img_size": C.IMG_SIZE}, out / "lab3_model.pt")
    print("lab3_model.pt збережено. Тест не використовувався; далі: analyze.py")


if __name__ == "__main__":
    main()
