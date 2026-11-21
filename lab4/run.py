"""Єдина команда ЛР4: експорт, перевірка відповідності, калібрування, квантизація, якість,
дослід із потензорними вагами, затримка, запис lab4_results.json і lab4_tables.md.

    uv run python run.py --data-dir <шлях до теки, де лежать seg_train і seg_test> \
                         [--model ../lab3/lab3_model.pt] [--split ../lab3/split.csv]

Файли .onnx пишуться в --out-dir (за замовчуванням - тека lab4) і до репозиторію не додаються.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

import common as C
import export as E
import latency as L
import quantize as Q

# Прогноз для досліду з потензорними вагами. Записано ДО запуску досліду: цей текст друкується
# перед побудовою lab4_int8_per_tensor.onnx і потрапляє в lab4_results.json без змін.
PER_TENSOR_PREDICTION = (
    "Прогноз (записано до запуску): потензорний крок коштує каналу c приблизно log2(max_c alpha_c / alpha_c) "
    "біта. Для медіанної згортки r = 3,44, тобто 1,78 біта (8 - 1,78 ≈ 6,2 біта лишається типовому каналу), "
    "а для найгіршої згортки conv1 r = 6,67, тобто 2,74 біта. Це далеко від MobileNetV3, де r ≈ 90 (≈ 6,5 біта), "
    "тому катастрофи не очікую. Згода передбачень з FP32 ВПАДЕ порівняно з per-channel INT8, бо кожне "
    "додаткове округлення зсуває логіти й перекидає найнепевніші зображення біля меж класів, але лишиться "
    "високою: оцінюю 0,97-0,99 (3-45 зображень із 1502 змінять клас) і в усякому разі вище 0,95. "
    "macro-F1 за правилом пункту 3 НЕ зміниться понад шум (|b - c| <= 2 sqrt(b + c)): зміни класу "
    "відбуваються в обидва боки (було правильно -> стало неправильно і навпаки), тому b і c майже "
    "компенсуються, а різниця accuracy лишається в межах кількох зображень."
)


class Tee:
    """Дублює вивід у файл логу."""

    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="тека, у якій лежать seg_train/seg_train і seg_test/seg_test")
    ap.add_argument("--model", type=Path, default=C.HERE.parent / "lab3" / "lab3_model.pt",
                    help="lab3_model.pt з ЛР3")
    ap.add_argument("--split", type=Path, default=C.HERE.parent / "lab3" / "split.csv",
                    help="split.csv з ЛР3")
    ap.add_argument("--out-dir", type=Path, default=C.HERE, help="куди писати .onnx, результати, лог")
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(out / "lab4_log.txt")
    t_start = time.perf_counter()

    res: dict = {"git": C.git_info(), "versions": C.versions(),
                 "paths": {"model": str(args.model), "split": str(args.split),
                           "data_dir": str(args.data_dir)}}
    fp32, prep = out / "lab4_fp32.onnx", out / "lab4_fp32_prep.onnx"
    int8, int8_pt = out / "lab4_int8.onnx", out / "lab4_int8_per_tensor.onnx"

    # ------------------------------------------------------------ вхідні дані
    print("== Вхідні дані ==")
    split = C.read_split(args.split)
    res["split_sha256"] = C.sha256_file(args.split)
    res["split_control"] = C.check_split(split)
    for part in ("train", "test"):
        v = res["split_control"][part]
        print(f"   {part}: {v['count']} / {v['name_sum']} (очікується {v['expected_count']} / "
              f"{v['expected_name_sum']}): {'OK' if v['ok'] else 'РОЗБІЖНІСТЬ'}")
    if not res["split_control"]["all_ok"]:
        raise SystemExit("Поділ не збігається з контрольною таблицею - виправте це в ЛР3 до початку ЛР4.")

    model, meta = C.load_model(args.model)
    res["model"] = meta
    print(f"   модель: {meta}")

    calib_items = C.calibration_rows(split)
    res["calibration_control"] = C.check_calibration(calib_items)
    cc = res["calibration_control"]
    print(f"   калібрування: n = {cc['n']}, сума імен = {cc['sum']}, за класами {cc['per_class']}: "
          f"{'OK' if cc['ok'] else 'РОЗБІЖНІСТЬ'}")
    if not cc["ok"]:
        raise SystemExit("Калібрувальна вибірка не збігається з контрольними числами - помилка реалізації.")

    test_items = split["test"]
    y_true = C.targets_of(test_items)
    t0 = time.perf_counter()
    x_test = C.load_part(test_items, "test", args.data_dir)
    x_calib = C.load_part(calib_items, "train", args.data_dir)
    print(f"   завантажено test {x_test.shape} і калібрування {x_calib.shape} за "
          f"{time.perf_counter() - t0:.0f} с")

    # ---------------------------------------------- 1. експорт і відповідність
    print("== Пункт 1: експорт і відповідність ==")
    t_export = E.export_fp32(model, fp32)
    counts = E.node_counts(fp32)
    z_pt = C.pytorch_logits(model, x_test)
    sess_fp32 = C.make_session(fp32)
    z_onnx = C.onnx_logits(sess_fp32, x_test)
    max_diff = float(np.abs(z_onnx - z_pt).max())
    pred_pt, pred_onnx = z_pt.argmax(1), z_onnx.argmax(1)
    res["export"] = {
        "export_time_s": t_export, "opset": C.OPSET, "file_bytes": fp32.stat().st_size,
        "max_abs_logit_diff": max_diff, "logit_diff_limit": C.LOGIT_DIFF_LIMIT,
        "logit_diff_ok": max_diff < C.LOGIT_DIFF_LIMIT,
        "class_agreement": float((pred_pt == pred_onnx).mean()),
        "macro_f1_pytorch": C.macro_f1(y_true, pred_pt), "macro_f1_onnx_fp32": C.macro_f1(y_true, pred_onnx),
        "nodes": counts}
    e = res["export"]
    print(f"   час експорту {t_export:.2f} с; max|dz| = {max_diff:.3e}; збіг класів "
          f"{e['class_agreement']:.4f}; F1 PyTorch {e['macro_f1_pytorch']:.4f}, ONNX {e['macro_f1_onnx_fp32']:.4f}; "
          f"вузлів Conv {counts['Conv']}, BatchNormalization {counts['BatchNormalization']}")
    if not e["logit_diff_ok"]:
        raise SystemExit("max|dz| >= 1e-3: ONNX не відтворює модель PyTorch - помилка реалізації.")

    # ------------------------------------------ 2. калібрування й квантизація
    print("== Пункт 2: калібрування і квантизація (per-channel) ==")
    t_prep = Q.preprocess(fp32, prep)
    t_quant = Q.quantize(prep, int8, x_calib, per_channel=True)
    res["quantization"] = {"pre_process_time_s": t_prep, "quantize_static_time_s": t_quant,
                           "quantization_total_time_s": t_prep + t_quant,
                           "settings": {"quant_format": "QDQ", "activation_type": "QUInt8",
                                        "weight_type": "QInt8", "per_channel": True,
                                        "calibrate_method": "MinMax",
                                        "calibration_images": len(x_calib),
                                        "reader_batch": 1}}
    print(f"   quant_pre_process {t_prep:.2f} с; quantize_static {t_quant:.2f} с")

    # ------------------------------------------------------------ 3. якість
    print("== Пункт 3: якість ==")
    sess_int8 = C.make_session(int8)
    z_int8 = C.onnx_logits(sess_int8, x_test)
    n = len(y_true)
    row_fp32 = C.quality_row(y_true, pred_onnx, pred_onnx)
    row_int8 = C.quality_row(y_true, z_int8.argmax(1), pred_onnx)
    p = row_fp32["accuracy"]
    se_single = float(np.sqrt(p * (1 - p) / n))
    res["quality"] = {"n": n, "fp32": row_fp32, "int8": row_int8,
                      "se_single_model_fp32": se_single,
                      "se_pair_over_se_single": row_int8["se"] / se_single if row_int8["se"] else None}
    for name, r in (("FP32", row_fp32), ("INT8", row_int8)):
        print(f"   {name}: F1 {r['macro_f1']:.4f}, acc {r['accuracy']:.4f}, згода {r['agreement']:.4f}, "
              f"b {r['b']}, c {r['c']}, dAcc {r['delta_acc']:+.4f} ± {r['se']:.4f}; {r['verdict']}")

    # ------------------------------------------ 4. дослід із потензорними вагами
    print("== Пункт 4: діапазони каналів ==")
    rng = E.conv_ranges(fp32)
    res["per_tensor_experiment"] = {"ranges": rng}
    for x in rng["top3"]:
        print(f"   r = {x['r']:.2f}  {x['node']}")
    print(f"   медіана r за {rng['n_convs']} згортками = {rng['r_median']:.2f} "
          f"(= {rng['median_cost_bits']:.2f} біта)")
    print(PER_TENSOR_PREDICTION)
    res["per_tensor_experiment"]["prediction"] = PER_TENSOR_PREDICTION
    t_q_pt = Q.quantize(prep, int8_pt, x_calib, per_channel=False)
    sess_pt = C.make_session(int8_pt)
    z_pt8 = C.onnx_logits(sess_pt, x_test)
    row_pt = C.quality_row(y_true, z_pt8.argmax(1), pred_onnx)
    res["per_tensor_experiment"].update({"quantize_static_time_s": t_q_pt, "per_tensor": row_pt,
                                         "file_bytes": int8_pt.stat().st_size})
    r = row_pt
    print(f"   per-tensor INT8: F1 {r['macro_f1']:.4f}, acc {r['accuracy']:.4f}, згода {r['agreement']:.4f}, "
          f"b {r['b']}, c {r['c']}, dAcc {r['delta_acc']:+.4f} ± {r['se']:.4f}; {r['verdict']}")

    # ------------------------------------------------------------- 5. затримка
    print("== Пункт 5: затримка ==")
    res["environment"] = {"cpu": C.cpu_name(), "cpu_flags": C.cpu_flags(), "os": C.versions()["platform"],
                          "logical_cpus": os.cpu_count(), **C.versions(),
                          "threads": C.LAT_THREADS, "batch": C.LAT_BATCH,
                          "providers": ["CPUExecutionProvider"], "graph_optimization": "за замовчуванням (ORT_ENABLE_ALL)"}
    print(f"   {res['environment']['cpu']}; {res['environment']['os']}")
    res["latency"] = L.run_protocol({"fp32": fp32, "int8": int8}, x_test[:1].copy())
    lt = res["latency"]
    for v in ("fp32", "int8"):
        print(f"   {v}: {lt[v]['file_mb']:.2f} МБ, p50 {lt[v]['p50_ms']:.3f} мс, p95 {lt[v]['p95_ms']:.3f} мс, "
              f"p50 за раундами {[round(t, 3) for t in lt[v]['p50_by_round_ms']]}")
    print(f"   прискорення p50 = {lt['comparison']['speedup_p50']:.2f}; {lt['comparison']['verdict']}")

    res["total_run_time_s"] = time.perf_counter() - t_start
    C.dump_json(res, out / "lab4_results.json")
    import report
    (out / "lab4_tables.md").write_text(report.render_tables(res), encoding="utf-8")
    print(f"Записано lab4_results.json, lab4_tables.md, lab4_log.txt у {out}")


if __name__ == "__main__":
    main()
