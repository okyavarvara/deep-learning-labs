"""Перетворює lab4_results.json на готові markdown-таблиці (lab4_tables.md) для README.
Усі числа беруться лише з результатів запуску, нічого не вводиться вручну."""
from __future__ import annotations

import math


def _sp(x: int) -> str:
    return f"{x:,}".replace(",", " ")


def render_tables(res: dict) -> str:
    L: list[str] = []
    sc, cc = res["split_control"], res["calibration_control"]
    L += ["## Контроль поділу і калібрування\n", "| Частина | Кількість | Сума імен | Збігається |", "|---|---|---|---|"]
    for part in ("train", "test"):
        v = sc[part]
        L.append(f"| {part} | {v['count']} | {_sp(v['name_sum'])} | {'так' if v['ok'] else 'ні'} |")
    L.append(f"| калібрування | {cc['n']} | {_sp(cc['sum'])} | {'так' if cc['ok'] else 'ні'} |")
    L.append("\nКалібрування за класами: " + ", ".join(f"{k} {v}" for k, v in cc["per_class"].items()))
    L.append(f"\nSHA-256 split.csv: `{res['split_sha256']}`; модель: режим {res['model']['regime']}, "
             f"зерно {res['model']['seed']}.\n")

    e = res["export"]
    L += ["## Пункт 1. Експорт і відповідність\n", "| Величина | Значення |", "|---|---|",
          f"| max abs. різниця логітів | {e['max_abs_logit_diff']:.3e} |",
          f"| збіг передбачених класів | {e['class_agreement']:.4f} |",
          f"| macro-F1 PyTorch | {e['macro_f1_pytorch']:.4f} |",
          f"| macro-F1 ONNX FP32 | {e['macro_f1_onnx_fp32']:.4f} |",
          f"| вузлів Conv | {e['nodes']['Conv']} |",
          f"| вузлів BatchNormalization | {e['nodes']['BatchNormalization']} |",
          f"| час експорту, с | {e['export_time_s']:.2f} |"]
    q = res["quantization"]
    L += ["\n## Пункт 2. Час квантизації\n", "| Етап | Час, с |", "|---|---|",
          f"| quant_pre_process | {q['pre_process_time_s']:.2f} |",
          f"| quantize_static (калібрування + квантизація) | {q['quantize_static_time_s']:.2f} |",
          f"| разом | {q['quantization_total_time_s']:.2f} |"]

    ql = res["quality"]
    L += ["\n## Пункт 3. Якість\n",
          "| Варіант | macro-F1 | accuracy | Згода з FP32 | b | c | Δacc ± SE |", "|---|---|---|---|---|---|---|"]
    for name, key in (("FP32", "fp32"), ("INT8", "int8")):
        r = ql[key]
        L.append(f"| {name} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['agreement']:.4f} | "
                 f"{r['b']} | {r['c']} | {r['delta_acc']:+.4f} ± {r['se']:.4f} |")
    r = ql["int8"]
    L += [f"\n|b − c| = {r['abs_b_minus_c']}, 2√(b+c) = {r['threshold_2sqrt_bc']:.2f}: **{r['verdict']}**.",
          f"\nStandard error одної моделі √(p(1−p)/n) = {ql['se_single_model_fp32']:.4f}; "
          f"SE парної різниці = {r['se']:.4f}; відношення = "
          + (f"{ql['se_single_model_fp32'] / r['se']:.1f}×." if r["se"] else "не визначено (b + c = 0)."),
          f"\nЗмінили клас відносно FP32: {r['n_flipped']} зображень (з них обидві відповіді хибні: "
          f"{r['n_flipped_both_wrong']}; INT8 виправила {r['b']}; INT8 зіпсувала {r['c']})."]

    pe = res["per_tensor_experiment"]
    rg = pe["ranges"]
    L += ["\n## Пункт 4. Діапазони каналів і потензорний дослід\n",
          "| Згортка | r = max α / median α | log₂ r, біт |", "|---|---|---|"]
    for x in rg["top3"]:
        L.append(f"| `{x['node']}` | {x['r']:.2f} | {math.log2(x['r']):.2f} |")
    L.append(f"| **медіана за {rg['n_convs']} згортками** | **{rg['r_median']:.2f}** | **{rg['median_cost_bits']:.2f}** |")
    L += ["\n| Варіант | macro-F1 | accuracy | Згода з FP32 | b | c | Δacc ± SE |", "|---|---|---|---|---|---|---|"]
    for name, r in (("INT8 per-channel", ql["int8"]), ("INT8 per-tensor", pe["per_tensor"])):
        L.append(f"| {name} | {r['macro_f1']:.4f} | {r['accuracy']:.4f} | {r['agreement']:.4f} | "
                 f"{r['b']} | {r['c']} | {r['delta_acc']:+.4f} ± {r['se']:.4f} |")
    r = pe["per_tensor"]
    L.append(f"\nPer-tensor: |b − c| = {r['abs_b_minus_c']}, 2√(b+c) = {r['threshold_2sqrt_bc']:.2f}: "
             f"**{r['verdict']}**; змінили клас {r['n_flipped']} зображень (обидві відповіді хибні: "
             f"{r['n_flipped_both_wrong']}).")

    lt, env = res["latency"], res["environment"]
    L += ["\n## Пункт 5. Затримка\n", "| Варіант | Файл, МБ | p50, мс | p95, мс | p50 за раундами, мс |", "|---|---|---|---|---|"]
    for name, key in (("FP32", "fp32"), ("INT8", "int8")):
        v = lt[key]
        L.append(f"| {name} | {v['file_mb']:.2f} | {v['p50_ms']:.3f} | {v['p95_ms']:.3f} | "
                 + ", ".join(f"{t:.3f}" for t in v["p50_by_round_ms"]) + " |")
    c = lt["comparison"]
    L += [f"\nРозмір FP32/INT8 = {c['size_ratio_fp32_over_int8']:.2f}×; прискорення p50 = {c['speedup_p50']:.2f}; "
          f"прискорення p95 = {c['speedup_p95']:.2f}.",
          f"\nВідрізки p50 за раундами: FP32 [{lt['fp32']['p50_round_min']:.3f}; {lt['fp32']['p50_round_max']:.3f}], "
          f"INT8 [{lt['int8']['p50_round_min']:.3f}; {lt['int8']['p50_round_max']:.3f}]: **{c['verdict']}**.",
          "\n## Середовище\n", "| Параметр | Значення |", "|---|---|",
          f"| CPU | {env['cpu']} |", f"| Логічних ядер | {env['logical_cpus']} |", f"| ОС | {env['os']} |",
          f"| Python | {env['python']} |", f"| PyTorch | {env['torch']} |", f"| ONNX | {env['onnx']} |",
          f"| ONNX Runtime | {env['onnxruntime']} |", f"| Потоків (intra/inter) | {env['threads']}/{env['threads']} |",
          f"| Розмір пакета | {env['batch']} |",
          "| Цілочисельні SIMD | " + ", ".join(f"{k}={'так' if v else 'ні' if v is False else '?'}"
                                              for k, v in env["cpu_flags"].items()) + " |"]
    return "\n".join(L) + "\n"
