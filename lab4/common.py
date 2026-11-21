"""Спільний код ЛР4: константи, поділ, калібрувальна вибірка, перетворення входу,
завантаження моделі, сесії ONNX Runtime, метрики, відомості про середовище.

Усі фіксовані налаштування з умови лабораторної зібрано тут, щоб export.py, quantize.py,
evaluate.py, latency.py та run.py використовували тільки одні й ті самі значення.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- константи
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EVAL_BS = 64                 # пакет для оцінки якості
OPSET = 17
N_CALIB = 200
CALIB_SEED = 0
INPUT_NAME, OUTPUT_NAME = "input", "logits"

# латентність
LAT_BATCH = 1
LAT_THREADS = 1
LAT_ROUNDS = 3
LAT_WARMUP = 20
LAT_MEASURES = 100

# у якій підтеці набору лежить кожна частина split.csv
SRC_DIR = {"train": Path("seg_train") / "seg_train", "test": Path("seg_test") / "seg_test",
           "validation": Path("seg_test") / "seg_test"}

# контрольні числа з умови: (кількість, сума числових імен файлів)
EXPECTED_SPLIT = {"train": (1200, 11_993_084), "test": (1502, 33_351_444)}
EXPECTED_CALIB = {"n": 200, "sum": 1_921_470,
                  "per_class": {"buildings": 40, "forest": 35, "glacier": 36,
                                "mountain": 33, "sea": 29, "street": 27}}
LOGIT_DIFF_LIMIT = 1e-3      # межа з умови для max|z_ONNX - z_PyTorch|


# --------------------------------------------------------------------- поділ
def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split(path: Path) -> dict[str, list[tuple[str, str]]]:
    """part -> список (file, label) у порядку рядків split.csv."""
    out: dict[str, list[tuple[str, str]]] = {"train": [], "validation": [], "test": []}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["part"]].append((r["file"], r["label"]))
    return out


def name_sum(items) -> int:
    """Сума числових імен файлів (20057.jpg -> 20057)."""
    return sum(int(Path(f).stem) for f, _ in items)


def check_split(split) -> dict:
    """Контроль двох частин поділу (кількість і сума імен). Розбіжність - помилка реалізації."""
    res = {}
    for part, (n_exp, s_exp) in EXPECTED_SPLIT.items():
        got = (len(split[part]), name_sum(split[part]))
        res[part] = {"count": got[0], "name_sum": got[1], "expected_count": n_exp,
                     "expected_name_sum": s_exp, "ok": got == (n_exp, s_exp)}
    res["all_ok"] = all(v["ok"] for v in res.values())
    return res


def calibration_rows(split) -> list[tuple[str, str]]:
    """Калібрувальна вибірка: 1200 train-рядків, відсортованих за іменем як рядки;
    idx = default_rng(0).permutation(1200)[:200]; рядки з номерами idx (у порядку idx)."""
    rows = sorted(split["train"], key=lambda r: r[0])
    assert len(rows) == 1200, len(rows)
    idx = np.random.default_rng(CALIB_SEED).permutation(1200)[:N_CALIB]
    return [rows[i] for i in idx]


def check_calibration(rows) -> dict:
    per_class = {c: sum(1 for _, l in rows if l == c) for c in CLASSES}
    got = {"n": len(rows), "sum": name_sum(rows), "per_class": per_class}
    return {**got, "expected": EXPECTED_CALIB, "ok": got == EXPECTED_CALIB}


# ------------------------------------------------------------ перетворення входу
def load_image_array(path: Path) -> np.ndarray:
    """Resize((160,160)) -> ToTensor() -> Normalize(mean, std); результат float32 3x160x160.
    Це те саме перетворення, що для test у ЛР3 (torchvision), застосоване до PIL-зображення."""
    return _TRANSFORM(Image.open(path).convert("RGB")).numpy().astype(np.float32, copy=False)


def _make_transform():
    from torchvision import transforms as T
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


_TRANSFORM = _make_transform()


def load_part(items, part: str, data_dir: Path) -> np.ndarray:
    """Масив N x 3 x 160 x 160 float32 у порядку items."""
    root = Path(data_dir) / SRC_DIR[part]
    out = np.empty((len(items), 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, (f, label) in enumerate(items):
        out[i] = load_image_array(root / label / f)
    return out


def targets_of(items) -> np.ndarray:
    return np.array([CLASS_TO_IDX[l] for _, l in items])


# ------------------------------------------------------------------ модель
def load_model(path: Path):
    """lab3_model.pt -> resnet18(weights=None), fc = Linear(512, 6), eval()."""
    import torch
    import torchvision
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["classes"] != CLASSES:
        raise SystemExit(f"Порядок класів у файлі моделі {ckpt['classes']} не збігається з {CLASSES}")
    if int(ckpt["img_size"]) != IMG_SIZE:
        raise SystemExit(f"img_size у файлі моделі = {ckpt['img_size']}, очікується {IMG_SIZE}")
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {"regime": ckpt["regime"], "seed": int(ckpt["seed"]),
                   "classes": ckpt["classes"], "img_size": int(ckpt["img_size"])}


def pytorch_logits(model, x: np.ndarray, bs: int = EVAL_BS) -> np.ndarray:
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            out.append(model(torch.from_numpy(x[i:i + bs])).float().numpy())
    return np.concatenate(out)


# ------------------------------------------------------------- ONNX Runtime
def make_session(path: Path):
    """Сесія з фіксованими налаштуваннями: CPU, 1+1 потік, рівень оптимізації за замовчуванням."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = LAT_THREADS
    so.inter_op_num_threads = LAT_THREADS
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def onnx_logits(sess, x: np.ndarray, bs: int = EVAL_BS) -> np.ndarray:
    out = [sess.run([OUTPUT_NAME], {INPUT_NAME: x[i:i + bs]})[0] for i in range(0, len(x), bs)]
    return np.concatenate(out)


# ------------------------------------------------------------------ метрики
def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", labels=list(range(6)), zero_division=0))


def quality_row(y_true, y_pred, y_pred_fp32) -> dict:
    """Рядок таблиці якості: macro-F1, accuracy, згода з FP32, b, c, Delta acc, SE, вердикт."""
    n = len(y_true)
    ok, ok32 = y_pred == y_true, y_pred_fp32 == y_true
    b = int(np.sum(ok & ~ok32))          # варіант правильний, FP32 помиляється
    c = int(np.sum(~ok & ok32))          # навпаки
    flips = y_pred != y_pred_fp32
    d_acc = (b - c) / n
    se = float(np.sqrt(b + c) / n)
    threshold = 2.0 * float(np.sqrt(b + c))
    significant = abs(b - c) > threshold
    if not significant:
        verdict = "різниця якості не відрізняється від шуму"
    else:
        verdict = "різниця відрізняється від шуму: " + ("якість вища" if b > c else "якість нижча")
    return {"macro_f1": macro_f1(y_true, y_pred), "accuracy": float(ok.mean()),
            "agreement": float((~flips).mean()), "n_flipped": int(flips.sum()),
            "n_flipped_both_wrong": int(np.sum(flips & ~ok & ~ok32)),
            "b": b, "c": c, "delta_acc": d_acc, "se": se,
            "abs_b_minus_c": abs(b - c), "threshold_2sqrt_bc": threshold,
            "differs_from_noise": bool(significant), "verdict": verdict}


# ------------------------------------------------------------ середовище
def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, check=True).stdout.strip()
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def cpu_flags() -> dict:
    """Наявність цілочисельних SIMD-розширень, важливих для INT8 (за py-cpuinfo; None, якщо невідомо)."""
    want = ["avx", "avx2", "avx512f", "avx512_vnni", "avx_vnni", "avx512bw", "fma", "amx_int8", "sse4_1"]
    try:
        import cpuinfo
        flags = {f.lower() for f in cpuinfo.get_cpu_info().get("flags", [])}
        return {k: (k in flags) for k in want}
    except Exception:
        return {k: None for k in want}


def versions() -> dict:
    import onnx
    import onnxruntime
    import torch
    import torchvision
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--", "*.py", "pyproject.toml", "uv.lock"))
        return {"commit": commit, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def dump_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
