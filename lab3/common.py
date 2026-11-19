"""Спільний код ЛР3: дані, поділ, перетворення, модель, режими, цикл оцінки.

Усі фіксовані налаштування з умови лабораторної зібрано тут, щоб train.py,
checks.py та analyze.py використовували тільки одні й ті самі значення.
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
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- константи
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_SIZE = 160
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
N_TRAIN_PER_CLASS = 200
EPOCHS = 8
TRAIN_BS = 32
EVAL_BS = 64
WEIGHT_DECAY = 1e-4
LR_HEAD = 3e-3
LR_BACKBONE = 3e-4
PARTS = ("train", "validation", "test")
# у якій підтеці набору лежить кожна частина
SRC_DIR = {
    "train": Path("seg_train") / "seg_train",
    "validation": Path("seg_test") / "seg_test",
    "test": Path("seg_test") / "seg_test",
}

BACKBONE_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]
TOP_MODULES = BACKBONE_MODULES + ["fc"]

# Режими: що навчається, що в eval(), групи оптимізатора (модуль, lr).
REGIMES = {
    "scratch": dict(
        pretrained=False,
        trainable=["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"],
        eval_modules=[],
        groups=[(["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"], LR_HEAD)],
        expected_trainable=11_179_590,
    ),
    "frozen": dict(
        pretrained=True,
        trainable=["fc"],
        eval_modules=list(BACKBONE_MODULES),
        groups=[(["fc"], LR_HEAD)],
        expected_trainable=3_078,
    ),
    "unfreeze": dict(
        pretrained=True,
        trainable=["layer4", "fc"],
        eval_modules=["conv1", "bn1", "layer1", "layer2", "layer3"],
        groups=[(["fc"], LR_HEAD), (["layer4"], LR_BACKBONE)],
        expected_trainable=8_396_806,
    ),
}
EXPECTED_TOTAL_PARAMS = 11_179_590

# Контрольна таблиця поділу: (клас, частина) -> (кількість, сума числових імен).
EXPECTED_SPLIT = {
    ("buildings", "train"): (200, 1_996_761), ("buildings", "validation"): (218, 4_837_323),
    ("buildings", "test"): (219, 4_837_559),
    ("forest", "train"): (200, 2_032_625), ("forest", "validation"): (237, 5_276_962),
    ("forest", "test"): (237, 5_291_326),
    ("glacier", "train"): (200, 2_183_923), ("glacier", "validation"): (276, 6_135_652),
    ("glacier", "test"): (277, 6_157_250),
    ("mountain", "train"): (200, 1_950_345), ("mountain", "validation"): (262, 5_789_535),
    ("mountain", "test"): (263, 5_849_765),
    ("sea", "train"): (200, 1_987_613), ("sea", "validation"): (255, 5_655_068),
    ("sea", "test"): (255, 5_668_011),
    ("street", "train"): (200, 1_841_817), ("street", "validation"): (250, 5_541_845),
    ("street", "test"): (251, 5_547_533),
}
EXPECTED_TOTALS = {"train": (1200, 11_993_084), "validation": (1498, 33_236_385),
                   "test": (1502, 33_351_444)}


# --------------------------------------------------------------------- поділ
def _list_class(root: Path, label: str) -> list[str]:
    """Імена файлів теки класу, відсортовані як рядки (sorted(), як в ImageFolder)."""
    return sorted(p.name for p in (root / label).iterdir() if p.suffix.lower() == ".jpg")


def build_split(data_dir: Path) -> list[tuple[str, str, str]]:
    """Будує поділ за процедурою з умови. Повертає рядки (file, label, part)."""
    train_root = data_dir / SRC_DIR["train"]
    test_root = data_dir / SRC_DIR["validation"]

    rows: list[tuple[str, str, str]] = []

    # train: один генератор на всі класи, у фіксованому порядку класів
    rng = np.random.default_rng(0)
    for label in CLASSES:
        files = _list_class(train_root, label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        chosen = sorted(files[i] for i in idx[:N_TRAIN_PER_CLASS])
        rows += [(f, label, "train") for f in chosen]

    # validation / test: новий генератор, так само для всіх класів по черзі
    rng = np.random.default_rng(0)
    val_rows, test_rows = [], []
    for label in CLASSES:
        files = _list_class(test_root, label)
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        val_rows += [(f, label, "validation") for f in sorted(files[i] for i in idx[:cut])]
        test_rows += [(f, label, "test") for f in sorted(files[i] for i in idx[cut:])]
    return rows + val_rows + test_rows


def write_split(rows, path: Path) -> None:
    # "\n" та newline="" - щоб SHA-256 файлу був однаковим на Windows/Linux/macOS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["file", "label", "part"])
        w.writerows(rows)


def read_split(path: Path) -> dict[str, list[tuple[str, str]]]:
    """part -> список (file, label) у порядку рядків split.csv."""
    out: dict[str, list[tuple[str, str]]] = {p: [] for p in PARTS}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["part"]].append((r["file"], r["label"]))
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_control(split: dict[str, list[tuple[str, str]]]):
    """Таблиця контролю: (клас, частина) -> (кількість, сума числових імен)."""
    table = {}
    for part in PARTS:
        for label in CLASSES:
            names = [f for f, l in split[part] if l == label]
            table[(label, part)] = (len(names), sum(int(Path(n).stem) for n in names))
    return table


# -------------------------------------------------------------------- дані
def make_transforms(train: bool):
    norm = T.Normalize(MEAN, STD)
    if train:
        return T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
                          T.RandomHorizontalFlip(), T.ToTensor(), norm])
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), norm])


class SplitDataset(Dataset):
    """Зображення з однієї частини split.csv. Декодовані PIL-зображення кешуються
    (на результат не впливає: перетворення застосовуються щоразу заново)."""

    def __init__(self, items, part: str, data_dir: Path, train: bool):
        self.items = items
        self.root = Path(data_dir) / SRC_DIR[part]
        self.transform = make_transforms(train)
        self.targets = np.array([CLASS_TO_IDX[l] for _, l in items])
        self._cache: dict[int, Image.Image] = {}

    def path(self, i: int) -> Path:
        f, l = self.items[i]
        return self.root / l / f

    def image(self, i: int) -> Image.Image:
        if i not in self._cache:
            self._cache[i] = Image.open(self.path(i)).convert("RGB")
        return self._cache[i]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.transform(self.image(i)), int(self.targets[i])


def train_loader(ds):
    # без власного generator/sampler: випадковість - з глобального генератора PyTorch
    return DataLoader(ds, batch_size=TRAIN_BS, shuffle=True, drop_last=False, num_workers=0)


def eval_loader(ds):
    return DataLoader(ds, batch_size=EVAL_BS, shuffle=False, num_workers=0)


# ------------------------------------------------------------------ модель
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    if device.type == "mps":
        return f"Apple MPS ({platform.machine()})"
    name = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return f"CPU: {name or platform.machine()}"


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_model(regime: str, seed: int) -> torch.nn.Module:
    """Кроки 1-4 побудови моделі."""
    cfg = REGIMES[regime]
    torch.manual_seed(seed)                                    # 1
    np.random.seed(seed)
    weights = ResNet18_Weights.IMAGENET1K_V1 if cfg["pretrained"] else None
    model = torchvision.models.resnet18(weights=weights)       # 2
    model.fc = torch.nn.Linear(512, 6)                         # 3
    for name, p in model.named_parameters():                   # 4
        p.requires_grad = name.split(".")[0] in cfg["trainable"]
    return model


def make_optimizer(model: torch.nn.Module, regime: str) -> torch.optim.Optimizer:
    groups = []
    for modules, lr in REGIMES[regime]["groups"]:
        params = [p for m in modules for p in getattr(model, m).parameters()]
        groups.append({"params": params, "lr": lr})
    # параметри з requires_grad=False в оптимізатор не передаються
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def build(regime: str, seed: int, device: torch.device):
    """Кроки 1-5: модель, перенесення на пристрій, оптимізатор."""
    model = build_model(regime, seed)
    model.to(device)
    return model, make_optimizer(model, regime)


def set_train_mode(model: torch.nn.Module, regime: str) -> None:
    """model.train(), потім .eval() для модулів режиму зі стовпця «У eval()»."""
    model.train()
    for name in REGIMES[regime]["eval_modules"]:
        getattr(model, name).eval()


def count_params(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    out = [model(x.to(device)).float().cpu() for x, _ in loader]
    return torch.cat(out).numpy()


def macro_f1(y_true, logits) -> float:
    return float(f1_score(y_true, np.asarray(logits).argmax(1), average="macro",
                          labels=list(range(6)), zero_division=0))


def select_best(results: list[dict]) -> tuple[str, int, dict]:
    """Правило вибору: режим із найбільшим середнім F (для scratch - єдине значення),
    у ньому - запуск із найбільшим F. Повертає (режим, зерно, середні за режимами)."""
    means = {}
    for r in ("scratch", "frozen", "unfreeze"):
        vals = [x["f1"] for x in results if x["regime"] == r]
        means[r] = float(np.mean(vals))
    best_regime = max(means, key=means.get)   # при рівності - перший за порядком
    cands = [x for x in results if x["regime"] == best_regime]
    best = max(cands, key=lambda x: x["f1"])
    return best_regime, best["seed"], means


# ---------------------------------------------------------------- довідкове
def git_info() -> dict:
    def run(*a):
        return subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        commit = run("rev-parse", "HEAD")
        # "брудність" - лише для файлів коду (артефакти запусків не рахуємо)
        dirty = bool(run("status", "--porcelain", "--", "*.py", "pyproject.toml", "uv.lock"))
        return {"commit": commit, "uncommitted_changes": dirty}
    except Exception:
        return {"commit": None, "uncommitted_changes": None}


def versions() -> dict:
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "torchvision": torchvision.__version__, "numpy": np.__version__,
            "platform": platform.platform()}


def dump_json(obj, path: Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")
