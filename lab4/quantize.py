"""Пункти 2 і 4: підготовка графа і статична квантизація після навчання (QDQ, MinMax)."""
from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)

try:  # у різних версіях ORT функція доступна за різними шляхами
    from onnxruntime.quantization import quant_pre_process
except ImportError:  # pragma: no cover
    from onnxruntime.quantization.shape_inference import quant_pre_process

import common as C


def _ascii_workdir() -> Path:
    """Тека з шляхом лише з ASCII-символів. onnx.shape_inference.infer_shapes_path (її викликає
    quantize_static) на Windows не може записати файл у теку з кирилицею в шляху, тому квантизація
    виконується тут, а результат копіюється назад. На параметри квантизації це не впливає."""
    candidates = [Path(tempfile.gettempdir()), Path(os.environ.get("SystemDrive", "C:") + os.sep)]
    for base in candidates:
        if str(base).isascii() and base.exists():
            return Path(tempfile.mkdtemp(prefix="lab4_q_", dir=base))
    raise SystemExit("Не знайдено теки з ASCII-шляхом для квантизації; задайте TMP/TEMP без кирилиці.")


class CalibReader(CalibrationDataReader):
    """Калібрувальні зображення подаються по одному (1 x 3 x 160 x 160)."""

    def __init__(self, images: np.ndarray):
        self.images = images
        self.i = 0

    def get_next(self):
        if self.i >= len(self.images):
            return None
        x = self.images[self.i:self.i + 1]
        self.i += 1
        return {C.INPUT_NAME: x}

    def rewind(self):
        self.i = 0


def preprocess(fp32: Path, prep: Path) -> float:
    work = _ascii_workdir()
    try:
        src, dst = work / "in.onnx", work / "prep.onnx"
        shutil.copyfile(fp32, src)
        t0 = time.perf_counter()
        quant_pre_process(str(src), str(dst), skip_symbolic_shape=True)
        dt = time.perf_counter() - t0
        shutil.copyfile(dst, prep)
        return dt
    finally:
        shutil.rmtree(work, ignore_errors=True)


def quantize(prep: Path, out: Path, calib_images: np.ndarray, per_channel: bool) -> float:
    """Статична квантизація з параметрами з таблиці фіксованих налаштувань; решта - за замовчуванням."""
    work = _ascii_workdir()
    try:
        src, dst = work / "prep.onnx", work / "q.onnx"
        shutil.copyfile(prep, src)
        reader = CalibReader(calib_images)
        t0 = time.perf_counter()
        quantize_static(str(src), str(dst), reader,
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=per_channel,
                        calibrate_method=CalibrationMethod.MinMax)
        dt = time.perf_counter() - t0
        shutil.copyfile(dst, out)
        return dt
    finally:
        shutil.rmtree(work, ignore_errors=True)
