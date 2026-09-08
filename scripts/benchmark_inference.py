"""Do thoi gian suy luan cua luong nhan dien tren CPU.

Sinh so lieu cho chuong danh gia cua bao cao: thoi gian tung tang, thong
luong (anh/giay), va muc dung bo nho.

Cach chay:
    python scripts/benchmark_inference.py
    python scripts/benchmark_inference.py --runs 100 --image anh_that.jpg
"""

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cv.classifier import CLASSIFIER_INPUT_SIZE  # noqa: E402
from src.cv.detector import YOLO_INPUT_SIZE  # noqa: E402
from src.cv.pipeline import (  # noqa: E402
    CLASS_NAMES_FILE,
    CLASSIFIER_MODEL,
    DETECTOR_MODEL,
    RecognitionPipeline,
    load_image,
)
from src.utils import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

DEFAULT_RUNS = 50
WARMUP_RUNS = 5


def make_synthetic_image(width=640, height=480) -> np.ndarray:
    """Anh tong hop: nen xam voi mot hinh chu nhat dam o giua."""
    image = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.rectangle(
        image,
        (width // 4, height // 3),
        (3 * width // 4, 2 * height // 3),
        (120, 60, 40), -1,
    )
    return image


def measure(func, *args, runs: int) -> dict:
    """Do thoi gian chay (ms), tra ve cac chi so thong ke.

    Dung trung vi lam so lieu chinh: phep do tren may ca nhan hay bi nhieu
    boi cac tien trinh khac, trung vi it bi anh huong hon trung binh.
    """
    for _ in range(WARMUP_RUNS):
        func(*args)

    timings = []
    for _ in range(runs):
        started = time.perf_counter()
        func(*args)
        timings.append((time.perf_counter() - started) * 1000)

    timings.sort()
    return {
        "median": statistics.median(timings),
        "mean": statistics.mean(timings),
        "stdev": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "min": timings[0],
        "max": timings[-1],
        # p95: 95% so lan chay nhanh hon muc nay.
        "p95": timings[int(len(timings) * 0.95) - 1],
    }


def format_row(name: str, stats: dict) -> str:
    return (
        f"{name:<26} "
        f"{stats['median']:>8.2f} "
        f"{stats['mean']:>8.2f} "
        f"{stats['stdev']:>8.2f} "
        f"{stats['p95']:>8.2f} "
        f"{stats['max']:>8.2f}"
    )


def print_environment() -> None:
    """In thong tin moi truong de so lieu co ngu canh."""
    import onnxruntime as ort

    print("=" * 70)
    print("MOI TRUONG DO")
    print("=" * 70)
    print(f"  He dieu hanh   : {platform.system()} {platform.release()}")
    print(f"  CPU            : {platform.processor()[:52]}")
    print(f"  Python         : {platform.python_version()}")
    print(f"  ONNX Runtime   : {ort.__version__}")
    print(f"  Provider       : CPUExecutionProvider")


def print_model_info() -> None:
    """In kich thuoc cac file mo hinh."""
    print()
    print("=" * 70)
    print("MO HINH")
    print("=" * 70)
    for path, role in (
        (DETECTOR_MODEL, "Tang 1 - phat hien"),
        (CLASSIFIER_MODEL, "Tang 2 - phan loai"),
        (CLASS_NAMES_FILE, "Danh sach nhan"),
    ):
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  {role:<22} {path.name:<24} {size_mb:>7.1f} MB")


def run_benchmark(pipeline, image: np.ndarray, runs: int) -> dict:
    """Do tung tang rieng va ca luong hoan chinh."""
    detector = pipeline.detector
    classifier = pipeline.classifier

    # Anh crop dung lam dau vao cho tang phan loai.
    detections = detector.detect(image)
    crop = (
        detections[0].crop(image, padding=0.08) if detections else image
    )

    return {
        "detect": measure(detector.detect, image, runs=runs),
        "classify": measure(classifier.predict, crop, runs=runs),
        "pipeline": measure(pipeline.recognize, image, runs=runs),
    }


def print_results(results: dict, image_shape: tuple, runs: int) -> None:
    """In bang ket qua do."""
    height, width = image_shape[:2]

    print()
    print("=" * 70)
    print(f"KET QUA ({runs} lan chay, anh {width}x{height})")
    print("=" * 70)
    print(f"{'Buoc xu ly':<26} {'Trung vi':>8} {'TB':>8} "
          f"{'Do lech':>8} {'p95':>8} {'Max':>8}")
    print(f"{'':<26} {'(ms)':>8} {'(ms)':>8} "
          f"{'(ms)':>8} {'(ms)':>8} {'(ms)':>8}")
    print("-" * 70)
    print(format_row(
        f"Tang 1 detect ({YOLO_INPUT_SIZE}px)", results["detect"]
    ))
    print(format_row(
        f"Tang 2 classify ({CLASSIFIER_INPUT_SIZE}px)", results["classify"]
    ))
    print("-" * 70)
    print(format_row("Ca luong (end-to-end)", results["pipeline"]))
    print("=" * 70)

    total = results["pipeline"]["median"]
    detect_ms = results["detect"]["median"]
    classify_ms = results["classify"]["median"]

    print()
    print("PHAN TICH")
    print("-" * 70)
    print(f"  Thong luong        : {1000 / total:.1f} anh/giay")
    print(f"  Thoi gian mot anh  : {total:.1f} ms")
    print(f"  Ty trong tang 1    : {detect_ms / total:.0%}")
    print(f"  Ty trong tang 2    : {classify_ms / total:.0%}")

    overhead = total - detect_ms - classify_ms
    print(f"  Chi phi con lai    : {overhead:.1f} ms "
          f"({max(overhead, 0) / total:.0%}) — crop, doi mau, ghep ket qua")
    print("=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Do thoi gian suy luan cua luong nhan dien tren CPU."
    )
    parser.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS,
        help=f"So lan chay moi phep do (mac dinh: {DEFAULT_RUNS})",
    )
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Anh that de do; bo trong thi dung anh tong hop",
    )
    args = parser.parse_args()

    setup_logging()

    try:
        pipeline = RecognitionPipeline()
    except FileNotFoundError as exc:
        print(f"Loi: {exc}", file=sys.stderr)
        return 1

    if args.image is not None:
        image = load_image(args.image)
        logger.info("Dung anh that: %s", args.image.name)
    else:
        image = make_synthetic_image()
        logger.info("Dung anh tong hop 640x480")

    print_environment()
    print_model_info()

    results = run_benchmark(pipeline, image, args.runs)
    print_results(results, image.shape, args.runs)

    if is_dummy_model(pipeline):
        print()
        print("  CANH BAO: dang dung MO HINH GIA.")
        print("  So lieu phan anh dung cau truc dau vao/ra, nhung KHONG")
        print("  phai hieu nang that. Chay lai sau khi co mo hinh that.")

    return 0


def is_dummy_model(pipeline) -> bool:
    """Nhan biet mo hinh gia qua so luong node trong do thi tinh toan.

    Mo hinh gia chi co vai node (Reshape, MatMul), trong khi EfficientNet-B0
    va YOLOv8 that co hang tram node. Khong dung kich thuoc file de phan
    biet: ma tran trong so ngau nhien cua mo hinh gia cung rat lon.
    """
    try:
        import onnx
    except ImportError:
        return False

    for path in (DETECTOR_MODEL, CLASSIFIER_MODEL):
        try:
            model = onnx.load(str(path))
        except Exception:
            return False
        if len(model.graph.node) > 20:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
