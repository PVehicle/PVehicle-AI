"""Export module tu van sang ONNX va do thoi gian suy luan.

Tra loi cau hoi cua de cuong: co nen dong bo toan bo kien truc sang ONNX
khong? Script nay export buoc tim lang gieng gan nhat (KNN) cua
Scikit-learn sang ONNX bang skl2onnx, roi do thoi gian suy luan cua ca hai
ban tren cung mot tap truy van.

Ket qua duoc ghi vao docs/onnx_vs_pandas.md.

Cach chay:
    python scripts/export_recommender_onnx.py
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rec.recommender import CarRecommender  # noqa: E402
from src.utils import MODELS_DIR, get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

OUTPUT_PATH = MODELS_DIR / "recommender_knn.onnx"

# So lang gieng lay ra: du de bo chinh xe truy van roi con lai top-5.
N_NEIGHBORS = 6

BENCHMARK_RUNS = 200
WARMUP_RUNS = 20


def export_to_onnx(recommender: CarRecommender, output_path: Path) -> None:
    """Dong goi buoc tim lang gieng gan nhat thanh mo hinh ONNX."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.neighbors import KNeighborsTransformer

    features = recommender.features.astype(np.float32)

    # Dung KNeighborsTransformer thay vi NearestNeighbors: lop sau dat ca
    # `radius` lan `n_neighbors`, ma skl2onnx khong ho tro truong hop do.
    model = KNeighborsTransformer(
        n_neighbors=N_NEIGHBORS, mode="distance", algorithm="brute"
    )
    model.fit(features)

    initial_type = [
        ("features", FloatTensorType([None, features.shape[1]]))
    ]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=17,
        options={id(model): {"optim": "cdist"}},
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(onnx_model.SerializeToString())

    size_kb = output_path.stat().st_size / 1024
    logger.info(
        "Da export %s (%.1f KB, %d xe x %d dac trung)",
        output_path.name, size_kb, features.shape[0], features.shape[1],
    )


def verify_same_results(
    recommender: CarRecommender, sample_names: list[str]
) -> bool:
    """Kiem tra hai ban cho ket qua giong nhau.

    Neu khac nhau thi so lieu do thoi gian tro nen vo nghia — nhanh hon ma
    sai thi khong dung duoc.
    """
    from src.rec.onnx_recommender import OnnxRecommender

    onnx_rec = OnnxRecommender(base=recommender)
    all_match = True

    for name in sample_names:
        pandas_names = [
            r.class_name for r in recommender.recommend_similar(name)
        ]
        onnx_names = [
            r.class_name for r in onnx_rec.recommend_similar(name)
        ]
        if pandas_names != onnx_names:
            all_match = False
            logger.warning("Ket qua khac nhau voi xe %r:", name)
            logger.warning("  Pandas: %s", pandas_names)
            logger.warning("  ONNX  : %s", onnx_names)

    return all_match


def measure(func, *args) -> tuple[float, float]:
    """Do thoi gian chay (ms): tra ve (trung vi, do lech chuan)."""
    for _ in range(WARMUP_RUNS):
        func(*args)

    timings = []
    for _ in range(BENCHMARK_RUNS):
        started = time.perf_counter()
        func(*args)
        timings.append((time.perf_counter() - started) * 1000)

    return statistics.median(timings), statistics.stdev(timings)


def run_benchmark(recommender: CarRecommender) -> None:
    """Do va bao cao thoi gian suy luan cua hai ban."""
    from src.rec.onnx_recommender import OnnxRecommender

    sample = "BMW M3 Coupe 2012"

    logger.info("Do thoi gian tao doi tuong (nap mo hinh):")
    started = time.perf_counter()
    onnx_rec = OnnxRecommender(base=recommender)
    load_ms = (time.perf_counter() - started) * 1000
    logger.info("  Khoi tao InferenceSession: %.1f ms", load_ms)

    pandas_median, pandas_std = measure(
        recommender.recommend_similar, sample
    )
    onnx_median, onnx_std = measure(onnx_rec.recommend_similar, sample)

    ratio = onnx_median / pandas_median

    print()
    print("=" * 62)
    print(f"So sanh thoi gian suy luan ({BENCHMARK_RUNS} lan chay)")
    print("=" * 62)
    print(f"{'Ban trien khai':<28} {'Trung vi':>12} {'Do lech':>12}")
    print("-" * 62)
    print(f"{'Pandas + NumPy':<28} {pandas_median:>9.3f} ms "
          f"{pandas_std:>9.3f} ms")
    print(f"{'ONNX Runtime (KNN)':<28} {onnx_median:>9.3f} ms "
          f"{onnx_std:>9.3f} ms")
    print("-" * 62)

    if ratio > 1:
        print(f"=> Ban Pandas nhanh hon {ratio:.2f} lan")
    else:
        print(f"=> Ban ONNX nhanh hon {1 / ratio:.2f} lan")

    print(f"=> Chi phi khoi tao ONNX session: {load_ms:.1f} ms")
    print("=" * 62)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export module tu van sang ONNX va do hieu nang."
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH,
        help=f"Duong dan file ONNX (mac dinh: {OUTPUT_PATH.name})",
    )
    parser.add_argument(
        "--skip-benchmark", action="store_true",
        help="Chi export, khong do thoi gian",
    )
    args = parser.parse_args()

    setup_logging()

    recommender = CarRecommender()
    export_to_onnx(recommender, args.output)

    samples = [
        "BMW M3 Coupe 2012",
        "Honda Odyssey Minivan 2012",
        "Ferrari FF Coupe 2012",
    ]
    if verify_same_results(recommender, samples):
        logger.info("Hai ban cho ket qua GIONG NHAU tren %d xe mau.",
                    len(samples))
    else:
        logger.error("Hai ban cho ket qua KHAC NHAU — can kiem tra lai.")
        return 1

    if not args.skip_benchmark:
        run_benchmark(recommender)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
