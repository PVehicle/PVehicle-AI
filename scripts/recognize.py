"""Nhan dien dong xe tu dong lenh, khong can mo giao dien.

Huu ich khi can thu nhanh, xu ly hang loat nhieu anh, hoac lay so lieu de
dua vao bao cao.

Vi du:
    python scripts/recognize.py anh_xe.jpg
    python scripts/recognize.py anh_xe.jpg --save ket_qua.jpg
    python scripts/recognize.py thu_muc_anh/ --recommend
    python scripts/recognize.py anh_xe.jpg --json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from src.cv.pipeline import (  # noqa: E402
    RecognitionPipeline,
    draw_detections,
    load_image,
)
from src.rec.recommender import CarRecommender  # noqa: E402
from src.utils import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(source: Path) -> list[Path]:
    """Lay danh sach anh tu mot file hoac ca thu muc."""
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(
            path for path in source.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
    raise FileNotFoundError(f"Khong tim thay: {source}")


def format_price(million_vnd: float) -> str:
    """Dinh dang gia cho de doc."""
    if million_vnd >= 1000:
        return f"{million_vnd / 1000:.2f} ty".replace(".", ",")
    return f"{million_vnd:,.0f} tr".replace(",", ".")


def print_result(
    image_path: Path, results, recommender, show_recommend: bool
) -> None:
    """In ket qua nhan dien duoi dang bang cho de doc."""
    print()
    print("=" * 66)
    print(f"Anh: {image_path.name}")
    print("=" * 66)

    if not results:
        print("  Khong nhan dien duoc gi.")
        return

    if results[0].is_likely_not_a_car:
        print("  CANH BAO: nhieu kha nang anh nay khong chua o to.")
        print("  (khong phat hien duoc xe, va phan loai khong chac chan)")
        print()

    for index, result in enumerate(results, start=1):
        detection = result.detection
        if detection is None:
            where = "phan loai toan bo anh (khong detect duoc xe)"
        else:
            where = (
                f"box=({detection.x1},{detection.y1})-"
                f"({detection.x2},{detection.y2}) "
                f"conf={detection.confidence:.0%} "
                f"loai={detection.class_name}"
            )
        print(f"Xe {index}: {where}")

        if not result.is_confident:
            print(f"  (do tin cay chi {result.best.confidence:.1%} — "
                  "duoi nguong dang tin)")

        for prediction in result.predictions:
            print(f"    {prediction.confidence:6.2%}  "
                  f"{prediction.class_name}")

        specs = recommender.get_car(result.best.class_name)
        if specs is not None:
            print(f"    -> {specs['body_style']}, "
                  f"{int(specs['seats'])} cho, "
                  f"{format_price(specs['price_million_vnd'])}, "
                  f"{specs['fuel_l_per_100km']:.1f} L/100km")

        if show_recommend and specs is not None:
            print("    Xe tuong tu:")
            similar = recommender.recommend_similar(
                result.best.class_name, top_n=3
            )
            for item in similar:
                print(f"      - {item.class_name} "
                      f"({format_price(item.price_million_vnd)})")
        print()


def build_json(image_path: Path, results, recommender) -> dict:
    """Dung ket qua dang JSON de dung cho xu ly tiep."""
    vehicles = []
    for result in results:
        detection = result.detection
        specs = recommender.get_car(result.best.class_name)
        vehicles.append({
            "box": None if detection is None else {
                "x1": detection.x1, "y1": detection.y1,
                "x2": detection.x2, "y2": detection.y2,
                "confidence": round(detection.confidence, 4),
                "coco_class": detection.class_name,
            },
            "is_confident": result.is_confident,
            "predictions": [
                {
                    "class_id": p.class_id,
                    "class_name": p.class_name,
                    "confidence": round(p.confidence, 4),
                }
                for p in result.predictions
            ],
            "specs": None if specs is None else {
                "body_style": specs["body_style"],
                "seats": int(specs["seats"]),
                "price_million_vnd": float(specs["price_million_vnd"]),
                "fuel_l_per_100km": float(specs["fuel_l_per_100km"]),
            },
        })

    return {
        "image": image_path.name,
        "likely_not_a_car": bool(
            results and results[0].is_likely_not_a_car
        ),
        "vehicle_count": len(results),
        "vehicles": vehicles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nhan dien dong xe tu anh, khong can giao dien."
    )
    parser.add_argument(
        "source", type=Path,
        help="Duong dan mot anh hoac mot thu muc chua anh",
    )
    parser.add_argument(
        "--save", type=Path, default=None,
        help="Luu anh da ve hop bao (chi ap dung khi xu ly mot anh)",
    )
    parser.add_argument(
        "--recommend", action="store_true",
        help="Hien them cac xe tuong tu",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Xuat ket qua dang JSON thay vi bang chu",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="So du doan hien thi cho moi xe (mac dinh: 5)",
    )
    args = parser.parse_args()

    # Che do JSON: khong in log de dau ra van la JSON hop le.
    if not args.json:
        setup_logging()

    try:
        images = collect_images(args.source)
    except FileNotFoundError as exc:
        print(f"Loi: {exc}", file=sys.stderr)
        return 1

    if not images:
        print(f"Khong co anh nao trong {args.source}", file=sys.stderr)
        return 1

    try:
        pipeline = RecognitionPipeline()
    except FileNotFoundError as exc:
        print(f"Loi: {exc}", file=sys.stderr)
        return 1

    recommender = CarRecommender()
    json_output = []

    for image_path in images:
        try:
            image = load_image(image_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Bo qua {image_path.name}: {exc}", file=sys.stderr)
            continue

        results = pipeline.recognize(image, top_k=args.top_k)

        if args.json:
            json_output.append(build_json(image_path, results, recommender))
        else:
            print_result(
                image_path, results, recommender, args.recommend
            )

        if args.save is not None and len(images) == 1:
            drawn = draw_detections(image, results)
            # imencode thay cho imwrite de ghi duoc duong dan tieng Viet.
            success, buffer = cv2.imencode(args.save.suffix, drawn)
            if success:
                args.save.write_bytes(buffer.tobytes())
                if not args.json:
                    print(f"Da luu anh ket qua: {args.save}")

    if args.json:
        print(json.dumps(json_output, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
