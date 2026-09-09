"""Chuan bi tap du lieu xe Viet Nam de huan luyen.

Cong viec:
  1. Doc anh da duyet trong data/raw/vn_cars/
  2. Chia train/test theo ti le, giu nguyen ti le cua tung lop
  3. Tinh trong so lop de bu mat can bang
  4. Sinh file cau hinh cho notebook huan luyen

Ve mat can bang lop: cac lop it anh (vi du VinFast Fadil 14 anh so voi
Toyota Vios 200 anh) se bi mo hinh "bo qua" neu khong xu ly. Script tinh
san trong so va he so lay mau de notebook dung.

Cach chay:
    python scripts/prepare_vn_dataset.py
    python scripts/prepare_vn_dataset.py --min-images 20 --test-ratio 0.2
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ensure_dir,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

RAW_VN_DIR = RAW_DATA_DIR / "vn_cars"
OUTPUT_DIR = PROCESSED_DATA_DIR / "vn_cars"
CLASSES_FILE = DATA_DIR / "vn_car_classes.json"
CONFIG_FILE = DATA_DIR / "vn_dataset_config.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

DEFAULT_TEST_RATIO = 0.2

# Duoi nguong nay thi mot lop khong du anh de chia train/test co y nghia:
# 6 anh -> chi 1 anh test, ket qua danh gia gan nhu ngau nhien.
ABSOLUTE_MIN = 8

# Tren nguong nay coi la lop "day du", dung lam moc tinh trong so.
REFERENCE_COUNT = 100


def list_class_images(class_dir: Path) -> list[Path]:
    """Danh sach anh cua mot lop, da sap xep de ket qua tai lap duoc."""
    return sorted(
        path for path in class_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
    )


def split_train_test(
    images: list[Path], test_ratio: float
) -> tuple[list[Path], list[Path]]:
    """Chia train/test.

    Lay anh test theo BUOC DEU thay vi cat mot doan lien tuc: anh Wikimedia
    thuong sap theo ten, cac anh lien tiep hay chup cung mot chiec xe cung
    mot buoi. Cat lien tuc se khien tap test toan xe la, danh gia sai lech.
    """
    n_test = max(1, round(len(images) * test_ratio))
    step = len(images) / n_test

    test_indices = {int(i * step) for i in range(n_test)}
    test = [img for i, img in enumerate(images) if i in test_indices]
    train = [img for i, img in enumerate(images) if i not in test_indices]
    return train, test


def compute_class_weights(counts: dict[str, int]) -> dict[str, float]:
    """Tinh trong so cho tung lop de bu mat can bang.

    Cong thuc: trong so = (so anh cua lop day du / so anh cua lop nay),
    gioi han trong khoang [1, 8] de lop qua it khong lan at moi thu.
    """
    weights = {}
    for name, count in counts.items():
        raw = REFERENCE_COUNT / max(count, 1)
        weights[name] = round(min(max(raw, 1.0), 8.0), 3)
    return weights


def copy_split(
    pairs: list[tuple[str, Path]], out_dir: Path, split: str
) -> int:
    """Chep anh vao thu muc train/ hoac test/ theo tung lop."""
    copied = 0
    for class_name, image_path in pairs:
        target_dir = ensure_dir(out_dir / split / class_name)
        target = target_dir / image_path.name
        if not target.exists():
            shutil.copy2(image_path, target)
        copied += 1
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chuan bi tap du lieu xe VN de huan luyen."
    )
    parser.add_argument(
        "--test-ratio", type=float, default=DEFAULT_TEST_RATIO,
        help=f"Ti le anh danh cho test (mac dinh: {DEFAULT_TEST_RATIO})",
    )
    parser.add_argument(
        "--min-images", type=int, default=ABSOLUTE_MIN,
        help=(
            f"Loai lop co it hon so anh nay (mac dinh: {ABSOLUTE_MIN}). "
            "Dat 0 de giu tat ca."
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chi hien thong ke, khong chep file",
    )
    args = parser.parse_args()

    setup_logging()

    if not RAW_VN_DIR.exists():
        logger.error(
            "Chua co anh trong %s. Chay truoc:\n"
            "  python scripts/collect_vn_images.py", RAW_VN_DIR,
        )
        return 1

    catalog = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
    known_classes = {car["class_name"]: car for car in catalog["cars"]}

    # --- Doc so luong anh tung lop ---
    class_dirs = sorted(
        path for path in RAW_VN_DIR.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    counts: dict[str, int] = {}
    images_by_class: dict[str, list[Path]] = {}

    for class_dir in class_dirs:
        images = list_class_images(class_dir)
        if not images:
            continue
        counts[class_dir.name] = len(images)
        images_by_class[class_dir.name] = images

    if not counts:
        logger.error("Khong tim thay anh nao trong %s", RAW_VN_DIR)
        return 1

    # --- Loc lop qua it anh ---
    kept = {n: c for n, c in counts.items() if c >= args.min_images}
    dropped = {n: c for n, c in counts.items() if c < args.min_images}

    unknown = [n for n in kept if n not in known_classes]
    if unknown:
        logger.warning(
            "%d lop khong co trong vn_car_classes.json: %s",
            len(unknown), ", ".join(unknown[:3]),
        )

    # --- Chia train/test ---
    train_pairs: list[tuple[str, Path]] = []
    test_pairs: list[tuple[str, Path]] = []
    split_info: dict[str, dict] = {}

    for name in sorted(kept):
        train, test = split_train_test(
            images_by_class[name], args.test_ratio
        )
        train_pairs += [(name, p) for p in train]
        test_pairs += [(name, p) for p in test]
        split_info[name] = {"train": len(train), "test": len(test)}

    weights = compute_class_weights(kept)

    # --- Bao cao ---
    total = sum(kept.values())
    sorted_counts = sorted(kept.items(), key=lambda kv: -kv[1])

    print()
    print("=" * 68)
    print(f"TAP DU LIEU XE VIET NAM: {len(kept)} lop, {total} anh")
    print("=" * 68)
    print(f"{'Lop':<34} {'Anh':>5} {'Train':>6} {'Test':>5} {'Trong so':>9}")
    print("-" * 68)
    for name, count in sorted_counts:
        info = split_info[name]
        print(
            f"{name:<34} {count:>5} {info['train']:>6} "
            f"{info['test']:>5} {weights[name]:>9.2f}"
        )
    print("-" * 68)
    print(f"{'TONG':<34} {total:>5} {len(train_pairs):>6} "
          f"{len(test_pairs):>5}")

    if dropped:
        print(f"\nDA LOAI {len(dropped)} lop (duoi {args.min_images} anh):")
        for name, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<34} {count:>5}")

    # --- Canh bao mat can bang ---
    if kept:
        most = max(kept.values())
        least = min(kept.values())
        ratio = most / max(least, 1)
        print()
        print(f"Muc mat can bang: {ratio:.1f}x "
              f"(nhieu nhat {most}, it nhat {least})")
        if ratio > 5:
            print()
            print("  Chenh lech lon. Notebook huan luyen SE dung:")
            print("    - Trong so lop trong ham mat mat")
            print("    - Lay mau can bang moi epoch")
            print("  Neu khong, cac lop it anh se gan nhu khong bao gio")
            print("  duoc du doan, du do chinh xac tong the van cao.")

    if args.dry_run:
        print("\n(--dry-run: khong chep file)")
        return 0

    # --- Chep anh ---
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    ensure_dir(OUTPUT_DIR)

    copy_split(train_pairs, OUTPUT_DIR, "train")
    copy_split(test_pairs, OUTPUT_DIR, "test")
    logger.info("Da chep %d anh train, %d anh test",
                len(train_pairs), len(test_pairs))

    # --- Ghi cau hinh cho notebook huan luyen ---
    config = {
        "num_classes": len(kept),
        "total_images": total,
        "test_ratio": args.test_ratio,
        "classes": sorted(kept),
        "counts": {n: kept[n] for n in sorted(kept)},
        "split": split_info,
        "class_weights": {n: weights[n] for n in sorted(kept)},
        "specs": {
            n: {
                k: v for k, v in known_classes[n].items()
                if k not in ("search",)
            }
            for n in sorted(kept) if n in known_classes
        },
        "_luu_y": (
            "class_weights dung cho ham mat mat va bo lay mau. Lop it anh "
            "co trong so cao hon de mo hinh khong bo qua."
        ),
    }
    CONFIG_FILE.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nAnh da chia: {OUTPUT_DIR}")
    print(f"Cau hinh:    {CONFIG_FILE}")
    print("\nBuoc tiep theo: sinh notebook huan luyen cho tap du lieu nay.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
