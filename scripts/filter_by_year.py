"""Loc bo anh xe doi cu khong khop nhan.

Van de: tim "Toyota Vios" tren Wikimedia tra ve anh tu 2003 den 2025 — bon
the he xe khac han nhau ve ngoai hinh. Gan tat ca vao lop
"Toyota Vios Sedan 2023" se day mo hinh hoc rang chung la mot.

May man la ten file Wikimedia thuong CO nam san xuat o dau
("2011 Toyota Vios J front.jpg"), nen loc duoc tu dong.

Quy tac:
  - Ten file co nam VA nam do cach nam cua nhan qua xa  -> loai
  - Ten file khong co nam                               -> giu (khong doan)
  - Ten file chua tu khoa bo phan (rim, wheel...)        -> loai

Anh bi loai chuyen sang `_year_rejected/`, khong xoa han.

Cach chay:
    python scripts/filter_by_year.py --dry-run
    python scripts/filter_by_year.py --tolerance 3
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    RAW_DATA_DIR,
    ensure_dir,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

IMAGES_DIR = RAW_DATA_DIR / "vn_cars"
REJECT_DIRNAME = "_year_rejected"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Nam cua xe thuong nam o dau ten file, nhung cung co the o giua.
YEAR_PATTERN = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")

# Khoang cach nam toi da con chap nhan duoc.
#
# Vi sao 3: mot the he xe thuong keo dai 5-7 nam, va ban facelift giua
# vong doi van rat giong ban goc. Lech 3 nam thuong van cung the he.
DEFAULT_TOLERANCE = 3

# Anh chi chup mot bo phan, khong dung de nhan dien ca chiec xe.
PART_KEYWORDS = (
    "rim of", "wheel of", "engine of", "badge", "emblem",
    "logo of", "headlamp", "tail lamp", "grille of",
)


def extract_year(filename: str) -> int | None:
    """Lay nam san xuat tu ten file. Tra None neu khong tim thay."""
    match = YEAR_PATTERN.search(filename)
    return int(match.group(1)) if match else None


def target_year(class_name: str) -> int | None:
    """Lay nam cua nhan, vi du 'Toyota Vios Sedan 2023' -> 2023."""
    match = re.search(r"(\d{4})$", class_name)
    return int(match.group(1)) if match else None


def is_part_photo(filename: str) -> bool:
    """Anh chi chup mot bo phan xe."""
    lowered = filename.lower()
    return any(word in lowered for word in PART_KEYWORDS)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Loc bo anh xe doi cu khong khop nhan."
    )
    parser.add_argument(
        "--tolerance", type=int, default=DEFAULT_TOLERANCE,
        help=(
            f"Khoang cach nam toi da chap nhan duoc "
            f"(mac dinh: {DEFAULT_TOLERANCE})"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chi hien thong ke, khong chuyen file",
    )
    args = parser.parse_args()

    setup_logging()

    if not IMAGES_DIR.exists():
        logger.error("Chua co anh trong %s", IMAGES_DIR)
        return 1

    class_dirs = sorted(
        path for path in IMAGES_DIR.iterdir()
        if path.is_dir() and not path.name.startswith("_")
    )

    print()
    print("=" * 72)
    print(f"LOC ANH THEO NAM (chap nhan lech toi da {args.tolerance} nam)")
    print("=" * 72)
    print(f"{'Lop':<32} {'Truoc':>6} {'Giu':>5} {'Loai':>5} {'Ly do':<20}")
    print("-" * 72)

    total_before = total_kept = total_rejected = 0

    for class_dir in class_dirs:
        target = target_year(class_dir.name)
        images = [
            path for path in class_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if not images:
            continue

        to_reject: list[tuple[Path, str]] = []

        for path in images:
            if is_part_photo(path.name):
                to_reject.append((path, "chi chup bo phan"))
                continue

            year = extract_year(path.name)
            # Khong co nam trong ten -> giu lai, khong doan bua.
            if year is None or target is None:
                continue

            if abs(year - target) > args.tolerance:
                to_reject.append((path, f"nam {year}"))

        kept = len(images) - len(to_reject)
        total_before += len(images)
        total_kept += kept
        total_rejected += len(to_reject)

        reasons = {reason for _, reason in to_reject}
        year_reasons = [r for r in reasons if r.startswith("nam")]
        if year_reasons:
            years = sorted(int(r.split()[1]) for r in year_reasons)
            note = f"nam {years[0]}-{years[-1]}"
        elif reasons:
            note = "bo phan"
        else:
            note = ""

        warn = "  !" if kept < 20 else ""
        print(
            f"{class_dir.name:<32} {len(images):>6} {kept:>5} "
            f"{len(to_reject):>5} {note:<20}{warn}"
        )

        if not args.dry_run and to_reject:
            reject_dir = ensure_dir(
                IMAGES_DIR / REJECT_DIRNAME / class_dir.name
            )
            for path, _ in to_reject:
                shutil.move(str(path), str(reject_dir / path.name))

    print("-" * 72)
    print(f"{'TONG':<32} {total_before:>6} {total_kept:>5} "
          f"{total_rejected:>5}")

    if total_before:
        print(f"\nDa loai {total_rejected / total_before:.1%} so anh.")

    print("\nLuu y: anh KHONG co nam trong ten file duoc giu lai — script")
    print("khong doan bua. Duyet bang mat van la cach chinh xac nhat.")

    if args.dry_run:
        print("\n(--dry-run: khong chuyen file)")
    else:
        print(f"\nAnh bi loai da chuyen sang {IMAGES_DIR / REJECT_DIRNAME}")
        print("Chay lai prepare_vn_dataset.py va pack_vn_dataset.py.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
