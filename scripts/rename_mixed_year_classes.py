"""Doi ten cac lop co anh lan lon nhieu doi xe.

Van de: bo loc theo nam (`filter_by_year.py`) chi bat duoc anh CO nam trong
ten file. Wikimedia con dung nhieu dang viet tat (`'00-'02`, `03-'06`) va
rat nhieu anh khong ghi nam gi ca.

Hau qua nang nhat o lop `Hyundai Accent Sedan 2024`: chi 5% anh dung doi,
con lai la Accent 2006-2014 (phan lon la hatchback). Mo hinh vi the khong
nhan ra Accent 2024 that.

Cach xu ly: **doi ten lop cho dung thuc te** thay vi khang dinh mot doi xe
cu the. `Hyundai Accent Sedan 2024` -> `Hyundai Accent Sedan`.

Uu diem: trung thuc voi du lieu, KHONG PHAI train lai — chi doi nhan hien
thi. Nhuoc diem: mat thong tin ve doi xe.

Cach chay:
    python scripts/rename_mixed_year_classes.py --dry-run
    python scripts/rename_mixed_year_classes.py
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

TRAIN_DIR = PROCESSED_DATA_DIR / "vn_cars" / "train"
LABELS_FILE = MODELS_DIR / "vn_class_names.json"
SPECS_FILE = MODELS_DIR / "vn_car_specs.json"

# Nam trong ten file, ke ca dang viet tat.
FULL_YEAR = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
SHORT_YEAR = re.compile(r"'(\d{2})\b")
RANGE_2D = re.compile(r"\b(\d{2})-'?(\d{2})\b")

# Duoi ti le nay thi coi la lop lan lon nhieu doi.
MIN_CORRECT_RATIO = 0.75

# Chap nhan lech bao nhieu nam thi van coi la cung the he.
YEAR_TOLERANCE = 3


def years_in_filename(name: str) -> list[int]:
    """Lay tat ca nam trong ten file, ke ca dang viet tat 2 chu so."""
    years = [int(m.group(1)) for m in FULL_YEAR.finditer(name)]
    for pattern in (SHORT_YEAR, RANGE_2D):
        for match in pattern.finditer(name):
            two_digit = int(match.group(1))
            # '00 -> 2000, '95 -> 1995
            years.append(
                2000 + two_digit if two_digit < 50 else 1900 + two_digit
            )
    return years


def strip_year(class_name: str) -> str:
    """Bo nam o cuoi ten lop."""
    return re.sub(r"\s+\d{4}$", "", class_name).strip()


def analyse_class(class_dir: Path) -> tuple[int, float]:
    """Tra ve (so anh, ti le anh dung doi).

    Anh khong co nam trong ten khong duoc tinh vao ti le — khong doan bua.
    """
    match = re.search(r"(\d{4})$", class_dir.name)
    if match is None:
        return 0, 1.0
    target = int(match.group(1))

    files = [p for p in class_dir.iterdir() if p.is_file()]
    all_years: list[int] = []
    for path in files:
        all_years += years_in_filename(path.name)

    if not all_years:
        # Khong co thong tin nam -> khong ket luan duoc, coi nhu dat.
        return len(files), 1.0

    correct = sum(
        1 for y in all_years if abs(y - target) <= YEAR_TOLERANCE
    )
    return len(files), correct / len(all_years)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Doi ten cac lop co anh lan lon nhieu doi xe."
    )
    parser.add_argument(
        "--min-ratio", type=float, default=MIN_CORRECT_RATIO,
        help=(
            "Ti le anh dung doi toi thieu de giu nguyen ten "
            f"(mac dinh: {MIN_CORRECT_RATIO})"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chi hien thong ke, khong sua file",
    )
    args = parser.parse_args()

    setup_logging()

    if not TRAIN_DIR.exists():
        logger.error("Chua co du lieu trong %s", TRAIN_DIR)
        return 1
    if not LABELS_FILE.exists():
        logger.error("Chua co %s", LABELS_FILE)
        return 1

    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    specs = (
        json.loads(SPECS_FILE.read_text(encoding="utf-8"))
        if SPECS_FILE.exists() else {}
    )

    print()
    print("=" * 70)
    print("PHAN TICH DO LAN LON DOI XE")
    print("=" * 70)
    print(f"{'Lop':<32} {'Anh':>4} {'Dung doi':>9}  {'Ket luan'}")
    print("-" * 70)

    renames: dict[str, str] = {}

    for class_dir in sorted(x for x in TRAIN_DIR.iterdir() if x.is_dir()):
        count, ratio = analyse_class(class_dir)
        old_name = class_dir.name

        if ratio >= args.min_ratio:
            print(f"{old_name:<32} {count:>4} {ratio:>8.0%}  giu nguyen")
            continue

        new_name = strip_year(old_name)
        if new_name == old_name or new_name in labels:
            print(f"{old_name:<32} {count:>4} {ratio:>8.0%}  "
                  "khong doi duoc (trung ten)")
            continue

        renames[old_name] = new_name
        print(f"{old_name:<32} {count:>4} {ratio:>8.0%}  -> {new_name}")

    if not renames:
        print("\nKhong co lop nao can doi ten.")
        return 0

    print()
    print("=" * 70)
    print(f"SE DOI TEN {len(renames)} LOP")
    print("=" * 70)
    for old, new in renames.items():
        print(f"  {old}")
        print(f"    -> {new}")

    if args.dry_run:
        print("\n(--dry-run: khong sua file)")
        return 0

    # Doi ten trong danh sach nhan. Thu tu PHAI giu nguyen — no ung voi
    # thu tu dau ra cua mo hinh.
    new_labels = [renames.get(name, name) for name in labels]
    LABELS_FILE.write_text(
        json.dumps(new_labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Da cap nhat %s", LABELS_FILE.name)

    # Doi ten trong bang thong so, giu nguyen cac truong khac.
    if specs:
        new_specs = {}
        for name, row in specs.items():
            new_name = renames.get(name, name)
            row = dict(row)
            row["class_name"] = new_name
            new_specs[new_name] = row
        SPECS_FILE.write_text(
            json.dumps(new_specs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("Da cap nhat %s", SPECS_FILE.name)

    print()
    print("Da doi ten. Mo hinh KHONG can train lai — chi doi nhan hien thi.")
    print("Chay lai test de kiem tra: python -m unittest discover tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
