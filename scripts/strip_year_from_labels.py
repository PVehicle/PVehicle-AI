"""Bo nam san xuat khoi ten lop, tru cac lop se bi trung ten.

Ly do: nhan doi xe khong dang tin va cung khong can thiet cho muc dich su
dung. Voi mo hinh xe Viet Nam, anh thu thap tu Wikimedia lan lon nhieu doi
(lop Accent chi 5% dung doi). Voi Stanford Cars, nhan doi chinh xac nhung
nguoi dung thuong chi quan tam dong xe.

Ngoai le: 7 cap trong Stanford Cars chi khac nhau o nam (vi du
`Audi S4 Sedan 2007` va `Audi S4 Sedan 2012`). Bo nam se lam hai lop khac
nhau co cung ten — mo hinh van xuat ra hai lop rieng nhung nguoi dung thay
hai dong giong het, va module tu van khong tra cuu duoc. Cac lop nay GIU
NGUYEN nam.

Cach chay:
    python scripts/strip_year_from_labels.py --dry-run
    python scripts/strip_year_from_labels.py
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

# Mo hinh quoc te.
INTL_LABELS = MODELS_DIR / "class_names.json"
CAR_SPECS_CSV = DATA_DIR / "car_specs.csv"

# Mo hinh xe Viet Nam.
VN_LABELS = MODELS_DIR / "vn_class_names.json"
VN_SPECS = MODELS_DIR / "vn_car_specs.json"

YEAR_SUFFIX = re.compile(r"\s+\d{4}$")


def strip_year(name: str) -> str:
    return YEAR_SUFFIX.sub("", name).strip()


def build_rename_map(
    labels: list[str], reserved: set[str] | None = None
) -> dict[str, str]:
    """Tao anh xa ten cu -> ten moi.

    Giu nguyen ten cu trong hai truong hop:
      1. Bo nam se trung voi mot lop khac TRONG CUNG mo hinh
      2. Ten sau khi bo nam da bi mo hinh KIA dung (tham so `reserved`)

    Truong hop 2 quan trong vi module tu van dung ten lop lam khoa tra
    cuu — hai xe cung ten se tra ra sai thong so. Vi du ca hai mo hinh
    deu co Camry Sedan, nhung gia niem yet khac nhau.
    """
    reserved = reserved or set()
    stripped = [strip_year(name) for name in labels]
    counts = Counter(stripped)

    mapping = {}
    for original, short in zip(labels, stripped):
        if counts[short] > 1:
            # Trung trong cung mo hinh -> giu nam de phan biet.
            mapping[original] = original
        elif short in reserved:
            # Trung voi mo hinh kia. Neu ten goc con nam thi giu nam,
            # khong thi them hau to de phan biet.
            mapping[original] = (
                original if original != short
                else f"{short} (Viet Nam)"
            )
        else:
            mapping[original] = short
    return mapping


def report(labels: list[str], mapping: dict[str, str], title: str) -> None:
    changed = [n for n in labels if mapping[n] != n]
    kept = [n for n in labels if mapping[n] == n]

    print()
    print("=" * 66)
    print(f"{title}: {len(labels)} lop")
    print("=" * 66)
    print(f"  Bo nam    : {len(changed)} lop")
    print(f"  Giu nguyen: {len(kept)} lop (se bi trung ten neu bo nam)")

    if kept:
        print("\n  Cac lop giu nguyen nam:")
        for name in kept:
            print(f"    {name}")


def update_json_labels(path: Path, mapping: dict[str, str]) -> None:
    """Cap nhat file danh sach nhan, GIU NGUYEN THU TU.

    Thu tu ung voi chi so dau ra cua mo hinh — dao lon se lam sai het.
    """
    labels = json.loads(path.read_text(encoding="utf-8"))
    new_labels = [mapping.get(name, name) for name in labels]
    path.write_text(
        json.dumps(new_labels, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Da cap nhat %s", path.name)


def update_json_specs(path: Path, mapping: dict[str, str]) -> None:
    """Cap nhat bang thong so dang JSON (khoa la ten lop)."""
    if not path.exists():
        return
    specs = json.loads(path.read_text(encoding="utf-8"))
    new_specs = {}
    for name, row in specs.items():
        new_name = mapping.get(name, name)
        row = dict(row)
        row["class_name"] = new_name
        new_specs[new_name] = row
    path.write_text(
        json.dumps(new_specs, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Da cap nhat %s", path.name)


def update_csv_specs(path: Path, mapping: dict[str, str]) -> None:
    """Cap nhat cot class_name trong bang thong so dang CSV."""
    if not path.exists():
        return

    with path.open(encoding="utf-8-sig", newline="") as fp:
        reader = csv.DictReader(fp)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        row["class_name"] = mapping.get(row["class_name"], row["class_name"])

    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Da cap nhat %s", path.name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bo nam khoi ten lop, tru cac lop se bi trung ten."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chi hien thong ke, khong sua file",
    )
    args = parser.parse_args()

    setup_logging()

    if not INTL_LABELS.exists():
        logger.error("Chua co %s", INTL_LABELS)
        return 1

    intl_labels = json.loads(INTL_LABELS.read_text(encoding="utf-8"))
    intl_map = build_rename_map(intl_labels)
    report(intl_labels, intl_map, "MO HINH QUOC TE")

    vn_map: dict[str, str] = {}
    if VN_LABELS.exists():
        vn_labels = json.loads(VN_LABELS.read_text(encoding="utf-8"))
        # Truyen ten cua mo hinh quoc te vao de tranh trung cheo.
        vn_map = build_rename_map(vn_labels, set(intl_map.values()))
        report(vn_labels, vn_map, "MO HINH XE VIET NAM")

    # Kiem tra lai lan cuoi: khong duoc con ten nao trung.
    all_new = list(intl_map.values()) + list(vn_map.values())
    cross = [name for name, n in Counter(all_new).items() if n > 1]
    if cross:
        print()
        print("  LOI: van con ten trung sau khi xu ly:")
        for name in cross:
            print(f"    {name}")
        return 1

    if args.dry_run:
        print("\n(--dry-run: khong sua file)")
        return 0

    print()
    update_json_labels(INTL_LABELS, intl_map)
    update_csv_specs(CAR_SPECS_CSV, intl_map)

    if vn_map:
        update_json_labels(VN_LABELS, vn_map)
        update_json_specs(VN_SPECS, vn_map)

    print()
    print("Xong. Mo hinh KHONG can train lai — chi doi nhan hien thi.")
    print("Chay lai: python -m unittest discover tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
