"""Sinh bang thong so xe (car_specs.csv) tu 196 ten lop Stanford Cars.

Bang nay la dau vao cua module tu van. Stanford Cars chi cung cap ten lop
(hang + dong + kieu dang + nam), KHONG co gia ban hay muc tieu hao nhien
lieu, nen script sinh du lieu theo hai muc do tin cay khac nhau:

  - derived   : suy truc tiep tu ten lop (kieu dang, so cho, hang, nam).
                Day la du lieu CHINH XAC, khong phai uoc luong.
  - simulated : gia ban va muc tieu hao, sinh theo quy tac xac dinh
                (deterministic) tu phan khuc + kieu dang + nam.

Cot `data_source` trong CSV ghi ro tung dong thuoc loai nao.
Xem docs/car_specs_generation.md de biet chi tiet quy tac.

Cach chay:
    python scripts/generate_car_specs.py
"""

import argparse
import csv
import hashlib
import sys
from pathlib import Path

# Cho phep chay truc tiep bang `python scripts/generate_car_specs.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    DATA_DIR,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

CLASSES_FILE = DATA_DIR / "stanford_cars_classes.txt"
OUTPUT_FILE = DATA_DIR / "car_specs.csv"

EXPECTED_CLASS_COUNT = 196

# --- Bang 1: kieu dang -> (body_style, so cho ngoi) ----------------------
# Duyet theo THU TU, lay ket qua khop dau tien. Cac tu khoa dai ("Crew Cab")
# phai dung truoc tu khoa ngan ("Cab") de tranh khop nham.
BODY_STYLE_RULES = [
    ("Crew Cab", ("Pickup", 5)),
    ("Extended Cab", ("Pickup", 5)),
    ("Regular Cab", ("Pickup", 3)),
    ("Club Cab", ("Pickup", 5)),
    ("Quad Cab", ("Pickup", 5)),
    ("SuperCab", ("Pickup", 5)),
    ("Cab", ("Pickup", 5)),
    ("Minivan", ("Minivan", 7)),
    ("SUV", ("SUV", 7)),
    ("Convertible", ("Convertible", 4)),
    ("Coupe", ("Coupe", 4)),
    ("Hatchback", ("Hatchback", 5)),
    ("Wagon", ("Wagon", 5)),
    ("Sedan", ("Sedan", 5)),
    ("Van", ("Van", 8)),
]

# --- Bang 2: 14 lop khong chua tu khoa kieu dang -------------------------
# Cac ban hieu nang cao / dac biet, phai anh xa tuong minh.
SPECIAL_BODY_STYLES = {
    "Acura TL Type-S 2008": ("Sedan", 5),
    "Acura Integra Type R 2001": ("Coupe", 4),
    "Buick Regal GS 2012": ("Sedan", 5),
    "Chevrolet Corvette ZR1 2012": ("Coupe", 2),
    "Chevrolet Corvette Ron Fellows Edition Z06 2007": ("Coupe", 2),
    "Chevrolet HHR SS 2010": ("Wagon", 5),
    "Chevrolet Cobalt SS 2010": ("Coupe", 4),
    "Chevrolet TrailBlazer SS 2009": ("SUV", 5),
    "Chrysler 300 SRT-8 2010": ("Sedan", 5),
    "Dodge Challenger SRT8 2011": ("Coupe", 4),
    "Dodge Charger SRT-8 2009": ("Sedan", 5),
    "FIAT 500 Abarth 2012": ("Hatchback", 4),
    "Jaguar XK XKR 2012": ("Coupe", 4),
    "Lamborghini Gallardo LP 570-4 Superleggera 2012": ("Coupe", 2),
}

# Cac dong xe 2 cho, ghi de so cho suy tu kieu dang.
TWO_SEATER_KEYWORDS = (
    "Corvette",
    "Viper",
    "smart fortwo",
    "Ferrari",
    "Lamborghini",
    "McLaren",
    "Bugatti",
    "Spyker",
    "Audi R8",
    "Porsche",
    "Fisker",
)

# --- Bang 3: hang xe -> phan khuc gia -----------------------------------
LUXURY_BRANDS = frozenset({
    "Acura",
    "Audi",
    "BMW",
    "Buick",
    "Cadillac",
    "Infiniti",
    "Jaguar",
    "Land Rover",
    "Lincoln",
    "Mercedes-Benz",
    "Volvo",
})
EXOTIC_BRANDS = frozenset({
    "Aston Martin",
    "Bentley",
    "Bugatti",
    "Ferrari",
    "Lamborghini",
    "Maybach",
    "McLaren",
    "Rolls-Royce",
    "Spyker",
    "Fisker",
    "Tesla",
})
# Cac hang con lai mac dinh la "economy".

# Mot so dong xe hieu nang cao thuoc hang pho thong nhung gia ngang xe
# sang/sieu xe. Phan khuc suy tu hang se sai, nen ghi de tuong minh.
SEGMENT_OVERRIDES = {
    "Chevrolet Corvette ZR1 2012": "exotic",
    "Chevrolet Corvette Ron Fellows Edition Z06 2007": "exotic",
    "Dodge Challenger SRT8 2011": "luxury",
    "Dodge Charger SRT-8 2009": "luxury",
    "Chrysler 300 SRT-8 2010": "luxury",
}

# Ten hang gom nhieu tu, phai kiem tra truoc khi lay tu dau tien.
MULTI_WORD_BRANDS = (
    "Aston Martin",
    "Land Rover",
    "Rolls-Royce",
    "AM General",
)

# --- Bang 4: gia co so theo phan khuc (trieu VND) ------------------------
BASE_PRICE_BY_SEGMENT = {
    "economy": 600,
    "luxury": 1800,
    "exotic": 12000,
}

# He so nhan gia theo kieu dang.
PRICE_FACTOR_BY_BODY = {
    "Sedan": 1.00,
    "Hatchback": 0.85,
    "Wagon": 1.05,
    "Coupe": 1.20,
    "Convertible": 1.35,
    "SUV": 1.25,
    "Pickup": 1.10,
    "Minivan": 1.15,
    "Van": 0.95,
}

# --- Bang 5: muc tieu hao co so theo kieu dang (L/100km) -----------------
BASE_FUEL_BY_BODY = {
    "Hatchback": 6.5,
    "Sedan": 7.5,
    "Wagon": 8.0,
    "Coupe": 9.5,
    "Convertible": 10.0,
    "Minivan": 10.5,
    "Van": 11.5,
    "SUV": 11.0,
    "Pickup": 12.0,
}

FUEL_FACTOR_BY_SEGMENT = {
    "economy": 1.00,
    "luxury": 1.15,
    "exotic": 1.60,
}

# Nam moi hon -> tiet kiem nhien lieu hon (moc quy chieu: 2012).
REFERENCE_YEAR = 2012
FUEL_IMPROVEMENT_PER_YEAR = 0.015  # 1.5%/nam
PRICE_DEPRECIATION_PER_YEAR = 0.02  # 2%/nam so voi moc 2012

# Bien thien rieng cho tung dong xe, xem ham `model_variation`.
PRICE_VARIATION_SPREAD = 0.18  # gia lech toi da +/-18%
FUEL_VARIATION_SPREAD = 0.10  # muc tieu hao lech toi da +/-10%

CSV_FIELDNAMES = [
    "class_id",
    "class_name",
    "brand",
    "model",
    "body_style",
    "year",
    "seats",
    "segment",
    "price_million_vnd",
    "fuel_l_per_100km",
    "data_source",
]


def parse_brand(class_name: str) -> str:
    """Tach ten hang xe tu ten lop."""
    for brand in MULTI_WORD_BRANDS:
        if class_name.startswith(brand):
            return brand
    return class_name.split()[0]


def parse_year(class_name: str) -> int:
    """Tach nam san xuat (4 chu so cuoi ten lop)."""
    year_token = class_name.split()[-1]
    if not (year_token.isdigit() and len(year_token) == 4):
        raise ValueError(f"Khong tim thay nam trong ten lop: {class_name!r}")
    return int(year_token)


def parse_model(class_name: str, brand: str, year: int) -> str:
    """Tach ten dong xe: bo ten hang, nam va tu khoa kieu dang.

    Kieu dang nam GIUA ten lop ("Acura RL Sedan 2012") nen phai loai rieng,
    neu khong cot `model` se lap lai thong tin cua cot `body_style`.
    """
    middle = class_name[len(brand):-len(str(year))].strip()

    # Loai tu khoa kieu dang dai truoc, tranh "Crew Cab" chi bi cat "Cab".
    for keyword, _ in BODY_STYLE_RULES:
        if middle.endswith(keyword):
            middle = middle[:-len(keyword)].strip()
            break

    # Vai lop chi gom hang + kieu dang (vd: "Volvo C30 Hatchback 2012" thi
    # con "C30", nhung "Chevrolet Express Van 2007" co the rong).
    return middle or class_name[len(brand):-len(str(year))].strip()


def resolve_body_style(class_name: str) -> tuple[str, int]:
    """Suy kieu dang va so cho tu ten lop.

    Uu tien bang anh xa tuong minh, sau do duyet BODY_STYLE_RULES theo
    thu tu (tu khoa dai truoc tu khoa ngan).
    """
    if class_name in SPECIAL_BODY_STYLES:
        body_style, seats = SPECIAL_BODY_STYLES[class_name]
    else:
        for keyword, (body_style, seats) in BODY_STYLE_RULES:
            if keyword in class_name:
                break
        else:
            raise ValueError(
                f"Khong suy duoc kieu dang cho lop: {class_name!r}. "
                "Hay bo sung vao SPECIAL_BODY_STYLES."
            )

    # Ghi de so cho cho cac dong 2 cho.
    if any(kw in class_name for kw in TWO_SEATER_KEYWORDS):
        seats = 2
    return body_style, seats


def resolve_segment(class_name: str, brand: str) -> str:
    """Xac dinh phan khuc gia: uu tien ghi de, sau do suy tu hang xe."""
    if class_name in SEGMENT_OVERRIDES:
        return SEGMENT_OVERRIDES[class_name]
    if brand in EXOTIC_BRANDS:
        return "exotic"
    if brand in LUXURY_BRANDS:
        return "luxury"
    return "economy"


def model_variation(class_name: str, spread: float) -> float:
    """Sinh he so bien thien rieng cho tung dong xe.

    Neu chi dua vao (phan khuc, kieu dang, nam) thi rat nhieu xe se co gia
    y het nhau — 196 xe chi con 87 to hop khac nhau, khien ket qua tu van
    trong nhu xep hang tuy tien.

    Ham nay tao do lech rieng cho tung ten xe, nam trong khoang
    [1 - spread, 1 + spread]. Dung hash cua ten xe nen ket qua **luon tai
    lap duoc**, khong phai so ngau nhien.
    """
    # md5 cho gia tri on dinh giua cac lan chay, khac voi hash() cua Python.
    digest = hashlib.md5(class_name.encode("utf-8")).hexdigest()
    # Lay 6 chu so hex dau -> so nguyen -> dua ve khoang [-1, 1].
    unit = int(digest[:6], 16) / 0xFFFFFF * 2 - 1
    return 1 + unit * spread


def estimate_price(
    class_name: str, segment: str, body_style: str, year: int
) -> float:
    """Uoc luong gia ban (trieu VND) - DU LIEU MO PHONG.

    gia = co_so(phan_khuc) * he_so(kieu_dang) * khau_hao(nam) * bien_thien
    """
    base = BASE_PRICE_BY_SEGMENT[segment]
    factor = PRICE_FACTOR_BY_BODY[body_style]
    age = max(REFERENCE_YEAR - year, 0)
    depreciation = (1 - PRICE_DEPRECIATION_PER_YEAR) ** age
    variation = model_variation(class_name, PRICE_VARIATION_SPREAD)
    return round(base * factor * depreciation * variation, 1)


def estimate_fuel(
    class_name: str, segment: str, body_style: str, year: int
) -> float:
    """Uoc luong muc tieu hao (L/100km) - DU LIEU MO PHONG.

    Xe cang cu cang ton nhien lieu so voi moc quy chieu 2012.
    """
    base = BASE_FUEL_BY_BODY[body_style]
    factor = FUEL_FACTOR_BY_SEGMENT[segment]
    age = max(REFERENCE_YEAR - year, 0)
    penalty = (1 + FUEL_IMPROVEMENT_PER_YEAR) ** age
    variation = model_variation(class_name + "fuel", FUEL_VARIATION_SPREAD)
    return round(base * factor * penalty * variation, 1)


def build_spec_row(class_id: int, class_name: str) -> dict:
    """Dung mot dong thong so day du tu ten lop."""
    brand = parse_brand(class_name)
    year = parse_year(class_name)
    model = parse_model(class_name, brand, year)
    body_style, seats = resolve_body_style(class_name)
    segment = resolve_segment(class_name, brand)

    return {
        "class_id": class_id,
        "class_name": class_name,
        "brand": brand,
        "model": model,
        "body_style": body_style,
        "year": year,
        "seats": seats,
        "segment": segment,
        "price_million_vnd": estimate_price(
            class_name, segment, body_style, year
        ),
        "fuel_l_per_100km": estimate_fuel(
            class_name, segment, body_style, year
        ),
        # Cac truong suy tu ten lop la chinh xac, rieng price/fuel mo phong.
        "data_source": "derived+simulated",
    }


def load_class_names(path: Path) -> list[str]:
    """Doc file danh sach ten lop, moi dong mot ten."""
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Khong tim thay file danh sach lop: {path}. "
            "Xem docs/car_specs_generation.md de biet cach tao lai."
        ) from None

    names = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(names) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"File {path.name} co {len(names)} lop, "
            f"ky vong {EXPECTED_CLASS_COUNT}."
        )
    return names


def write_csv(rows: list[dict], path: Path) -> None:
    """Ghi danh sach thong so ra CSV (UTF-8 co BOM de Excel doc dung)."""
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], field: str) -> dict[str, int]:
    """Dem so luong dong theo tung gia tri cua mot cot."""
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row[field])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sinh bang thong so xe tu danh sach lop Stanford Cars."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Duong dan file CSV dau ra (mac dinh: {OUTPUT_FILE.name})",
    )
    args = parser.parse_args()

    setup_logging()

    class_names = load_class_names(CLASSES_FILE)
    logger.info("Da doc %d ten lop tu %s", len(class_names), CLASSES_FILE.name)

    rows = [
        build_spec_row(class_id, name)
        for class_id, name in enumerate(class_names, start=1)
    ]

    write_csv(rows, args.output)
    logger.info("Da ghi %d dong vao %s", len(rows), args.output)
    logger.info("Phan khuc: %s", summarize(rows, "segment"))
    logger.info("Kieu dang: %s", summarize(rows, "body_style"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
