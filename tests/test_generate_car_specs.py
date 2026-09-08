"""Kiem thu logic sinh bang thong so xe.

Chay bang: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_car_specs import (  # noqa: E402
    BODY_STYLE_RULES,
    EXPECTED_CLASS_COUNT,
    SEGMENT_OVERRIDES,
    SPECIAL_BODY_STYLES,
    build_spec_row,
    load_class_names,
    parse_brand,
    parse_model,
    parse_year,
    resolve_body_style,
    resolve_segment,
)
from src.utils import DATA_DIR  # noqa: E402


class TestParsing(unittest.TestCase):
    """Kiem tra viec tach hang / nam / dong xe tu ten lop."""

    def test_brand_mot_tu(self):
        self.assertEqual(parse_brand("Acura RL Sedan 2012"), "Acura")

    def test_brand_nhieu_tu(self):
        self.assertEqual(
            parse_brand("Land Rover Range Rover SUV 2012"), "Land Rover"
        )
        self.assertEqual(
            parse_brand("AM General Hummer SUV 2000"), "AM General"
        )

    def test_year(self):
        self.assertEqual(parse_year("Volvo 240 Sedan 1993"), 1993)

    def test_year_khong_hop_le(self):
        with self.assertRaises(ValueError):
            parse_year("Xe khong co nam")

    def test_model_bo_kieu_dang(self):
        """Cot model khong duoc lap lai thong tin cua cot body_style."""
        self.assertEqual(
            parse_model("Acura RL Sedan 2012", "Acura", 2012), "RL"
        )
        self.assertEqual(
            parse_model("Land Rover Range Rover SUV 2012", "Land Rover", 2012),
            "Range Rover",
        )

    def test_model_khong_bao_gio_rong(self):
        for name in load_class_names(DATA_DIR / "stanford_cars_classes.txt"):
            brand = parse_brand(name)
            year = parse_year(name)
            self.assertTrue(
                parse_model(name, brand, year),
                f"Model rong voi lop: {name}",
            )


class TestBodyStyle(unittest.TestCase):
    """Kiem tra suy luan kieu dang va so cho ngoi."""

    def test_uu_tien_tu_khoa_dai(self):
        """"Crew Cab" phai khop truoc "Cab"."""
        body, seats = resolve_body_style("Ford F-450 Super Duty Crew Cab 2012")
        self.assertEqual(body, "Pickup")
        self.assertEqual(seats, 5)

    def test_regular_cab_3_cho(self):
        _, seats = resolve_body_style("Chevrolet Silverado Regular Cab 2012")
        self.assertEqual(seats, 3)

    def test_xe_2_cho_ghi_de_so_cho(self):
        _, seats = resolve_body_style("Chevrolet Corvette ZR1 2012")
        self.assertEqual(seats, 2)

    def test_lop_dac_biet(self):
        body, _ = resolve_body_style("Chevrolet TrailBlazer SS 2009")
        self.assertEqual(body, "SUV")

    def test_lop_khong_suy_duoc_thi_bao_loi(self):
        with self.assertRaises(ValueError):
            resolve_body_style("Ten Xe La 2012")


class TestSegment(unittest.TestCase):
    """Kiem tra phan khuc gia."""

    def test_theo_hang(self):
        cases = [
            ("Acura RL Sedan 2012", "Acura", "luxury"),
            ("Ferrari FF Coupe 2012", "Ferrari", "exotic"),
            ("Honda Odyssey Minivan 2012", "Honda", "economy"),
        ]
        for class_name, brand, expected in cases:
            with self.subTest(brand=brand):
                self.assertEqual(
                    resolve_segment(class_name, brand), expected
                )

    def test_ghi_de_thang_hang_pho_thong(self):
        """Corvette ZR1 la hang pho thong nhung gia sieu xe."""
        self.assertEqual(
            resolve_segment("Chevrolet Corvette ZR1 2012", "Chevrolet"),
            "exotic",
        )


class TestBangDuLieu(unittest.TestCase):
    """Kiem tra tinh nhat quan cua cac bang cau hinh."""

    def test_du_196_lop(self):
        names = load_class_names(DATA_DIR / "stanford_cars_classes.txt")
        self.assertEqual(len(names), EXPECTED_CLASS_COUNT)

    def test_moi_ten_trong_bang_ghi_de_deu_ton_tai(self):
        """Tranh viet sai ten lop trong cac bang anh xa tuong minh."""
        names = set(load_class_names(DATA_DIR / "stanford_cars_classes.txt"))
        for table_name, table in (
            ("SPECIAL_BODY_STYLES", SPECIAL_BODY_STYLES),
            ("SEGMENT_OVERRIDES", SEGMENT_OVERRIDES),
        ):
            for key in table:
                self.assertIn(
                    key, names,
                    f"{table_name} chua lop khong ton tai: {key!r}",
                )

    def test_tu_khoa_dai_dung_truoc_tu_khoa_ngan(self):
        """Bao dam thu tu duyet trong BODY_STYLE_RULES la dung."""
        keywords = [kw for kw, _ in BODY_STYLE_RULES]
        for i, kw in enumerate(keywords):
            for later in keywords[i + 1:]:
                self.assertNotIn(
                    kw, later,
                    f"{kw!r} dung truoc {later!r} se khop nham",
                )

    def test_moi_lop_deu_sinh_duoc_dong_du_lieu(self):
        """Chay het 196 lop, khong lop nao gay loi."""
        names = load_class_names(DATA_DIR / "stanford_cars_classes.txt")
        for class_id, name in enumerate(names, start=1):
            row = build_spec_row(class_id, name)
            self.assertGreater(row["price_million_vnd"], 0)
            self.assertGreater(row["fuel_l_per_100km"], 0)
            self.assertGreaterEqual(row["seats"], 2)


if __name__ == "__main__":
    unittest.main()
