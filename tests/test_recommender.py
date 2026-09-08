"""Kiem thu module tu van.

Chay bang: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_car_specs import model_variation  # noqa: E402
from src.rec.recommender import CarRecommender  # noqa: E402


class TestBienThien(unittest.TestCase):
    """Kiem tra ham sinh bien thien theo ten xe."""

    def test_luon_tai_lap_duoc(self):
        """Cung ten xe phai cho cung ket qua qua nhieu lan chay."""
        first = model_variation("Honda Odyssey Minivan 2012", 0.18)
        second = model_variation("Honda Odyssey Minivan 2012", 0.18)
        self.assertEqual(first, second)

    def test_nam_trong_khoang_cho_phep(self):
        names = [
            "Acura RL Sedan 2012",
            "Ferrari FF Coupe 2012",
            "smart fortwo Convertible 2012",
        ]
        for name in names:
            with self.subTest(name=name):
                value = model_variation(name, 0.18)
                self.assertGreaterEqual(value, 1 - 0.18)
                self.assertLessEqual(value, 1 + 0.18)

    def test_ten_khac_nhau_cho_ket_qua_khac_nhau(self):
        a = model_variation("Honda Odyssey Minivan 2012", 0.18)
        b = model_variation("Honda Odyssey Minivan 2007", 0.18)
        self.assertNotEqual(a, b)


class TestRecommender(unittest.TestCase):
    """Kiem tra logic goi y xe."""

    @classmethod
    def setUpClass(cls):
        cls.rec = CarRecommender()

    def test_nap_du_196_xe(self):
        self.assertEqual(len(self.rec.specs), 196)

    def test_moi_xe_co_dac_trung_rieng(self):
        """Tranh truong hop nhieu xe trung het thong so, khien ket qua
        tu van trong nhu xep hang tuy tien."""
        combos = self.rec.specs.groupby([
            "price_million_vnd", "seats", "fuel_l_per_100km",
            "body_style", "segment",
        ])
        self.assertEqual(len(combos), 196)

    def test_ton_trong_ngan_sach(self):
        """Dieu kien loc la rang buoc cung, khong duoc goi y xe vuot gia."""
        budget = 800
        for item in self.rec.recommend_by_needs(max_price=budget):
            self.assertLessEqual(item.price_million_vnd, budget)

    def test_ton_trong_so_cho(self):
        for item in self.rec.recommend_by_needs(min_seats=7):
            self.assertGreaterEqual(item.seats, 7)

    def test_ton_trong_kieu_dang(self):
        wanted = ["Sedan", "Hatchback"]
        for item in self.rec.recommend_by_needs(body_styles=wanted):
            self.assertIn(item.body_style, wanted)

    def test_ket_hop_nhieu_dieu_kien(self):
        results = self.rec.recommend_by_needs(
            max_price=900, min_seats=7, max_fuel=11.0
        )
        self.assertTrue(results, "Phai tim duoc it nhat mot xe")
        for item in results:
            self.assertLessEqual(item.price_million_vnd, 900)
            self.assertGreaterEqual(item.seats, 7)
            self.assertLessEqual(item.fuel_l_per_100km, 11.0)

    def test_khong_co_xe_thoa_man_tra_ve_rong(self):
        self.assertEqual(
            self.rec.recommend_by_needs(max_price=1, min_seats=9), []
        )

    def test_gioi_han_so_luong_ket_qua(self):
        self.assertLessEqual(len(self.rec.recommend_by_needs(top_n=3)), 3)

    def test_xe_tuong_tu_khong_chua_chinh_no(self):
        target = "BMW M3 Coupe 2012"
        names = [r.class_name for r in self.rec.recommend_similar(target)]
        self.assertNotIn(target, names)

    def test_xe_tuong_tu_cung_kieu_dang(self):
        """Kieu dang co trong so cao nen ket qua phai cung loai."""
        results = self.rec.recommend_similar("Honda Odyssey Minivan 2012")
        self.assertEqual(results[0].body_style, "Minivan")

    def test_xe_tuong_tu_sap_xep_giam_dan(self):
        scores = [
            r.score
            for r in self.rec.recommend_similar("BMW M3 Coupe 2012")
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ten_xe_khong_ton_tai(self):
        with self.assertRaises(KeyError):
            self.rec.recommend_similar("Xe Khong Co That 2099")

    def test_get_car(self):
        row = self.rec.get_car("BMW M3 Coupe 2012")
        self.assertIsNotNone(row)
        self.assertEqual(row["body_style"], "Coupe")
        self.assertIsNone(self.rec.get_car("Xe Khong Co That 2099"))

    def test_thuoc_tinh_ho_tro_giao_dien(self):
        self.assertIn("Sedan", self.rec.body_styles)
        low, high = self.rec.price_range
        self.assertLess(low, high)


if __name__ == "__main__":
    unittest.main()
