"""Kiem thu ban ONNX cua module tu van.

Cac test tu bo qua neu chua export mo hinh ONNX.

Chay bang: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rec.onnx_recommender import ONNX_MODEL_PATH  # noqa: E402
from src.rec.recommender import CarRecommender  # noqa: E402

def sample_cars(recommender, count: int = 3) -> list[str]:
    """Lay vai ten xe co that tu bang thong so.

    Khong viet cung ten xe: ten lop co the doi (vi du bo nam san xuat),
    test se hong ma khong phai vi loi that.
    """
    return recommender.specs["class_name"].head(count).tolist()


@unittest.skipUnless(
    ONNX_MODEL_PATH.exists(),
    "Chua co mo hinh ONNX. Chay scripts/export_recommender_onnx.py",
)
class TestOnnxRecommender(unittest.TestCase):
    """So sanh ban ONNX voi ban Pandas."""

    @classmethod
    def setUpClass(cls):
        from src.rec.onnx_recommender import OnnxRecommender

        cls.pandas_rec = CarRecommender()
        cls.samples = sample_cars(cls.pandas_rec)
        cls.target = cls.samples[0]
        cls.onnx_rec = OnnxRecommender(base=cls.pandas_rec)

    def test_cho_ket_qua_giong_ban_pandas(self):
        """Hai ban phai tra ve cung danh sach xe, cung thu tu.

        Neu khac nhau thi so lieu do thoi gian tro nen vo nghia.
        """
        for name in self.samples:
            with self.subTest(car=name):
                expected = [
                    r.class_name
                    for r in self.pandas_rec.recommend_similar(name)
                ]
                actual = [
                    r.class_name
                    for r in self.onnx_rec.recommend_similar(name)
                ]
                self.assertEqual(expected, actual)

    def test_khong_chua_chinh_xe_truy_van(self):
        target = self.target
        names = [
            r.class_name for r in self.onnx_rec.recommend_similar(target)
        ]
        self.assertNotIn(target, names)

    def test_sap_xep_giam_dan_theo_diem(self):
        scores = [
            r.score
            for r in self.onnx_rec.recommend_similar(self.target)
        ]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_gioi_han_so_luong(self):
        results = self.onnx_rec.recommend_similar(
            self.target, top_n=3
        )
        self.assertLessEqual(len(results), 3)

    def test_ten_xe_khong_ton_tai(self):
        with self.assertRaises(KeyError):
            self.onnx_rec.recommend_similar("Xe Khong Co That 2099")


if __name__ == "__main__":
    unittest.main()
