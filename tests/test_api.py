"""Kiem thu REST API.

Cac test can mo hinh ONNX se tu bo qua neu thu muc models/ con trong.

Chay bang: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cv.pipeline import CLASSIFIER_MODEL, DETECTOR_MODEL  # noqa: E402

try:
    from fastapi.testclient import TestClient

    from src.api.main import app
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

MODELS_READY = DETECTOR_MODEL.exists() and CLASSIFIER_MODEL.exists()

# Anh JPEG nho nhat hop le, dung de kiem tra viec nhan dien dinh dang.
TINY_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300"
    "ffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffd9"
)


@unittest.skipUnless(API_AVAILABLE, "Chua cai fastapi/httpx")
class TestHealthEndpoints(unittest.TestCase):
    """Endpoint kiem tra suc khoe, khong can xac thuc."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health_luon_tra_200(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_ready_bao_so_luong_xe(self):
        response = self.client.get("/ready")
        # It nhat 196 xe quoc te; nhieu hon neu co du lieu xe VN.
        self.assertGreaterEqual(response.json()["car_count"], 196)

    def test_root_tra_ve_duong_dan_tai_lieu(self):
        response = self.client.get("/")
        self.assertEqual(response.json()["docs"], "/docs")

    def test_co_tai_lieu_openapi(self):
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200)
        self.assertIn("/api/v1/recognize", response.json()["paths"])


@unittest.skipUnless(API_AVAILABLE, "Chua cai fastapi/httpx")
class TestCarsEndpoints(unittest.TestCase):
    """Endpoint tra cuu danh muc — khong phu thuoc mo hinh ONNX."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_liet_ke_va_phan_trang(self):
        response = self.client.get("/api/v1/cars?limit=5&offset=10")
        data = response.json()
        self.assertGreaterEqual(data["total"], 196)
        self.assertEqual(len(data["cars"]), 5)
        self.assertEqual(data["offset"], 10)

    def test_loc_khong_phan_biet_hoa_thuong(self):
        lower = self.client.get("/api/v1/cars?brand=tesla").json()["total"]
        upper = self.client.get("/api/v1/cars?brand=TESLA").json()["total"]
        self.assertEqual(lower, upper)
        self.assertGreater(lower, 0)

    def test_limit_vuot_gioi_han_bi_tu_choi(self):
        response = self.client.get("/api/v1/cars?limit=9999")
        self.assertEqual(response.status_code, 422)

    def test_lay_gia_tri_de_loc(self):
        data = self.client.get("/api/v1/cars/filters").json()
        self.assertIn("Sedan", data["body_styles"])
        self.assertLess(data["price_min"], data["price_max"])

    def test_tra_cuu_mot_dong_xe(self):
        response = self.client.get("/api/v1/cars/Tesla Model S Sedan 2012")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["brand"], "Tesla")

    def test_ten_xe_khong_ton_tai_tra_404(self):
        response = self.client.get("/api/v1/cars/Xe Khong Co That 2099")
        self.assertEqual(response.status_code, 404)

    def test_tim_xe_tuong_tu(self):
        response = self.client.get(
            "/api/v1/cars/Tesla Model S Sedan 2012/similar?top_n=3"
        )
        data = response.json()
        self.assertEqual(data["count"], 3)
        names = [car["class_name"] for car in data["cars"]]
        self.assertNotIn("Tesla Model S Sedan 2012", names)


@unittest.skipUnless(API_AVAILABLE, "Chua cai fastapi/httpx")
class TestRecommendEndpoint(unittest.TestCase):
    """Endpoint goi y theo nhu cau."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_ton_trong_rang_buoc_cung(self):
        """Xe khong thoa man phai bi loai han, khong phai chi tru diem."""
        response = self.client.post(
            "/api/v1/recommend",
            json={"max_price": 800, "min_seats": 7, "top_n": 5},
        )
        for car in response.json()["cars"]:
            self.assertLessEqual(car["price_million_vnd"], 800)
            self.assertGreaterEqual(car["seats"], 7)

    def test_khoang_gia_nguoc_bi_tu_choi(self):
        response = self.client.post(
            "/api/v1/recommend",
            json={"min_price": 500, "max_price": 100},
        )
        self.assertEqual(response.status_code, 422)

    def test_gia_am_bi_tu_choi(self):
        response = self.client.post(
            "/api/v1/recommend", json={"max_price": -100}
        )
        self.assertEqual(response.status_code, 422)

    def test_khong_co_ket_qua_tra_danh_sach_rong(self):
        """Dieu kien qua chat thi tra rong, khong phai loi."""
        response = self.client.post(
            "/api/v1/recommend", json={"max_price": 1}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 0)

    def test_sap_xep_giam_dan_theo_diem(self):
        response = self.client.post(
            "/api/v1/recommend", json={"max_price": 5000, "top_n": 10}
        )
        scores = [car["score"] for car in response.json()["cars"]]
        self.assertEqual(scores, sorted(scores, reverse=True))


@unittest.skipUnless(API_AVAILABLE, "Chua cai fastapi/httpx")
class TestRecognizeValidation(unittest.TestCase):
    """Kiem tra dau vao cua endpoint nhan dien."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_tu_choi_file_gia_mao_content_type(self):
        """Khong tin content-type client khai bao, kiem tra byte dau file."""
        response = self.client.post(
            "/api/v1/recognize",
            files={"file": ("x.jpg", b"day khong phai anh", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 415)

    def test_tu_choi_file_rong(self):
        response = self.client.post(
            "/api/v1/recognize",
            files={"file": ("x.jpg", b"", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 400)

    def test_top_k_vuot_gioi_han_bi_tu_choi(self):
        response = self.client.post(
            "/api/v1/recognize?top_k=999",
            files={"file": ("x.jpg", TINY_JPEG, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 422)

    def test_nhan_dang_duoc_chu_ky_anh(self):
        from src.api.routers.recognition import detect_image_format

        cases = [
            (b"\xff\xd8\xff\xe0", "JPEG"),
            (b"\x89PNG\r\n\x1a\n", "PNG"),
            (b"BM\x00\x00", "BMP"),
            (b"GIF89a\x00", "GIF"),
            (b"RIFF\x00\x00\x00\x00WEBP", "WEBP"),
            (b"khong phai anh", None),
        ]
        for data, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(detect_image_format(data), expected)


@unittest.skipUnless(
    API_AVAILABLE and MODELS_READY,
    "Chua co mo hinh ONNX trong models/",
)
class TestRecognizeWithModels(unittest.TestCase):
    """Nhan dien that — chi chay khi da co mo hinh."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    @staticmethod
    def make_image_bytes():
        import cv2
        import numpy as np

        image = np.full((480, 640, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (160, 160), (480, 320), (120, 60, 40), -1)
        return cv2.imencode(".jpg", image)[1].tobytes()

    def test_tra_ve_dung_cau_truc(self):
        response = self.client.post(
            "/api/v1/recognize?top_k=3",
            files={"file": ("xe.jpg", self.make_image_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        for field in ("vehicle_count", "likely_not_a_car", "vehicles",
                      "processing_ms"):
            self.assertIn(field, data)

        if data["vehicles"]:
            vehicle = data["vehicles"][0]
            self.assertLessEqual(len(vehicle["predictions"]), 3)
            # model_index la chi so cua mo hinh, KHAC class_id cua bang
            # thong so — hai he danh so khac nhau.
            self.assertIn("model_index", vehicle["predictions"][0])

    def test_xac_suat_giam_dan(self):
        response = self.client.post(
            "/api/v1/recognize?top_k=5",
            files={"file": ("xe.jpg", self.make_image_bytes(), "image/jpeg")},
        )
        for vehicle in response.json()["vehicles"]:
            scores = [p["confidence"] for p in vehicle["predictions"]]
            self.assertEqual(scores, sorted(scores, reverse=True))


if __name__ == "__main__":
    unittest.main()
