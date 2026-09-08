"""Kiem thu luong nhan dien (dung mo hinh gia trong models/).

Cac test nay kiem tra code chay dung, KHONG kiem tra do chinh xac cua mo
hinh — mo hinh gia co trong so ngau nhien.

Chay bang: python -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cv.classifier import _softmax  # noqa: E402
from src.cv.detector import (  # noqa: E402
    COCO_VEHICLE_CLASSES,
    Detection,
    _letterbox,
    _non_max_suppression,
)
from src.cv.pipeline import (  # noqa: E402
    CLASS_NAMES_FILE,
    CLASSIFIER_MODEL,
    DETECTOR_MODEL,
    RecognitionPipeline,
    draw_detections,
    load_image,
)

MODELS_READY = (
    DETECTOR_MODEL.exists()
    and CLASSIFIER_MODEL.exists()
    and CLASS_NAMES_FILE.exists()
)


def make_test_image(width=640, height=480):
    """Anh gia: nen xam, mot hinh chu nhat dam o giua."""
    image = np.full((height, width, 3), 180, dtype=np.uint8)
    cv2.rectangle(image, (200, 180), (440, 330), (120, 60, 40), -1)
    return image


class TestLetterbox(unittest.TestCase):
    """Kiem tra viec resize giu ty le."""

    def test_ra_dung_khung_vuong(self):
        image = make_test_image(800, 400)
        padded, _, _, _ = _letterbox(image, size=640)
        self.assertEqual(padded.shape, (640, 640, 3))

    def test_giu_nguyen_ty_le(self):
        """Anh 800x400 (ty le 2:1) phai duoc thu nho dung ty le do."""
        image = make_test_image(800, 400)
        _, scale, pad_left, pad_top = _letterbox(image, size=640)
        self.assertAlmostEqual(scale, 640 / 800, places=5)
        self.assertEqual(pad_left, 0)
        self.assertEqual(pad_top, (640 - 320) // 2)

    def test_anh_vuong_khong_can_dem(self):
        _, _, pad_left, pad_top = _letterbox(make_test_image(500, 500))
        self.assertEqual((pad_left, pad_top), (0, 0))


class TestNMS(unittest.TestCase):
    """Kiem tra thuat toan loai hop trung nhau."""

    def test_danh_sach_rong(self):
        self.assertEqual(
            _non_max_suppression(np.empty((0, 4)), np.empty(0), 0.5), []
        )

    def test_giu_hop_diem_cao_nhat(self):
        """Hai hop chong nhau nhieu: chi giu hop diem cao hon."""
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=float)
        scores = np.array([0.9, 0.8])
        self.assertEqual(_non_max_suppression(boxes, scores, 0.5), [0])

    def test_giu_ca_hai_hop_tach_roi(self):
        boxes = np.array(
            [[0, 0, 50, 50], [200, 200, 250, 250]], dtype=float
        )
        scores = np.array([0.9, 0.8])
        self.assertEqual(
            sorted(_non_max_suppression(boxes, scores, 0.5)), [0, 1]
        )

    def test_sap_xep_theo_diem_khong_theo_thu_tu_dau_vao(self):
        """Hop diem cao dung sau van phai duoc giu lai."""
        boxes = np.array([[0, 0, 100, 100], [2, 2, 102, 102]], dtype=float)
        scores = np.array([0.3, 0.95])
        self.assertEqual(_non_max_suppression(boxes, scores, 0.5), [1])


class TestDetection(unittest.TestCase):
    """Kiem tra lop du lieu Detection."""

    def test_dien_tich(self):
        det = Detection(10, 20, 110, 120, 0.9, "car")
        self.assertEqual(det.area, 100 * 100)

    def test_dien_tich_hop_khong_hop_le(self):
        """Hop nguoc (x2 < x1) phai cho dien tich 0, khong am."""
        det = Detection(100, 100, 50, 50, 0.9, "car")
        self.assertEqual(det.area, 0)

    def test_crop_cat_dung_vung(self):
        image = make_test_image()
        det = Detection(100, 50, 300, 250, 0.9, "car")
        crop = det.crop(image)
        self.assertEqual(crop.shape[:2], (200, 200))

    def test_crop_khong_vuot_khung_anh(self):
        """Noi rong hop sat mep khong duoc gay loi hay cat am."""
        image = make_test_image(640, 480)
        det = Detection(0, 0, 640, 480, 0.9, "car")
        crop = det.crop(image, padding=0.5)
        self.assertEqual(crop.shape[:2], (480, 640))


class TestSoftmax(unittest.TestCase):
    """Kiem tra ham softmax."""

    def test_tong_bang_mot(self):
        result = _softmax(np.array([1.0, 2.0, 3.0]))
        self.assertAlmostEqual(result.sum(), 1.0, places=6)

    def test_khong_tran_so_voi_gia_tri_lon(self):
        """Logits rat lon van phai cho ket qua hop le, khong ra nan."""
        result = _softmax(np.array([1000.0, 1001.0, 1002.0]))
        self.assertAlmostEqual(result.sum(), 1.0, places=6)
        self.assertFalse(np.isnan(result).any())

    def test_giu_thu_tu(self):
        result = _softmax(np.array([1.0, 3.0, 2.0]))
        self.assertEqual(int(result.argmax()), 1)


class TestCocoClasses(unittest.TestCase):
    """Kiem tra chi so lop COCO."""

    def test_dung_chi_so_chuan(self):
        """Chi so lay tu cau hinh chinh thuc cua Ultralytics."""
        self.assertEqual(COCO_VEHICLE_CLASSES[2], "car")
        self.assertEqual(COCO_VEHICLE_CLASSES[5], "bus")
        self.assertEqual(COCO_VEHICLE_CLASSES[7], "truck")


class TestLoadImage(unittest.TestCase):
    """Kiem tra ham doc anh."""

    def test_doc_tu_bytes(self):
        image = make_test_image()
        encoded = cv2.imencode(".jpg", image)[1].tobytes()
        self.assertEqual(load_image(encoded).shape, image.shape)

    def test_file_khong_ton_tai(self):
        with self.assertRaises(FileNotFoundError):
            load_image("khong_co_file_nay.jpg")

    def test_du_lieu_khong_hop_le(self):
        with self.assertRaises(ValueError):
            load_image(b"day khong phai anh")


@unittest.skipUnless(
    MODELS_READY,
    "Chua co mo hinh trong models/. Chay scripts/make_dummy_models.py",
)
class TestPipeline(unittest.TestCase):
    """Kiem tra luong nhan dien hoan chinh."""

    @classmethod
    def setUpClass(cls):
        cls.pipeline = RecognitionPipeline()

    def test_tra_ve_ket_qua(self):
        results = self.pipeline.recognize(make_test_image())
        self.assertTrue(results)

    def test_du_so_luong_top_k(self):
        results = self.pipeline.recognize(make_test_image(), top_k=5)
        self.assertEqual(len(results[0].predictions), 5)

    def test_xac_suat_giam_dan(self):
        predictions = self.pipeline.recognize(make_test_image())[0]
        scores = [p.confidence for p in predictions.predictions]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_tong_xac_suat_khong_vuot_mot(self):
        predictions = self.pipeline.recognize(make_test_image())[0]
        total = sum(p.confidence for p in predictions.predictions)
        self.assertLessEqual(total, 1.0 + 1e-6)

    def test_anh_rong_bao_loi(self):
        with self.assertRaises(ValueError):
            self.pipeline.recognize(np.empty((0, 0, 3), dtype=np.uint8))

    def test_ve_khong_lam_doi_kich_thuoc(self):
        image = make_test_image()
        results = self.pipeline.recognize(image)
        self.assertEqual(draw_detections(image, results).shape, image.shape)

    def test_ve_khong_sua_anh_goc(self):
        image = make_test_image()
        original = image.copy()
        draw_detections(image, self.pipeline.recognize(image))
        np.testing.assert_array_equal(image, original)


class TestNguongTinCay(unittest.TestCase):
    """Kiem tra co che phat hien anh khong phai o to.

    Mo hinh phan loai luon tra ve mot trong 196 lop, ke ca khi anh dau vao
    la con meo hay phong canh. Cac co suy nay giup phan biet "nhan ra xe"
    voi "doan bua".
    """

    @staticmethod
    def make_result(confidence, has_detection):
        from src.cv.classifier import Prediction
        from src.cv.pipeline import RecognitionResult

        detection = (
            Detection(0, 0, 100, 100, 0.9, "car") if has_detection else None
        )
        return RecognitionResult(
            detection=detection,
            predictions=[Prediction(0, "Xe Mau 2012", confidence)],
            crop=make_test_image(100, 100),
        )

    def test_du_tin_cay_khi_vuot_nguong(self):
        self.assertTrue(self.make_result(0.85, True).is_confident)

    def test_khong_du_tin_cay_khi_duoi_nguong(self):
        self.assertFalse(self.make_result(0.02, True).is_confident)

    def test_canh_bao_khi_vua_khong_detect_vua_khong_chac(self):
        """Hai dau hieu cung xuat hien moi ket luan la khong phai o to."""
        self.assertTrue(self.make_result(0.02, False).is_likely_not_a_car)

    def test_khong_canh_bao_khi_detect_duoc_xe(self):
        """Detect duoc xe thi khong ket luan 'khong phai o to',
        du do tin cay phan loai thap."""
        self.assertFalse(self.make_result(0.02, True).is_likely_not_a_car)

    def test_khong_canh_bao_khi_phan_loai_chac_chan(self):
        """Anh da crop sat xe: khong detect duoc nhung phan loai chac chan."""
        self.assertFalse(self.make_result(0.90, False).is_likely_not_a_car)


if __name__ == "__main__":
    unittest.main()
