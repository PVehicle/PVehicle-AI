"""Ghep hai tang thanh mot luong nhan dien hoan chinh.

Luong xu ly: anh -> YOLOv8 tim vung xe -> crop -> phan loai dong xe.

Neu YOLOv8 khong tim thay xe nao, pipeline se phan loai toan bo buc anh.
Truong hop nay xay ra khi anh da duoc cat san sat vao xe (nhu anh trong
Stanford Cars), luc do tang phat hien khong con gi de lam.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.cv.classifier import CarClassifier, Prediction
from src.cv.detector import Detection, VehicleDetector
from src.utils import MODELS_DIR, get_logger

logger = get_logger(__name__)

DETECTOR_MODEL = MODELS_DIR / "yolov8n.onnx"
CLASSIFIER_MODEL = MODELS_DIR / "car_classifier.onnx"
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"

# Noi rong hop bao truoc khi crop, giong luc huan luyen (xem notebook).
CROP_PADDING = 0.08


@dataclass(frozen=True)
class RecognitionResult:
    """Ket qua nhan dien cho MOT chiec xe trong anh."""

    detection: Detection | None
    predictions: list[Prediction]
    crop: np.ndarray

    @property
    def best(self) -> Prediction:
        """Du doan co xac suat cao nhat."""
        return self.predictions[0]

    @property
    def is_whole_image(self) -> bool:
        """True neu phan loai ca buc anh (khong detect duoc xe nao)."""
        return self.detection is None


class RecognitionPipeline:
    """Nhan dien dong xe tu anh: phat hien roi phan loai."""

    def __init__(
        self,
        detector_path: Path = DETECTOR_MODEL,
        classifier_path: Path = CLASSIFIER_MODEL,
        labels_path: Path = CLASS_NAMES_FILE,
    ) -> None:
        self.detector = VehicleDetector(detector_path)
        self.classifier = CarClassifier(classifier_path, labels_path)

    def recognize(
        self, image: np.ndarray, top_k: int = 5, max_vehicles: int = 5
    ) -> list[RecognitionResult]:
        """Nhan dien tat ca xe trong anh.

        Ket qua sap xep theo dien tich vung xe giam dan (xe lon nhat truoc).
        """
        detections = self.detector.detect(image)

        if not detections:
            # Khong tim thay xe: coi ca buc anh la vung can phan loai.
            logger.info("Khong phat hien xe, phan loai toan bo anh.")
            return [RecognitionResult(
                detection=None,
                predictions=self.classifier.predict(image, top_k=top_k),
                crop=image,
            )]

        results = []
        for detection in detections[:max_vehicles]:
            crop = detection.crop(image, padding=CROP_PADDING)
            if crop.size == 0:
                continue
            results.append(RecognitionResult(
                detection=detection,
                predictions=self.classifier.predict(crop, top_k=top_k),
                crop=crop,
            ))
        return results


def load_image(source: str | Path | bytes) -> np.ndarray:
    """Doc anh tu duong dan hoac tu du lieu nhi phan, tra ve mang BGR."""
    if isinstance(source, bytes):
        buffer = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Khong tim thay anh: {path}")
        # imdecode thay cho imread de doc duoc duong dan co ky tu tieng Viet.
        image = cv2.imdecode(
            np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR
        )

    if image is None:
        raise ValueError("Khong doc duoc anh: dinh dang khong hop le.")
    return image


def draw_detections(
    image: np.ndarray, results: list[RecognitionResult]
) -> np.ndarray:
    """Ve hop bao va ten xe len anh, tra ve ban sao da ve."""
    canvas = image.copy()
    box_color = (0, 200, 0)
    text_color = (255, 255, 255)

    for result in results:
        if result.detection is None:
            continue

        det = result.detection
        cv2.rectangle(canvas, (det.x1, det.y1), (det.x2, det.y2),
                      box_color, 2)

        label = f"{result.best.class_name} ({result.best.confidence:.0%})"
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        # Dat nhan phia tren hop; neu sat mep tren thi day xuong trong hop.
        text_top = max(det.y1 - text_h - 8, 0)
        cv2.rectangle(
            canvas,
            (det.x1, text_top),
            (det.x1 + text_w + 8, text_top + text_h + 8),
            box_color, -1,
        )
        cv2.putText(
            canvas, label, (det.x1 + 4, text_top + text_h + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA,
        )
    return canvas
