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

# Mo hinh 1: xe quoc te (Stanford Cars, 196 lop, doi <=2012).
CLASSIFIER_MODEL = MODELS_DIR / "car_classifier.onnx"
CLASS_NAMES_FILE = MODELS_DIR / "class_names.json"

# Mo hinh 2: xe thi truong Viet Nam (20 lop, doi 2021-2024).
# Tuy chon — thieu file nay thi he thong van chay voi mo hinh 1.
VN_CLASSIFIER_MODEL = MODELS_DIR / "vn_car_classifier.onnx"
VN_CLASS_NAMES_FILE = MODELS_DIR / "vn_class_names.json"

# Noi rong hop bao truoc khi crop, giong luc huan luyen (xem notebook).
CROP_PADDING = 0.08

# Nguong xac suat de coi ket qua phan loai la dang tin.
#
# Mo hinh phan loai LUON tra ve mot trong 196 lop, ke ca khi anh dau vao
# khong phai o to. Voi anh la (con meo, phong canh...), phan bo xac suat
# thuong rat deu - khong lop nao noi troi han. Nguong nay giup phat hien
# truong hop do de bao "khong nhan ra" thay vi doan bua.
MIN_CONFIDENCE = 0.15

# He so phat khi so sanh do tin cay giua hai mo hinh khac so lop.
#
# Mo hinh xe VN chi co 20 lop, mo hinh quoc te co 196 lop. Voi cung mot
# muc "chac chan", mo hinh it lop luon cho xac suat cao hon — doan mo o
# 20 lop la 5%, o 196 lop chi 0.5%. So sanh truc tiep se luon thien vi
# mo hinh VN.
#
# Gia tri 0.40 chon sau khi do tren 326 anh xe VN + 18 anh xe quoc te.
# Khong co he so nao tot cho ca hai chieu — day la mot danh doi that:
#
#   He so   VN top-1   QT top-1
#    0.30     47.5%      55.6%
#    0.40     54.6%      55.6%   <- can bang nhat
#    0.75     67.8%      44.4%
#    1.00     70.9%      38.9%
#
# 0.40 cho hai chieu gan bang nhau. Tu 0.50 tro len, xe quoc te tut nhanh
# vi mo hinh VN thang o qua nhieu truong hop.
#
# Xem docs/dual_model.md muc 2.
VN_CONFIDENCE_PENALTY = 0.40


@dataclass(frozen=True)
class RecognitionResult:
    """Ket qua nhan dien cho MOT chiec xe trong anh."""

    detection: Detection | None
    predictions: list[Prediction]
    crop: np.ndarray

    # Mo hinh nao dua ra ket qua nay: "international" (Stanford Cars,
    # 196 lop) hoac "vietnam" (20 lop xe thi truong VN).
    source: str = "international"

    # Ket qua cua mo hinh CON LAI, de nguoi dung doi chieu khi can.
    # None khi chi chay mot mo hinh.
    alternative: list[Prediction] | None = None

    @property
    def best(self) -> Prediction:
        """Du doan co xac suat cao nhat."""
        return self.predictions[0]

    @property
    def is_whole_image(self) -> bool:
        """True neu phan loai ca buc anh (khong detect duoc xe nao)."""
        return self.detection is None

    @property
    def is_confident(self) -> bool:
        """True neu ket qua du tin cay de hien thi nhu mot ket luan.

        Mo hinh luon tra ve mot lop nao do, ke ca voi anh khong phai o to.
        Kiem tra nay giup phan biet "nhan ra xe" voi "doan bua".
        """
        return self.best.confidence >= MIN_CONFIDENCE

    @property
    def is_likely_not_a_car(self) -> bool:
        """True khi nhieu kha nang anh khong chua o to.

        Hai dau hieu cung xuat hien: tang phat hien khong tim thay xe nao,
        VA tang phan loai cung khong chac chan.
        """
        return self.is_whole_image and not self.is_confident


class RecognitionPipeline:
    """Nhan dien dong xe tu anh: phat hien roi phan loai."""

    def __init__(
        self,
        detector_path: Path = DETECTOR_MODEL,
        classifier_path: Path = CLASSIFIER_MODEL,
        labels_path: Path = CLASS_NAMES_FILE,
        vn_classifier_path: Path = VN_CLASSIFIER_MODEL,
        vn_labels_path: Path = VN_CLASS_NAMES_FILE,
    ) -> None:
        self.detector = VehicleDetector(detector_path)
        self.classifier = CarClassifier(classifier_path, labels_path)

        # Mo hinh xe Viet Nam la TUY CHON: thieu file thi he thong van
        # chay binh thuong voi mo hinh quoc te.
        self.vn_classifier: CarClassifier | None = None
        if vn_classifier_path.exists() and vn_labels_path.exists():
            self.vn_classifier = CarClassifier(
                vn_classifier_path, vn_labels_path, expected_classes=None
            )
        else:
            logger.info(
                "Khong co mo hinh xe Viet Nam, chi dung mo hinh quoc te."
            )

    def _classify(
        self, image: np.ndarray, top_k: int
    ) -> tuple[list[Prediction], str, list[Prediction] | None]:
        """Phan loai bang ca hai mo hinh, chon ket qua cua MOT mo hinh.

        Hai mo hinh phu hai tap xe gan nhu khong giao nhau (quoc te doi
        <=2012, Viet Nam doi 2021-2024). Mo hinh nao tu tin hon thi lay
        TRON top-k cua mo hinh do.

        ## Vi sao khong gop chung danh sach

        Da thu gop (tron top-k cua hai mo hinh roi xep hang chung) va do
        tren 18 anh xe quoc te + 326 anh xe VN:

            Cach lam            QT top-1   VN top-1
            Chi mo hinh QT        61.1%       --
            Gop, phat 0.75        44.4%     67.8%
            Gop, phat 0.40        55.6%     54.6%
            Chon mot mo hinh      61.1%     67.8%

        Gop luon lam xe quoc te te di, vi cac lop xe VN vo nghia chen vao
        top-5 (o he so 0.75, 67.8% o trong top-5 cua anh xe quoc te bi xe
        VN chiem). Ha he so phat thi lai lam hong chieu nguoc lai.

        Chon mot mo hinh giu duoc do chinh xac tot nhat ca hai chieu.

        ## Van de khi so sanh xac suat

        Xac suat cua hai mo hinh khong so sanh truc tiep duoc: doan mo o
        20 lop cho 5%, o 196 lop chi cho 0.5%. Nhan xac suat cua mo hinh
        VN voi VN_CONFIDENCE_PENALTY truoc khi so.
        """
        primary = self.classifier.predict(image, top_k=top_k)

        if self.vn_classifier is None:
            return primary, "international", None

        vn_predictions = self.vn_classifier.predict(image, top_k=top_k)
        vn_score = vn_predictions[0].confidence * VN_CONFIDENCE_PENALTY

        if vn_score > primary[0].confidence:
            return vn_predictions, "vietnam", primary
        return primary, "international", vn_predictions

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
            predictions, source, alternative = self._classify(image, top_k)
            return [RecognitionResult(
                detection=None,
                predictions=predictions,
                crop=image,
                source=source,
                alternative=alternative,
            )]

        results = []
        for detection in detections[:max_vehicles]:
            crop = detection.crop(image, padding=CROP_PADDING)
            if crop.size == 0:
                continue
            predictions, source, alternative = self._classify(crop, top_k)
            results.append(RecognitionResult(
                detection=detection,
                predictions=predictions,
                crop=crop,
                source=source,
                alternative=alternative,
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
