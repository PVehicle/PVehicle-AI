"""Tang 2: phan loai dong xe (196 lop) bang mo hinh ONNX.

Nhan anh da duoc crop tu tang phat hien, tra ve cac dong xe co kha nang
nhat. Mo hinh la EfficientNet-B0 huan luyen tren Stanford Cars, xem
`docs/training_classifier.md`.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.utils import get_logger

logger = get_logger(__name__)

# Phai khop voi cau hinh luc huan luyen (xem notebook Colab).
CLASSIFIER_INPUT_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EXPECTED_NUM_CLASSES = 196
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class Prediction:
    """Mot du doan dong xe kem do tin cay."""

    class_id: int
    class_name: str
    confidence: float


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Chuyen logits thanh xac suat.

    Tru gia tri lon nhat truoc khi mu hoa de tranh tran so.
    """
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class CarClassifier:
    """Phan loai dong xe tu anh da crop."""

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        expected_classes: int | None = EXPECTED_NUM_CLASSES,
    ) -> None:
        """Nap mo hinh phan loai.

        `expected_classes` de kiem tra file nhan co dung so lop khong.
        Truyen None de bo qua — dung cho cac mo hinh co so lop khac
        (vi du mo hinh xe Viet Nam).
        """
        if not model_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay mo hinh phan loai: {model_path}. "
                "Huan luyen bang notebooks/train_classifier_colab.ipynb "
                "roi chep file ONNX vao thu muc models/."
            )
        if not labels_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay file nhan: {labels_path}. "
                "File nay duoc sinh ra cung luc voi mo hinh ONNX."
            )

        self.class_names: list[str] = json.loads(
            labels_path.read_text(encoding="utf-8")
        )
        if (
            expected_classes is not None
            and len(self.class_names) != expected_classes
        ):
            raise ValueError(
                f"File nhan co {len(self.class_names)} lop, "
                f"ky vong {expected_classes}."
            )

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info(
            "Da nap mo hinh phan loai: %s (%d lop)",
            model_path.name, len(self.class_names),
        )

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Chuyen anh BGR thanh tensor dau vao.

        Buoc tien xu ly phai giong het luc danh gia trong notebook:
        resize -> RGB -> [0,1] -> chuan hoa theo thong ke ImageNet.
        """
        resized = cv2.resize(
            image,
            (CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        normalized = (rgb / 255.0 - IMAGENET_MEAN) / IMAGENET_STD

        # HWC -> CHW, them chieu batch.
        return normalized.transpose(2, 0, 1)[np.newaxis, ...]

    def predict(
        self, image: np.ndarray, top_k: int = DEFAULT_TOP_K
    ) -> list[Prediction]:
        """Du doan dong xe, tra ve `top_k` ket qua co xac suat cao nhat.

        Voi 196 lop rat giong nhau (vi du Audi S4 doi 2007 va 2012), top-5
        huu ich hon nhieu so voi chi lay ket qua dau tien.
        """
        if image is None or image.size == 0:
            raise ValueError("Anh dau vao rong.")

        tensor = self._preprocess(image)
        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        probabilities = _softmax(logits)

        # argsort tang dan, dao nguoc de lay cac gia tri lon nhat truoc.
        top_indices = probabilities.argsort()[::-1][:top_k]
        return [
            Prediction(
                class_id=int(idx),
                class_name=self.class_names[idx],
                confidence=float(probabilities[idx]),
            )
            for idx in top_indices
        ]
