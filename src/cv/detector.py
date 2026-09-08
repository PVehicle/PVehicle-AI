"""Tang 1: phat hien vung chua xe trong anh bang YOLOv8 (ONNX Runtime).

Dung mo hinh YOLOv8 pretrained tren COCO, chi giu lai cac lop phuong tien
(car, bus, truck). Khong can huan luyen lai vi Stanford Cars khong co
bounding box theo tung dong xe.

Dau ra cua YOLOv8 ONNX co dang [1, 84, 8400]:
  - 84 = 4 toa do (cx, cy, w, h) + 80 diem so lop COCO
  - 8400 = so o du doan (voi anh dau vao 640x640)
  - KHONG co diem objectness rieng: diem so lop dong luon vai tro do
  - NMS KHONG duoc nhung san, phai tu xu ly (xem `_non_max_suppression`)
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.utils import get_logger

logger = get_logger(__name__)

# Chi so lop phuong tien trong bo COCO 80 lop.
COCO_VEHICLE_CLASSES = {
    2: "car",
    5: "bus",
    7: "truck",
}

# Kich thuoc dau vao chuan cua YOLOv8.
YOLO_INPUT_SIZE = 640

DEFAULT_CONF_THRESHOLD = 0.35
DEFAULT_IOU_THRESHOLD = 0.45
DEFAULT_MAX_DETECTIONS = 10


@dataclass(frozen=True)
class Detection:
    """Mot vung xe duoc phat hien trong anh.

    Toa do (x1, y1, x2, y2) tinh theo he toa do cua anh GOC, khong phai
    anh da resize.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_name: str

    @property
    def area(self) -> int:
        return max(self.x2 - self.x1, 0) * max(self.y2 - self.y1, 0)

    def crop(self, image: np.ndarray, padding: float = 0.0) -> np.ndarray:
        """Cat vung xe ra khoi anh goc.

        `padding` noi rong hop bao theo ty le moi chieu, giup giu lai mot
        phan boi canh cho tang phan loai.
        """
        height, width = image.shape[:2]
        pad_x = int((self.x2 - self.x1) * padding)
        pad_y = int((self.y2 - self.y1) * padding)

        x1 = max(self.x1 - pad_x, 0)
        y1 = max(self.y1 - pad_y, 0)
        x2 = min(self.x2 + pad_x, width)
        y2 = min(self.y2 + pad_y, height)
        return image[y1:y2, x1:x2]


def _letterbox(
    image: np.ndarray, size: int = YOLO_INPUT_SIZE
) -> tuple[np.ndarray, float, int, int]:
    """Resize anh ve khung vuong, giu nguyen ty le, phan thua to mau xam.

    Tra ve (anh_da_xu_ly, ty_le, le_trai, le_tren) de sau nay quy doi toa
    do ve he cua anh goc.
    """
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    new_h, new_w = int(round(height * scale)), int(round(width * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # To nen xam 114 - gia tri ma YOLO dung khi huan luyen.
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, scale, pad_left, pad_top


def _non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> list[int]:
    """Loai cac hop bao trung nhau, giu hop co diem cao nhat.

    YOLOv8 ONNX khong nhung san NMS nen phai tu lam. Tra ve danh sach chi
    so cua cac hop duoc giu lai.
    """
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        if order.size == 1:
            break

        # Tinh dien tich giao nhau giua hop tot nhat va cac hop con lai.
        rest = order[1:]
        inter_x1 = np.maximum(x1[best], x1[rest])
        inter_y1 = np.maximum(y1[best], y1[rest])
        inter_x2 = np.minimum(x2[best], x2[rest])
        inter_y2 = np.minimum(y2[best], y2[rest])

        inter_w = np.maximum(inter_x2 - inter_x1, 0)
        inter_h = np.maximum(inter_y2 - inter_y1, 0)
        intersection = inter_w * inter_h

        union = areas[best] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-9)

        order = rest[iou < iou_threshold]

    return keep


class VehicleDetector:
    """Phat hien xe trong anh bang YOLOv8 ONNX."""

    def __init__(
        self,
        model_path: Path,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_detections: int = DEFAULT_MAX_DETECTIONS,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay mo hinh YOLOv8: {model_path}. "
                "Xem docs/inference_pipeline.md de biet cach tao file nay."
            )

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("Da nap mo hinh phat hien: %s", model_path.name)

    def _preprocess(
        self, image: np.ndarray
    ) -> tuple[np.ndarray, float, int, int]:
        """Chuyen anh BGR sang tensor dau vao cua YOLOv8."""
        padded, scale, pad_left, pad_top = _letterbox(image)

        # BGR -> RGB, HWC -> CHW, chuan hoa ve [0, 1].
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor[np.newaxis, ...], scale, pad_left, pad_top

    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad_left: int,
        pad_top: int,
        image_shape: tuple[int, int],
    ) -> list[Detection]:
        """Giai ma dau ra [1, 84, 8400] thanh danh sach Detection."""
        # [1, 84, 8400] -> [8400, 84]: moi dong la mot du doan.
        predictions = output[0].T

        # 4 cot dau la toa do, 80 cot sau la diem so cua tung lop COCO.
        class_scores = predictions[:, 4:]

        # Chi quan tam cac lop phuong tien.
        vehicle_ids = np.array(sorted(COCO_VEHICLE_CLASSES))
        vehicle_scores = class_scores[:, vehicle_ids]

        best_local = vehicle_scores.argmax(axis=1)
        confidences = vehicle_scores[np.arange(len(predictions)), best_local]

        keep_mask = confidences >= self.conf_threshold
        if not keep_mask.any():
            return []

        boxes_cxcywh = predictions[keep_mask, :4]
        confidences = confidences[keep_mask]
        class_ids = vehicle_ids[best_local[keep_mask]]

        # (cx, cy, w, h) -> (x1, y1, x2, y2), van o he toa do anh 640x640.
        half_w = boxes_cxcywh[:, 2] / 2
        half_h = boxes_cxcywh[:, 3] / 2
        boxes = np.stack([
            boxes_cxcywh[:, 0] - half_w,
            boxes_cxcywh[:, 1] - half_h,
            boxes_cxcywh[:, 0] + half_w,
            boxes_cxcywh[:, 1] + half_h,
        ], axis=1)

        keep = _non_max_suppression(boxes, confidences, self.iou_threshold)
        keep = keep[:self.max_detections]

        # Quy doi toa do ve he cua anh goc: bo phan le roi chia ty le.
        height, width = image_shape
        detections = []
        for idx in keep:
            x1, y1, x2, y2 = boxes[idx]
            detections.append(Detection(
                x1=int(np.clip((x1 - pad_left) / scale, 0, width)),
                y1=int(np.clip((y1 - pad_top) / scale, 0, height)),
                x2=int(np.clip((x2 - pad_left) / scale, 0, width)),
                y2=int(np.clip((y2 - pad_top) / scale, 0, height)),
                confidence=float(confidences[idx]),
                class_name=COCO_VEHICLE_CLASSES[int(class_ids[idx])],
            ))
        return detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Tim tat ca xe trong anh (dinh dang BGR cua OpenCV).

        Ket qua sap xep theo dien tich giam dan: xe lon nhat (thuong la xe
        chu the cua buc anh) dung dau.
        """
        if image is None or image.size == 0:
            raise ValueError("Anh dau vao rong.")

        tensor, scale, pad_left, pad_top = self._preprocess(image)
        output = self.session.run(None, {self.input_name: tensor})[0]

        detections = self._postprocess(
            output, scale, pad_left, pad_top, image.shape[:2]
        )
        detections.sort(key=lambda d: d.area, reverse=True)

        logger.debug("Phat hien %d xe", len(detections))
        return detections
