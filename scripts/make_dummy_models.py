"""Sinh mo hinh ONNX gia de kiem thu pipeline khi chua co mo hinh that.

Muc dich: cho phep viet va chay thu toan bo luong suy luan + giao dien
trong luc mo hinh that con dang huan luyen tren Colab.

Mo hinh gia co DUNG cau truc dau vao/dau ra nhu mo hinh that, nhung trong
so la ngau nhien nen ket qua du doan vo nghia. Chung chi dung de kiem tra
code chay thong, KHONG dung de danh gia do chinh xac.

Cach chay:
    python scripts/make_dummy_models.py

Khi da co mo hinh that, chi can ghi de len cac file nay trong models/.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    DATA_DIR,
    MODELS_DIR,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

# Phai khop voi mo hinh that (xem src/cv/detector.py va classifier.py).
YOLO_INPUT_SIZE = 640
YOLO_NUM_OUTPUTS = 84  # 4 toa do + 80 lop COCO
YOLO_NUM_BOXES = 8400

CLASSIFIER_INPUT_SIZE = 224
NUM_CLASSES = 196

OPSET_VERSION = 17


def build_dummy_detector(output_path: Path) -> None:
    """Tao mo hinh ONNX gia bat chuoc YOLOv8: [1,3,640,640] -> [1,84,8400].

    Dung phep MatMul de dau ra thuc su phu thuoc dau vao — neu chi tra ve
    hang so, ONNX Runtime co the toi uu bo qua ca dau vao.
    """
    input_tensor = helper.make_tensor_value_info(
        "images", TensorProto.FLOAT,
        [1, 3, YOLO_INPUT_SIZE, YOLO_INPUT_SIZE],
    )
    output_tensor = helper.make_tensor_value_info(
        "output0", TensorProto.FLOAT, [1, YOLO_NUM_OUTPUTS, YOLO_NUM_BOXES]
    )

    rng = np.random.default_rng(seed=42)

    # Gom toan bo anh thanh 1 so, roi banh ra thanh kich thuoc dau ra.
    flatten_shape = helper.make_tensor(
        "flatten_shape", TensorProto.INT64, [2],
        [1, 3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE],
    )
    # Ma tran chieu: [3*640*640, 1] -> gom anh thanh mot gia tri duy nhat.
    projection = helper.make_tensor(
        "projection", TensorProto.FLOAT,
        [3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE, 1],
        (rng.standard_normal(3 * YOLO_INPUT_SIZE * YOLO_INPUT_SIZE)
         * 1e-6).astype(np.float32).ravel(),
    )
    # Mau dau ra: toa do hop bao hop le + diem so lop thap.
    base = np.zeros((1, YOLO_NUM_OUTPUTS, YOLO_NUM_BOXES), dtype=np.float32)
    base[0, 0, :] = 320.0  # cx
    base[0, 1, :] = 320.0  # cy
    base[0, 2, :] = 200.0  # w
    base[0, 3, :] = 150.0  # h
    # Lop 2 (car) cua COCO nam o hang 4+2 = 6.
    base[0, 6, 0] = 0.9  # mot hop co diem cao -> pipeline tim thay 1 xe
    base_tensor = helper.make_tensor(
        "base", TensorProto.FLOAT,
        [1, YOLO_NUM_OUTPUTS, YOLO_NUM_BOXES], base.ravel(),
    )

    nodes = [
        helper.make_node("Reshape", ["images", "flatten_shape"], ["flat"]),
        helper.make_node("MatMul", ["flat", "projection"], ["scalar"]),
        # Nhan voi 0 roi cong: giu phu thuoc dau vao ma khong lam hong mau.
        helper.make_node("Mul", ["scalar", "zero"], ["zeroed"]),
        helper.make_node("Add", ["base", "zeroed"], ["output0"]),
    ]
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])

    graph = helper.make_graph(
        nodes, "dummy_yolov8", [input_tensor], [output_tensor],
        initializer=[flatten_shape, projection, base_tensor, zero],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    logger.info("Da tao mo hinh phat hien gia: %s", output_path.name)


def build_dummy_classifier(output_path: Path) -> None:
    """Tao mo hinh phan loai gia: [1,3,224,224] -> [1,196] logits."""
    input_tensor = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT,
        [1, 3, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE],
    )
    output_tensor = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, NUM_CLASSES]
    )

    rng = np.random.default_rng(seed=7)
    in_features = 3 * CLASSIFIER_INPUT_SIZE * CLASSIFIER_INPUT_SIZE

    flatten_shape = helper.make_tensor(
        "flatten_shape", TensorProto.INT64, [2], [1, in_features]
    )
    weights = helper.make_tensor(
        "weights", TensorProto.FLOAT, [in_features, NUM_CLASSES],
        (rng.standard_normal((in_features, NUM_CLASSES))
         * 0.01).astype(np.float32).ravel(),
    )

    nodes = [
        helper.make_node("Reshape", ["input", "flatten_shape"], ["flat"]),
        helper.make_node("MatMul", ["flat", "weights"], ["logits"]),
    ]

    graph = helper.make_graph(
        nodes, "dummy_classifier", [input_tensor], [output_tensor],
        initializer=[flatten_shape, weights],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(output_path))
    logger.info("Da tao mo hinh phan loai gia: %s", output_path.name)


def build_class_names(output_path: Path) -> None:
    """Tao file nhan tu danh sach lop that cua Stanford Cars."""
    classes_file = DATA_DIR / "stanford_cars_classes.txt"
    names = [
        line.strip()
        for line in classes_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(names) != NUM_CLASSES:
        raise ValueError(
            f"File lop co {len(names)} dong, ky vong {NUM_CLASSES}."
        )

    output_path.write_text(
        json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Da tao file nhan: %s (%d lop)", output_path.name, len(names))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sinh mo hinh ONNX gia de kiem thu pipeline."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=MODELS_DIR,
        help="Thu muc luu mo hinh (mac dinh: models/)",
    )
    args = parser.parse_args()

    setup_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    build_dummy_detector(args.output_dir / "yolov8n.onnx")
    build_dummy_classifier(args.output_dir / "car_classifier.onnx")
    build_class_names(args.output_dir / "class_names.json")

    logger.warning(
        "Day la mo hinh GIA, trong so ngau nhien. Ket qua du doan vo nghia."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
