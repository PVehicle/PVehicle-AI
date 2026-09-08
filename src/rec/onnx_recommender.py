"""Ban ONNX cua module tu van, dung de so sanh voi ban Pandas.

Muc dich: tra loi cau hoi trong de cuong — "co nen dong bo toan bo kien
truc sang ONNX khong?". Ban nay dong goi buoc tim lang gieng gan nhat
(K-Nearest Neighbors) cua Scikit-learn thanh mo hinh ONNX, roi do thoi
gian suy luan so voi ban Pandas thuan.

Ket qua do duoc ghi trong docs/onnx_vs_pandas.md.

Luu y: chi buoc tim lang gieng duoc dong goi. Viec loc theo rang buoc cung
(ngan sach, so cho) van phai lam bang Pandas, vi ONNX khong bieu dien duoc
phep loc dong nhu vay.
"""

from pathlib import Path

import numpy as np

from src.rec.recommender import CarRecommender, Recommendation
from src.utils import MODELS_DIR, get_logger

logger = get_logger(__name__)

ONNX_MODEL_PATH = MODELS_DIR / "recommender_knn.onnx"


class OnnxRecommender:
    """Tim xe tuong tu bang mo hinh KNN da dong goi sang ONNX.

    Dung chung ma tran dac trung voi `CarRecommender` de ket qua hai ban
    so sanh duoc voi nhau.
    """

    def __init__(
        self,
        model_path: Path = ONNX_MODEL_PATH,
        base: CarRecommender | None = None,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay mo hinh tu van ONNX: {model_path}. "
                "Chay `python scripts/export_recommender_onnx.py` de tao."
            )

        # Import tai day de onnxruntime chi duoc nap khi thuc su can.
        import onnxruntime as ort

        self.base = base if base is not None else CarRecommender()
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        logger.info("Da nap mo hinh tu van ONNX: %s", model_path.name)

    def recommend_similar(
        self, class_name: str, top_n: int = 5
    ) -> list[Recommendation]:
        """Tim xe tuong tu, dung ONNX thay cho tinh toan bang NumPy."""
        matches = self.base.specs.index[
            self.base.specs["class_name"] == class_name
        ]
        if len(matches) == 0:
            raise KeyError(
                f"Khong tim thay xe {class_name!r} trong bang thong so."
            )

        target_idx = int(matches[0])
        query = self.base.features[target_idx:target_idx + 1].astype(
            np.float32
        )

        # KNeighborsTransformer tra ve MOT ma tran khoang cach [1, n_xe]:
        # chi cac lang gieng gan nhat co gia tri khac 0, phan con lai la 0.
        distance_row = self.session.run(
            None, {self.input_name: query}
        )[0][0]

        # Loai chinh xe truy van (khoang cach toi no bang 0) va cac o rong.
        neighbor_ids = np.nonzero(distance_row)[0]
        neighbor_ids = neighbor_ids[neighbor_ids != target_idx]

        # Sap xep theo khoang cach tang dan: gan nhat truoc.
        order = np.argsort(distance_row[neighbor_ids])
        nearest = neighbor_ids[order][:top_n]

        return [
            Recommendation.from_row(
                self.base.specs.iloc[int(idx)],
                float(1.0 / (1.0 + distance_row[idx])),
            )
            for idx in nearest
        ]
