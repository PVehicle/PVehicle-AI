"""Quan ly vong doi mo hinh va cung cap chung cho cac router.

Diem quan trong nhat: **ONNX Runtime khong an toan da luong** tren cung
mot `InferenceSession`. Goi song song tu nhieu request se cho ket qua sai
hoac lam sap tien trinh.

Cach xu ly: tao san mot nhom pipeline (moi cai giu session rieng), va cho
moi request muon MOT cai qua hang doi. Request den khi het pipeline se cho
den luot thay vi dung chung.
"""

import asyncio
import queue
from contextlib import asynccontextmanager
from dataclasses import dataclass

from src.api.config import Settings, get_settings
from src.cv.pipeline import RecognitionPipeline
from src.rec.recommender import CarRecommender
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ModelRegistry:
    """Giu cac mo hinh da nap, dung chung cho ca vong doi ung dung."""

    pipelines: queue.Queue[RecognitionPipeline]
    recommender: CarRecommender
    class_count: int
    load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.load_error is None and not self.pipelines.empty()


# Bien toan cuc cho tien trinh, dat trong lifespan luc khoi dong.
_registry: ModelRegistry | None = None


def _build_registry(settings: Settings) -> ModelRegistry:
    """Nap mo hinh mot lan luc khoi dong.

    Recommender luon nap duoc (chi can car_specs.csv). Neu thieu file
    mo hinh ONNX thi API van chay: cac endpoint tu van hoat dong binh
    thuong, rieng endpoint nhan dien tra ve 503.
    """
    recommender = CarRecommender()
    pipelines: queue.Queue[RecognitionPipeline] = queue.Queue()

    try:
        for _ in range(settings.inference_workers):
            pipelines.put(RecognitionPipeline())
        class_count = len(pipelines.queue[0].classifier.class_names)
        logger.info(
            "Da nap %d pipeline nhan dien (%d lop)",
            settings.inference_workers, class_count,
        )
        load_error = None
    except FileNotFoundError as exc:
        class_count = 0
        load_error = str(exc)
        logger.warning(
            "Chua nap duoc mo hinh nhan dien: %s. "
            "Cac endpoint tu van van hoat dong binh thuong.", exc,
        )

    return ModelRegistry(
        pipelines=pipelines,
        recommender=recommender,
        class_count=class_count,
        load_error=load_error,
    )


@asynccontextmanager
async def lifespan(app):
    """Nap mo hinh khi khoi dong, don dep khi tat.

    Nap o day thay vi trong tung request: mo hinh nang ~30 MB va mat vai
    giay de khoi tao session.
    """
    global _registry

    settings = get_settings()
    logger.info("Dang khoi dong %s v%s",
                settings.app_name, settings.app_version)

    # Nap mo hinh trong thread rieng de khong chan vong lap su kien.
    _registry = await asyncio.to_thread(_build_registry, settings)

    if not settings.auth_enabled:
        logger.warning(
            "API dang chay KHONG co xac thuc. Dat bien PVEHICLE_API_KEYS "
            "truoc khi trien khai that."
        )

    yield

    logger.info("Dang tat dich vu")
    _registry = None


def get_registry() -> ModelRegistry:
    """Lay kho mo hinh. Dung lam dependency cua FastAPI."""
    if _registry is None:
        raise RuntimeError(
            "Mo hinh chua duoc nap. Ung dung phai chay qua lifespan."
        )
    return _registry


@asynccontextmanager
async def borrow_pipeline(registry: ModelRegistry):
    """Muon mot pipeline tu nhom, tra lai sau khi dung xong.

    Dung `asyncio.to_thread` de viec cho hang doi khong chan vong lap su
    kien — request khac van duoc phuc vu trong luc cho.
    """
    pipeline = await asyncio.to_thread(registry.pipelines.get)
    try:
        yield pipeline
    finally:
        registry.pipelines.put(pipeline)
