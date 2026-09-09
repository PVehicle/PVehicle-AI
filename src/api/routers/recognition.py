"""Endpoint nhan dien dong xe tu anh tai len."""

import asyncio
import time

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)

from src.api.config import Settings, get_settings
from src.api.dependencies import (
    ModelRegistry,
    borrow_pipeline,
    get_registry,
)
from src.api.rate_limit import limiter
from src.api.schemas import (
    BoundingBox,
    CarSpecs,
    DetectedVehicle,
    Prediction,
    RecognitionResponse,
    RecommendedCar,
)
from src.api.security import verify_api_key
from src.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["recognition"],
    dependencies=[Depends(verify_api_key)],
)

# Cac dinh dang anh chap nhan duoc, nhan biet qua CHU KY BYTE dau file
# chu khong tin content-type do client khai bao.
IMAGE_SIGNATURES = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"BM": "BMP",
    b"GIF87a": "GIF",
    b"GIF89a": "GIF",
}


def detect_image_format(data: bytes) -> str | None:
    """Nhan dien dinh dang anh qua chu ky byte dau file.

    Khong tin `content-type` cua client: gia tri do do client tu dat va
    co the bi lam gia de day file khong phai anh len may chu.
    """
    for signature, name in IMAGE_SIGNATURES.items():
        if data.startswith(signature):
            return name
    # WEBP: "RIFF" o byte 0-3 va "WEBP" o byte 8-11.
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


async def read_upload(upload: UploadFile, settings: Settings) -> bytes:
    """Doc file tai len, chan file qua lon.

    Doc theo tung khoi va dung ngay khi vuot han muc, thay vi doc het roi
    moi kiem tra — tranh viec mot file rat lon lam can bo nho may chu.
    """
    chunks: list[bytes] = []
    total = 0
    chunk_size = 1024 * 1024

    while chunk := await upload.read(chunk_size):
        total += len(chunk)
        if total > settings.max_upload_bytes:
            limit_mb = settings.max_upload_bytes / 1e6
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File vuot qua gioi han {limit_mb:.0f} MB.",
            )
        chunks.append(chunk)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File rong.",
        )
    return b"".join(chunks)


def decode_image(data: bytes, settings: Settings) -> np.ndarray:
    """Giai ma bytes thanh ma tran anh BGR cua OpenCV."""
    image_format = detect_image_format(data)
    if image_format is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Dinh dang khong duoc ho tro. Chi nhan JPEG, PNG, BMP, "
                "GIF hoac WEBP."
            ),
        )

    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Khong doc duoc anh — file co the bi hong.",
        )

    height, width = image.shape[:2]
    if height * width > settings.max_image_pixels:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Anh {width}x{height} qua lon. Toi da "
                f"{settings.max_image_pixels:,} diem anh."
            ),
        )
    return image


def build_vehicle(
    result, registry: ModelRegistry, include_similar: bool, top_n: int
) -> DetectedVehicle:
    """Chuyen ket qua cua pipeline thanh khuon dang tra ve cua API."""
    detection = result.detection
    box = None
    if detection is not None:
        box = BoundingBox(
            x1=detection.x1, y1=detection.y1,
            x2=detection.x2, y2=detection.y2,
            confidence=round(detection.confidence, 4),
            coco_class=detection.class_name,
        )

    specs_row = registry.recommender.get_car(result.best.class_name)
    specs = None
    similar: list[RecommendedCar] = []

    if specs_row is not None:
        specs = CarSpecs(
            class_id=int(specs_row["class_id"]),
            class_name=str(specs_row["class_name"]),
            brand=str(specs_row["brand"]),
            model=str(specs_row["model"]),
            body_style=str(specs_row["body_style"]),
            year=int(specs_row["year"]),
            seats=int(specs_row["seats"]),
            segment=str(specs_row["segment"]),
            price_million_vnd=float(specs_row["price_million_vnd"]),
            fuel_l_per_100km=float(specs_row["fuel_l_per_100km"]),
        )

        if include_similar:
            items = registry.recommender.recommend_similar(
                result.best.class_name, top_n=top_n
            )
            similar = [
                RecommendedCar.from_recommendation(
                    item,
                    str(
                        registry.recommender.get_car(item.class_name)["model"]
                    ),
                )
                for item in items
            ]

    return DetectedVehicle(
        box=box,
        predictions=[
            Prediction(
                # Chi so cua MO HINH, khac voi class_id trong bang thong so.
                model_index=p.class_id,
                class_name=p.class_name,
                confidence=round(p.confidence, 4),
            )
            for p in result.predictions
        ],
        is_confident=result.is_confident,
        specs=specs,
        similar_cars=similar,
    )


@router.post(
    "/recognize",
    response_model=RecognitionResponse,
    summary="Nhan dien dong xe tu anh",
    responses={
        400: {"description": "File rong hoac anh hong"},
        413: {"description": "File hoac anh qua lon"},
        415: {"description": "Dinh dang khong duoc ho tro"},
        503: {"description": "Mo hinh nhan dien chua san sang"},
    },
)
@limiter.limit(lambda: get_settings().rate_limit_recognize)
async def recognize(
    request: Request,  # slowapi doc request tu tham so nay
    file: UploadFile = File(description="Anh chua xe can nhan dien"),
    top_k: int = Query(default=5, ge=1, le=20),
    include_similar: bool = Query(
        default=False, description="Tra kem cac xe tuong tu"
    ),
    similar_count: int = Query(default=3, ge=1, le=10),
    registry: ModelRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> RecognitionResponse:
    """Nhan dien cac dong xe co trong anh.

    Luong xu ly: YOLOv8 khoanh vung xe → crop → phan loai 196 lop → tra
    ve top-k kha nang kem thong so ky thuat.

    Neu khong phat hien duoc xe nao, he thong phan loai toan bo buc anh
    va danh dau `box = null`.
    """
    if not registry.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Mo hinh nhan dien chua san sang. "
                f"Chi tiet: {registry.load_error}"
            ),
        )

    started = time.perf_counter()

    data = await read_upload(file, settings)
    image = decode_image(data, settings)

    # Suy luan la tac vu nang CPU: chay trong thread rieng de khong chan
    # vong lap su kien, giu cho cac request khac van duoc phuc vu.
    async with borrow_pipeline(registry) as pipeline:
        results = await asyncio.to_thread(
            pipeline.recognize, image, top_k
        )

    vehicles = [
        build_vehicle(result, registry, include_similar, similar_count)
        for result in results
    ]
    elapsed_ms = (time.perf_counter() - started) * 1000

    logger.info(
        "Nhan dien xong: %d xe, %.0f ms, anh %dx%d",
        len(vehicles), elapsed_ms, image.shape[1], image.shape[0],
    )

    return RecognitionResponse(
        vehicle_count=len(vehicles),
        likely_not_a_car=bool(
            results and results[0].is_likely_not_a_car
        ),
        vehicles=vehicles,
        processing_ms=round(elapsed_ms, 2),
    )
