"""Endpoint kiem tra suc khoe dich vu.

Tach lam hai muc dich khac nhau:
  - /health  : tien trinh con song khong (dung cho liveness probe)
  - /ready   : da nap du mo hinh de phuc vu chua (readiness probe)

Ca hai deu KHONG yeu cau xac thuc — bo can bang tai phai goi duoc.
"""

from fastapi import APIRouter, Depends, Response, status

from src.api.config import Settings, get_settings
from src.api.dependencies import ModelRegistry, get_registry
from src.api.schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Kiem tra tien trinh con song. Luon tra 200 neu may chu con chay."""
    return HealthResponse(status="ok", version=settings.app_version)


@router.get("/ready", response_model=ReadinessResponse)
def ready(
    response: Response,
    registry: ModelRegistry = Depends(get_registry),
) -> ReadinessResponse:
    """Kiem tra da san sang phuc vu chua.

    Tra 503 khi chua nap duoc mo hinh nhan dien, de bo can bang tai khong
    dinh tuyen request vao ban sao nay.
    """
    if not registry.is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if registry.is_ready else "degraded",
        models_loaded=registry.is_ready,
        car_count=len(registry.recommender.specs),
        class_count=registry.class_count,
        detail=registry.load_error,
    )
