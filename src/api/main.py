"""Diem khoi chay cua REST API.

Chay o che do phat trien:
    .\\.venv\\Scripts\\python.exe -m uvicorn src.api.main:app --reload

Tai lieu tuong tac: http://localhost:8000/docs
"""

import uuid

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from src.api.config import get_settings
from src.api.dependencies import lifespan
from src.api.rate_limit import limiter
from src.api.routers import cars, health, recognition
from src.utils import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)

settings = get_settings()

# Han muc tinh theo phut nen cho 60 giay la chac chan het cua so.
RETRY_AFTER_SECONDS = 60

DESCRIPTION = """
REST API cho he thong nhan dien va tu van o to.

## Chuc nang

* **Nhan dien xe** — tai anh len, nhan ve dong xe kem thong so ky thuat
* **Tra cuu danh muc** — 196 dong xe voi day du thong so
* **Goi y theo nhu cau** — loc theo ngan sach, so cho, kieu dang
* **Tim xe tuong tu** — dua tren dac diem cua mot xe cho truoc

## Kien truc nhan dien

Hai tang: YOLOv8n khoanh vung xe trong anh, sau do EfficientNet-B0 phan
loai chi tiet 196 dong xe. Toan bo chay bang ONNX Runtime tren CPU.

## Xac thuc

Neu may chu duoc cau hinh API key, moi request phai kem header
`X-API-Key`. Cac endpoint `/health` va `/ready` luon mo.

## Luu y ve du lieu

Gia ban va muc tieu hao nhien lieu la **so lieu mo phong**, khong phai gia
thi truong that. Xem `docs/car_specs_generation.md`.
"""

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    # Trinh duyet chi cho JavaScript doc mot so header co ban. Muon
    # frontend doc duoc X-Request-ID (de bao loi kem ma tra cuu) thi
    # phai khai bao o day, neu khong header van ve nhung bi an di.
    expose_headers=["X-Request-ID", "Retry-After"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Gan ma dinh danh cho moi request de tra vet trong log.

    Client co the tu gui `X-Request-ID` de noi log hai phia voi nhau.
    """
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RateLimitExceeded)
async def handle_rate_limit(request: Request, exc: RateLimitExceeded):
    """Tra loi ro rang khi vuot han muc goi.

    Kem header `Retry-After` de client biet cho bao lau — slowapi khong
    tu them header nay.
    """
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": f"Vuot qua gioi han toc do: {exc.detail}.",
            "request_id": getattr(request.state, "request_id", None),
        },
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
):
    """Gom cac loi kiem tra dau vao thanh mot thong bao de doc."""
    problems = []
    for error in exc.errors():
        location = " → ".join(str(part) for part in error["loc"][1:])
        problems.append(f"{location}: {error['msg']}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Du lieu gui len khong hop le. " + "; ".join(problems),
            "request_id": getattr(request.state, "request_id", None),
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception):
    """Bat loi khong luong truoc, khong lo chi tiet noi bo ra ngoai.

    Thong bao chi tiet chi ghi vao log; client nhan ma request_id de bao
    lai cho quan tri vien tra cuu.
    """
    request_id = getattr(request.state, "request_id", None)
    logger.exception("Loi khong xu ly duoc (request_id=%s)", request_id)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Loi noi bo may chu.",
            "request_id": request_id,
        },
    )


app.include_router(health.router)
app.include_router(recognition.router)
app.include_router(cars.router)


@app.get("/", include_in_schema=False)
def root():
    """Chuyen huong nguoi dung toi trang tai lieu."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
    }
