"""Endpoint tra cuu danh muc xe va goi y theo nhu cau.

Cac endpoint o day chi can `car_specs.csv`, khong phu thuoc mo hinh ONNX,
nen van hoat dong ngay ca khi chua co file mo hinh.
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    status,
)

from src.api.config import Settings, get_settings
from src.api.dependencies import ModelRegistry, get_registry
from src.api.rate_limit import limiter
from src.api.schemas import (
    CarListResponse,
    CarSpecs,
    FilterOptions,
    RecommendRequest,
    RecommendResponse,
    RecommendedCar,
)
from src.api.security import verify_api_key
from src.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["cars"],
    dependencies=[Depends(verify_api_key)],
)


def row_to_specs(row) -> CarSpecs:
    """Chuyen mot dong DataFrame thanh khuon dang tra ve cua API."""
    return CarSpecs(
        class_id=int(row["class_id"]),
        class_name=str(row["class_name"]),
        brand=str(row["brand"]),
        model=str(row["model"]),
        body_style=str(row["body_style"]),
        year=int(row["year"]),
        seats=int(row["seats"]),
        segment=str(row["segment"]),
        price_million_vnd=float(row["price_million_vnd"]),
        fuel_l_per_100km=float(row["fuel_l_per_100km"]),
    )


@router.get(
    "/cars",
    response_model=CarListResponse,
    summary="Liet ke danh muc xe",
)
@limiter.limit(lambda: get_settings().rate_limit_default)
def list_cars(
    request: Request,
    brand: str | None = Query(default=None, description="Loc theo hang"),
    body_style: str | None = Query(default=None),
    segment: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    registry: ModelRegistry = Depends(get_registry),
) -> CarListResponse:
    """Tra ve danh muc xe, co the loc va phan trang.

    Cac bo loc so sanh khong phan biet hoa thuong.
    """
    specs = registry.recommender.specs

    if brand:
        specs = specs[specs["brand"].str.lower() == brand.lower()]
    if body_style:
        specs = specs[specs["body_style"].str.lower() == body_style.lower()]
    if segment:
        specs = specs[specs["segment"].str.lower() == segment.lower()]

    total = len(specs)
    page = specs.iloc[offset:offset + limit]

    return CarListResponse(
        total=total,
        limit=limit,
        offset=offset,
        cars=[row_to_specs(row) for _, row in page.iterrows()],
    )


@router.get(
    "/cars/filters",
    response_model=FilterOptions,
    summary="Lay cac gia tri hop le de loc",
)
@limiter.limit(lambda: get_settings().rate_limit_default)
def get_filters(
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
) -> FilterOptions:
    """Tra ve cac gia tri co the dung khi loc.

    Giao dien nen goi endpoint nay de dung form thay vi viet cung danh
    sach — bang thong so co the thay doi.
    """
    specs = registry.recommender.specs
    price_min, price_max = registry.recommender.price_range

    return FilterOptions(
        body_styles=registry.recommender.body_styles,
        segments=sorted(specs["segment"].unique().tolist()),
        price_min=float(price_min),
        price_max=float(price_max),
        seats_min=int(specs["seats"].min()),
        seats_max=int(specs["seats"].max()),
    )


@router.get(
    "/cars/{class_name}",
    response_model=CarSpecs,
    summary="Tra cuu thong so mot dong xe",
    responses={404: {"description": "Khong tim thay dong xe"}},
)
@limiter.limit(lambda: get_settings().rate_limit_default)
def get_car(
    request: Request,
    class_name: str,
    registry: ModelRegistry = Depends(get_registry),
) -> CarSpecs:
    """Tra cuu thong so theo ten lop day du.

    Vi du: `Tesla Model S Sedan 2012`
    """
    row = registry.recommender.get_car(class_name)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Khong tim thay dong xe {class_name!r}.",
        )
    return row_to_specs(row)


@router.get(
    "/cars/{class_name}/similar",
    response_model=RecommendResponse,
    summary="Tim cac xe tuong tu",
    responses={404: {"description": "Khong tim thay dong xe"}},
)
@limiter.limit(lambda: get_settings().rate_limit_default)
def similar_cars(
    request: Request,
    class_name: str,
    top_n: int = Query(default=5, ge=1, le=20),
    registry: ModelRegistry = Depends(get_registry),
) -> RecommendResponse:
    """Tim cac xe co dac diem gan giong xe da cho."""
    if registry.recommender.get_car(class_name) is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Khong tim thay dong xe {class_name!r}.",
        )

    items = registry.recommender.recommend_similar(class_name, top_n=top_n)
    cars = [
        RecommendedCar.from_recommendation(
            item,
            str(registry.recommender.get_car(item.class_name)["model"]),
        )
        for item in items
    ]
    return RecommendResponse(count=len(cars), cars=cars)


@router.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Goi y xe theo nhu cau",
)
@limiter.limit(lambda: get_settings().rate_limit_default)
def recommend(
    request: Request,
    payload: RecommendRequest,
    registry: ModelRegistry = Depends(get_registry),
) -> RecommendResponse:
    """Goi y xe phu hop voi nhu cau nguoi dung.

    Cac dieu kien la RANG BUOC CUNG: xe khong thoa man bi loai han khoi
    ket qua. Nguoi dung noi "toi da 800 trieu" thi khong goi y xe 1 ty du
    no tot den may.

    Danh sach rong nghia la khong co xe nao thoa man — hay noi long dieu
    kien.
    """
    items = registry.recommender.recommend_by_needs(
        max_price=payload.max_price,
        min_price=payload.min_price,
        min_seats=payload.min_seats,
        body_styles=payload.body_styles,
        max_fuel=payload.max_fuel,
        top_n=payload.top_n,
    )
    cars = [
        RecommendedCar.from_recommendation(
            item,
            str(registry.recommender.get_car(item.class_name)["model"]),
        )
        for item in items
    ]
    return RecommendResponse(count=len(cars), cars=cars)
