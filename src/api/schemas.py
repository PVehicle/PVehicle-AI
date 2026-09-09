"""Cac lop du lieu cho request va response cua API.

Pydantic dung cac lop nay de kiem tra dau vao va sinh tai lieu OpenAPI.
"""

from pydantic import BaseModel, Field, model_validator

from src.rec.recommender import Recommendation


class BoundingBox(BaseModel):
    """Hop bao quanh mot chiec xe trong anh, don vi pixel."""

    x1: int = Field(description="Toa do trai")
    y1: int = Field(description="Toa do tren")
    x2: int = Field(description="Toa do phai")
    y2: int = Field(description="Toa do duoi")
    confidence: float = Field(description="Do tin cay cua tang phat hien")
    coco_class: str = Field(description="Lop COCO: car, bus hoac truck")


class Prediction(BaseModel):
    """Mot kha nang ve dong xe, kem xac suat."""

    model_index: int = Field(
        description=(
            "Chi so dau ra cua mo hinh (0-195). Day KHONG phai `class_id` "
            "trong bang thong so — hai he danh so khac nhau. Hay dung "
            "`class_name` de doi chieu giua hai noi."
        )
    )
    class_name: str
    confidence: float


class CarSpecs(BaseModel):
    """Thong so ky thuat cua mot dong xe."""

    class_id: int = Field(
        description=(
            "So thu tu goc trong danh muc Stanford Cars (1-196). Khac voi "
            "`model_index` trong ket qua du doan."
        )
    )
    class_name: str
    brand: str
    model: str
    body_style: str
    year: int
    seats: int
    segment: str
    price_million_vnd: float = Field(
        description="Gia ban (trieu VND) — SO LIEU MO PHONG"
    )
    fuel_l_per_100km: float = Field(
        description="Muc tieu hao (L/100km) — SO LIEU MO PHONG"
    )


class RecommendedCar(CarSpecs):
    """Xe duoc goi y, kem diem phu hop."""

    score: float = Field(description="Diem phu hop, cang cao cang hop")

    @classmethod
    def from_recommendation(
        cls, item: Recommendation, model_name: str
    ) -> "RecommendedCar":
        return cls(
            class_id=item.class_id,
            class_name=item.class_name,
            brand=item.brand,
            model=model_name,
            body_style=item.body_style,
            year=item.year,
            seats=item.seats,
            segment=item.segment,
            price_million_vnd=item.price_million_vnd,
            fuel_l_per_100km=item.fuel_l_per_100km,
            score=item.score,
        )


class DetectedVehicle(BaseModel):
    """Mot chiec xe da nhan dien, kem thong so va goi y."""

    box: BoundingBox | None = Field(
        default=None,
        description=(
            "Hop bao. Bang null khi khong phat hien duoc xe nao va he "
            "thong phan loai toan bo buc anh."
        ),
    )
    predictions: list[Prediction]
    is_confident: bool = Field(
        description="Ket qua co vuot nguong dang tin khong"
    )
    specs: CarSpecs | None = Field(
        default=None, description="Thong so cua dong xe co kha nang nhat"
    )
    similar_cars: list[RecommendedCar] = Field(
        default_factory=list,
        description="Cac xe tuong tu (chi tra khi yeu cau)",
    )


class RecognitionResponse(BaseModel):
    """Ket qua nhan dien cho mot buc anh."""

    vehicle_count: int
    likely_not_a_car: bool = Field(
        description=(
            "True khi nhieu kha nang anh khong chua o to: vua khong phat "
            "hien duoc xe, vua phan loai khong chac chan."
        )
    )
    vehicles: list[DetectedVehicle]
    processing_ms: float = Field(description="Thoi gian xu ly, mili giay")


class RecommendRequest(BaseModel):
    """Nhu cau cua nguoi dung de goi y xe.

    Cac dieu kien la RANG BUOC CUNG: xe khong thoa man bi loai han khoi
    ket qua, khong phai chi bi tru diem.
    """

    max_price: float | None = Field(
        default=None, gt=0, description="Ngan sach toi da (trieu VND)"
    )
    min_price: float | None = Field(
        default=None, ge=0, description="Gia toi thieu (trieu VND)"
    )
    min_seats: int | None = Field(
        default=None, ge=1, le=20, description="So cho ngoi toi thieu"
    )
    body_styles: list[str] | None = Field(
        default=None, description="Cac kieu dang chap nhan duoc"
    )
    max_fuel: float | None = Field(
        default=None, gt=0, description="Muc tieu hao toi da (L/100km)"
    )
    top_n: int = Field(default=5, ge=1, le=50)

    @model_validator(mode="after")
    def check_price_range(self) -> "RecommendRequest":
        if (
            self.max_price is not None
            and self.min_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError(
                "min_price khong duoc lon hon max_price"
            )
        return self


class RecommendResponse(BaseModel):
    """Danh sach xe phu hop voi nhu cau."""

    count: int
    cars: list[RecommendedCar]


class CarListResponse(BaseModel):
    """Mot trang trong danh muc xe."""

    total: int = Field(description="Tong so xe khop dieu kien loc")
    limit: int
    offset: int
    cars: list[CarSpecs]


class FilterOptions(BaseModel):
    """Cac gia tri hop le de dung khi loc — cho giao dien dung."""

    body_styles: list[str]
    segments: list[str]
    price_min: float
    price_max: float
    seats_min: int
    seats_max: int


class HealthResponse(BaseModel):
    """Trang thai song cua dich vu."""

    status: str
    version: str


class ReadinessResponse(BaseModel):
    """Trang thai san sang phuc vu (da nap du mo hinh chua)."""

    status: str
    models_loaded: bool
    car_count: int
    class_count: int
    detail: str | None = None


class ErrorResponse(BaseModel):
    """Khuon dang thong nhat cho moi loi tra ve."""

    detail: str
    request_id: str | None = None
