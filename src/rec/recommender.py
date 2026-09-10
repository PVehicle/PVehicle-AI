"""Module tu van: goi y dong xe phu hop voi nhu cau nguoi dung.

Ho tro hai cach dung:

1. `recommend_by_needs` - nguoi dung nhap nhu cau qua form (ngan sach, so
   cho, kieu dang...). Loc cung theo cac rang buoc bat buoc, roi cham diem
   cac tieu chi mem.
2. `recommend_similar` - tim xe tuong tu mot xe da nhan dien. Day la cau
   noi giua hai module: nhan dien ra xe X, goi y ngay cac xe cung tam.

Thuat toan la content-based filtering: bieu dien moi xe thanh mot vector
dac trung da chuan hoa, roi do khoang cach. Xem docs/recommendation.md.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import DATA_DIR, MODELS_DIR, get_logger

logger = get_logger(__name__)

CAR_SPECS_FILE = DATA_DIR / "car_specs.csv"

# Thong so xe thi truong Viet Nam, sinh ra cung luc voi mo hinh xe VN.
# Tuy chon: thieu file nay thi chi co 196 dong xe quoc te.
VN_CAR_SPECS_FILE = MODELS_DIR / "vn_car_specs.json"

REQUIRED_COLUMNS = (
    "class_id", "class_name", "brand", "model", "body_style",
    "year", "seats", "segment", "price_million_vnd", "fuel_l_per_100km",
)

# Trong so cho tung dac trung khi tinh do tuong dong.
# Gia va kieu dang anh huong nhieu nhat toi quyet dinh mua xe.
SIMILARITY_WEIGHTS = {
    "price": 3.0,
    "seats": 2.0,
    "fuel": 1.5,
    "year": 1.0,
    "body_style": 2.5,
    "segment": 2.0,
}

DEFAULT_TOP_N = 5


@dataclass(frozen=True)
class Recommendation:
    """Mot xe duoc goi y kem diem phu hop."""

    class_id: int
    class_name: str
    brand: str
    body_style: str
    year: int
    seats: int
    segment: str
    price_million_vnd: float
    fuel_l_per_100km: float
    score: float

    @classmethod
    def from_row(cls, row: pd.Series, score: float) -> "Recommendation":
        return cls(
            class_id=int(row["class_id"]),
            class_name=str(row["class_name"]),
            brand=str(row["brand"]),
            body_style=str(row["body_style"]),
            year=int(row["year"]),
            seats=int(row["seats"]),
            segment=str(row["segment"]),
            price_million_vnd=float(row["price_million_vnd"]),
            fuel_l_per_100km=float(row["fuel_l_per_100km"]),
            score=score,
        )


def _minmax_normalize(values: pd.Series) -> np.ndarray:
    """Dua gia tri ve khoang [0, 1].

    Neu moi gia tri deu bang nhau thi tra ve toan 0 de tranh chia cho 0.
    """
    array = values.to_numpy(dtype=np.float64)
    span = array.max() - array.min()
    if span < 1e-9:
        return np.zeros_like(array)
    return (array - array.min()) / span


class CarRecommender:
    """He thong tu van xe dua tren thong so ky thuat."""

    def __init__(
        self,
        specs_path: Path = CAR_SPECS_FILE,
        vn_specs_path: Path = VN_CAR_SPECS_FILE,
    ) -> None:
        if not specs_path.exists():
            raise FileNotFoundError(
                f"Khong tim thay bang thong so xe: {specs_path}. "
                "Chay `python scripts/generate_car_specs.py` de tao."
            )

        self.specs = pd.read_csv(specs_path, encoding="utf-8-sig")

        missing = set(REQUIRED_COLUMNS) - set(self.specs.columns)
        if missing:
            raise ValueError(
                f"Bang thong so thieu cot: {sorted(missing)}"
            )

        n_international = len(self.specs)
        n_vietnam = self._append_vn_specs(vn_specs_path)

        self._build_feature_matrix()
        if n_vietnam:
            logger.info(
                "Da nap thong so cua %d dong xe (%d quoc te + %d Viet Nam)",
                len(self.specs), n_international, n_vietnam,
            )
        else:
            logger.info(
                "Da nap thong so cua %d dong xe", len(self.specs)
            )

    def _append_vn_specs(self, vn_specs_path: Path) -> int:
        """Gop them thong so xe Viet Nam neu co. Tra ve so dong da them.

        File nay sinh ra cung luc voi mo hinh xe VN. Thieu no thi module
        tu van van chay binh thuong voi 196 dong xe quoc te.
        """
        if not vn_specs_path.exists():
            return 0

        raw = json.loads(vn_specs_path.read_text(encoding="utf-8"))
        rows = list(raw.values())
        if not rows:
            return 0

        vn_frame = pd.DataFrame(rows)

        missing = set(REQUIRED_COLUMNS) - set(vn_frame.columns) - {
            "class_id"
        }
        if missing:
            logger.warning(
                "Bo qua %s: thieu cot %s",
                vn_specs_path.name, sorted(missing),
            )
            return 0

        # Danh so tiep noi bang quoc te de class_id khong bi trung.
        vn_frame["class_id"] = range(
            len(self.specs) + 1, len(self.specs) + 1 + len(vn_frame)
        )
        vn_frame["data_source"] = "vietnam"
        self.specs["data_source"] = self.specs.get(
            "data_source", "derived+simulated"
        )

        self.specs = pd.concat(
            [self.specs, vn_frame], ignore_index=True
        )
        return len(vn_frame)

    def _build_feature_matrix(self) -> None:
        """Dung ma tran dac trung da chuan hoa de do do tuong dong.

        Cac cot so duoc dua ve [0, 1] roi nhan trong so. Cot phan loai
        (kieu dang, phan khuc) duoc ma hoa one-hot.
        """
        numeric = np.column_stack([
            _minmax_normalize(self.specs["price_million_vnd"])
            * SIMILARITY_WEIGHTS["price"],
            _minmax_normalize(self.specs["seats"])
            * SIMILARITY_WEIGHTS["seats"],
            _minmax_normalize(self.specs["fuel_l_per_100km"])
            * SIMILARITY_WEIGHTS["fuel"],
            _minmax_normalize(self.specs["year"])
            * SIMILARITY_WEIGHTS["year"],
        ])

        body_dummies = pd.get_dummies(self.specs["body_style"]).to_numpy(
            dtype=np.float64
        ) * SIMILARITY_WEIGHTS["body_style"]
        segment_dummies = pd.get_dummies(self.specs["segment"]).to_numpy(
            dtype=np.float64
        ) * SIMILARITY_WEIGHTS["segment"]

        self.features = np.hstack([numeric, body_dummies, segment_dummies])

    def recommend_by_needs(
        self,
        max_price: float | None = None,
        min_price: float | None = None,
        min_seats: int | None = None,
        body_styles: list[str] | None = None,
        max_fuel: float | None = None,
        top_n: int = DEFAULT_TOP_N,
    ) -> list[Recommendation]:
        """Goi y xe theo nhu cau nhap tu form.

        Cac dieu kien la RANG BUOC CUNG: xe khong thoa man bi loai han,
        khong phai bi tru diem. Nguoi dung noi "toi da 800 trieu" thi khong
        nen goi y xe 1 ty du no tot den may.
        """
        mask = pd.Series(True, index=self.specs.index)

        if max_price is not None:
            mask &= self.specs["price_million_vnd"] <= max_price
        if min_price is not None:
            mask &= self.specs["price_million_vnd"] >= min_price
        if min_seats is not None:
            mask &= self.specs["seats"] >= min_seats
        if body_styles:
            mask &= self.specs["body_style"].isin(body_styles)
        if max_fuel is not None:
            mask &= self.specs["fuel_l_per_100km"] <= max_fuel

        candidates = self.specs[mask]
        if candidates.empty:
            logger.info("Khong co xe nao thoa man dieu kien loc.")
            return []

        scores = self._score_candidates(candidates, max_price, max_fuel)
        ranked = np.argsort(scores)[::-1][:top_n]

        return [
            Recommendation.from_row(
                candidates.iloc[int(i)], float(scores[int(i)])
            )
            for i in ranked
        ]

    def _score_candidates(
        self,
        candidates: pd.DataFrame,
        max_price: float | None,
        max_fuel: float | None,
    ) -> np.ndarray:
        """Cham diem cac xe da qua vong loc.

        Trong cung tam gia, uu tien xe moi hon, it ton nhien lieu hon va
        tan dung tot ngan sach (gan muc tran hon thi thuong trang bi tot
        hon, nhung khong vuot qua).
        """
        # Xe moi hon duoc diem cao hon.
        year_score = _minmax_normalize(candidates["year"])

        # Ton it nhien lieu hon duoc diem cao hon (dao nguoc).
        fuel_score = 1.0 - _minmax_normalize(
            candidates["fuel_l_per_100km"]
        )

        if max_price is not None and max_price > 0:
            # Gan muc tran ngan sach thi diem cao hon.
            price_score = (
                candidates["price_million_vnd"].to_numpy() / max_price
            )
        else:
            price_score = _minmax_normalize(
                candidates["price_million_vnd"]
            )

        weights = (0.35, 0.35, 0.30)
        return (
            weights[0] * year_score
            + weights[1] * fuel_score
            + weights[2] * price_score
        )

    def recommend_similar(
        self, class_name: str, top_n: int = DEFAULT_TOP_N
    ) -> list[Recommendation]:
        """Tim cac xe tuong tu mot xe da biet.

        Day la cau noi giua module nhan dien va module tu van: nhan dien
        ra xe nao thi goi y ngay cac xe cung tam.
        """
        matches = self.specs.index[self.specs["class_name"] == class_name]
        if len(matches) == 0:
            raise KeyError(
                f"Khong tim thay xe {class_name!r} trong bang thong so."
            )

        target_idx = int(matches[0])
        target = self.features[target_idx]

        # Khoang cach Euclid tren khong gian dac trung da chuan hoa.
        distances = np.linalg.norm(self.features - target, axis=1)
        distances[target_idx] = np.inf  # bo chinh no ra khoi ket qua

        nearest = np.argsort(distances)[:top_n]

        # Doi khoang cach thanh diem tuong dong trong khoang (0, 1].
        return [
            Recommendation.from_row(
                self.specs.iloc[int(i)],
                float(1.0 / (1.0 + distances[int(i)])),
            )
            for i in nearest
        ]

    def get_car(self, class_name: str) -> pd.Series | None:
        """Tra ve thong so cua mot xe theo ten lop, None neu khong co."""
        matches = self.specs[self.specs["class_name"] == class_name]
        return None if matches.empty else matches.iloc[0]

    @property
    def body_styles(self) -> list[str]:
        """Danh sach kieu dang co trong du lieu, da sap xep."""
        return sorted(self.specs["body_style"].unique())

    @property
    def price_range(self) -> tuple[float, float]:
        """Khoang gia (thap nhat, cao nhat) tinh bang trieu VND."""
        prices = self.specs["price_million_vnd"]
        return float(prices.min()), float(prices.max())
