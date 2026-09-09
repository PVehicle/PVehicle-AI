"""Cau hinh API, doc tu bien moi truong hoac file .env.

Khong hardcode bat ky thong tin nhay cam nao trong ma nguon. Xem file
`.env.example` de biet cac bien co the dat.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Cau hinh ung dung, uu tien bien moi truong hon file .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="PVEHICLE_",
        extra="ignore",
    )

    # --- Thong tin chung ---
    app_name: str = "PVehicle-AI API"
    app_version: str = "1.0.0"
    debug: bool = False

    # --- Bao mat ---
    # Danh sach API key hop le, ngan cach bang dau phay.
    # De TRONG thi API mo cong khai — chi nen dung khi phat trien.
    api_keys: str = ""

    # Cac origin duoc phep goi tu trinh duyet (CORS).
    cors_origins: str = "*"

    # --- Gioi han dau vao ---
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=50_000_000, gt=0)

    # --- Gioi han toc do ---
    rate_limit_default: str = "60/minute"
    rate_limit_recognize: str = "20/minute"

    # --- Suy luan ---
    # So luong session ONNX chay song song. ONNX Runtime KHONG an toan da
    # luong tren cung mot session, nen moi worker giu session rieng.
    inference_workers: int = Field(default=2, ge=1, le=8)

    # Nguong loc cua tang phat hien va so du doan tra ve.
    detection_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    default_top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("api_keys", "cors_origins")
    @classmethod
    def strip_value(cls, value: str) -> str:
        return value.strip()

    @property
    def allowed_api_keys(self) -> set[str]:
        """Tap API key hop le. Rong nghia la khong bat xac thuc."""
        return {
            key.strip() for key in self.api_keys.split(",") if key.strip()
        }

    @property
    def auth_enabled(self) -> bool:
        return bool(self.allowed_api_keys)

    @property
    def allowed_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache
def get_settings() -> Settings:
    """Doc cau hinh mot lan roi dung lai (co cache)."""
    return Settings()
