"""Xac thuc API key.

Key duoc cau hinh qua bien moi truong `PVEHICLE_API_KEYS` (ngan cach bang
dau phay), khong bao gio viet cung trong ma nguon.
"""

import secrets

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.api.config import Settings, get_settings

API_KEY_HEADER = "X-API-Key"

api_key_scheme = APIKeyHeader(
    name=API_KEY_HEADER,
    auto_error=False,
    description="API key. Bo qua neu may chu khong bat xac thuc.",
)


def verify_api_key(
    api_key: str | None = Security(api_key_scheme),
    settings: Settings = Depends(get_settings),
) -> None:
    """Kiem tra API key. Bo qua neu may chu chua cau hinh key nao.

    Dung `secrets.compare_digest` thay vi `==` de tranh ro ri thong tin
    qua thoi gian so sanh (timing attack).
    """
    if not settings.auth_enabled:
        return

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Thieu header {API_KEY_HEADER}.",
            headers={"WWW-Authenticate": API_KEY_HEADER},
        )

    # So sanh voi tung key hop le, khong thoat som khi tim thay.
    matched = False
    for valid_key in settings.allowed_api_keys:
        if secrets.compare_digest(api_key, valid_key):
            matched = True

    if not matched:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key khong hop le.",
        )
