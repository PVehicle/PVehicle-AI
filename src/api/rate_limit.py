"""Gioi han toc do goi API.

Dem theo API key neu co, khong thi theo dia chi IP. Bo dem nam trong bo
nho cua tien trinh — du cho mot may chu, nhung neu chay nhieu ban sao thi
can chuyen sang Redis.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from src.api.security import API_KEY_HEADER


def rate_limit_key(request: Request) -> str:
    """Khoa dem: uu tien API key, khong co thi dung IP.

    Dem theo key giup nhieu nguoi dung sau cung mot NAT khong lam anh
    huong han muc cua nhau.
    """
    api_key = request.headers.get(API_KEY_HEADER)
    if api_key:
        # Chi lay 16 ky tu dau: du de phan biet, khong luu tron key.
        return f"key:{api_key[:16]}"
    return f"ip:{get_remote_address(request)}"


limiter = Limiter(key_func=rate_limit_key)
