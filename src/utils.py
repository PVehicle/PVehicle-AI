"""Cac ham tien ich dung chung cho toan bo du an PVehicle-AI."""

import logging
from pathlib import Path

# Thu muc goc du an: src/utils.py -> src/ -> PVehicle-AI/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DOCS_DIR = PROJECT_ROOT / "docs"

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO) -> None:
    """Cau hinh logging cho toan bo ung dung.

    Goi mot lan duy nhat o entry point (app.py hoac script), khong goi
    lai trong tung module de tranh ghi log trung lap.
    """
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """Tra ve logger theo ten module. Dung `get_logger(__name__)`."""
    return logging.getLogger(name)


def ensure_dir(path: Path) -> Path:
    """Tao thu muc (ke ca cac cap cha) neu chua ton tai."""
    path.mkdir(parents=True, exist_ok=True)
    return path
