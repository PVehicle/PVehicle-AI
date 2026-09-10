"""Dong goi tap du lieu xe VN thanh file nen de tai len Colab.

Anh goc tu Wikimedia co the rat lon (vai nghin diem anh moi chieu), trong
khi mo hinh chi dung 224x224. Script thu nho anh truoc khi nen, giup file
nhe di nhieu lan ma khong anh huong ket qua huan luyen.

Cach chay:
    python scripts/pack_vn_dataset.py
    python scripts/pack_vn_dataset.py --max-size 512 --quality 88
"""

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from src.utils import (  # noqa: E402
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

SOURCE_DIR = PROCESSED_DATA_DIR / "vn_cars"
OUTPUT_ZIP = PROJECT_ROOT / "vn_cars.zip"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# Mo hinh dung 224x224. Giu 640 de con du bien cho phep cat ngau nhien
# (RandomResizedCrop) khi tang cuong du lieu.
DEFAULT_MAX_SIZE = 640
DEFAULT_QUALITY = 90


def resize_image(source: Path, target: Path, max_size: int,
                 quality: int) -> tuple[int, int]:
    """Thu nho anh neu canh dai hon max_size. Tra ve (byte cu, byte moi)."""
    original_bytes = source.stat().st_size

    with Image.open(source) as image:
        image = image.convert("RGB")
        width, height = image.size

        if max(width, height) > max_size:
            scale = max_size / max(width, height)
            new_size = (round(width * scale), round(height * scale))
            image = image.resize(new_size, Image.LANCZOS)

        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "JPEG", quality=quality, optimize=True)

    return original_bytes, target.stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dong goi tap du lieu xe VN de tai len Colab."
    )
    parser.add_argument(
        "--max-size", type=int, default=DEFAULT_MAX_SIZE,
        help=f"Canh dai toi da cua anh (mac dinh: {DEFAULT_MAX_SIZE})",
    )
    parser.add_argument(
        "--quality", type=int, default=DEFAULT_QUALITY,
        help=f"Chat luong JPEG 1-100 (mac dinh: {DEFAULT_QUALITY})",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_ZIP,
        help=f"File nen dau ra (mac dinh: {OUTPUT_ZIP.name})",
    )
    args = parser.parse_args()

    setup_logging()

    if not SOURCE_DIR.exists():
        logger.error(
            "Chua co du lieu trong %s. Chay truoc:\n"
            "  python scripts/prepare_vn_dataset.py", SOURCE_DIR,
        )
        return 1

    images = [
        path for path in SOURCE_DIR.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        logger.error("Khong tim thay anh nao trong %s", SOURCE_DIR)
        return 1

    logger.info("Dang thu nho %d anh (canh dai toi da %dpx)...",
                len(images), args.max_size)

    total_before = total_after = 0
    failed = 0

    # Dung thu muc tam de khong dong vao du lieu goc.
    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)

        for path in images:
            relative = path.relative_to(SOURCE_DIR)
            # Luon luu dang .jpg cho dong nhat.
            target = staging / relative.with_suffix(".jpg")
            try:
                before, after = resize_image(
                    path, target, args.max_size, args.quality
                )
                total_before += before
                total_after += after
            except OSError as exc:
                failed += 1
                logger.warning("Bo qua %s: %s", relative, exc)

        logger.info("Dang nen...")
        if args.output.exists():
            args.output.unlink()

        with zipfile.ZipFile(
            args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging))

    zip_bytes = args.output.stat().st_size

    print()
    print("=" * 56)
    print("DONG GOI XONG")
    print("=" * 56)
    print(f"  So anh          : {len(images) - failed}")
    if failed:
        print(f"  Bo qua (loi)    : {failed}")
    print(f"  Kich thuoc goc  : {total_before / 1e6:>8.1f} MB")
    print(f"  Sau khi thu nho : {total_after / 1e6:>8.1f} MB")
    print(f"  File nen        : {zip_bytes / 1e6:>8.1f} MB")
    if total_before:
        print(f"  Giam            : {(1 - zip_bytes / total_before):>8.1%}")
    print()
    print(f"  File: {args.output}")
    print()
    print("  Tai file nay VA data/vn_dataset_config.json len Colab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
