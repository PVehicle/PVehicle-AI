"""Thu thap anh xe thi truong Viet Nam tu Wikimedia Commons.

Wikimedia Commons chi chua anh co giay phep tu do (CC BY-SA, CC0, Public
Domain), nen dung duoc cho du an hoc tap ma khong vuong ban quyen.

Quy trinh:
  1. Tim anh theo tu khoa cua tung dong xe
  2. Loc bo anh khong phai ngoai that xe (noi that, dong co, logo...)
  3. Tai anh ve, moi lop mot thu muc
  4. Ghi lai nguon va giay phep tung anh de trich dan

Nhan duoc gan TU DONG tu ten file. Buoc duyet lai bang mat van bat buoc:
xem `scripts/review_images.py`.

Cach chay:
    python scripts/collect_vn_images.py
    python scripts/collect_vn_images.py --limit 100 --only "Toyota Vios"
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    DATA_DIR,
    RAW_DATA_DIR,
    ensure_dir,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

CLASSES_FILE = DATA_DIR / "vn_car_classes.json"
OUTPUT_DIR = RAW_DATA_DIR / "vn_cars"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "PVehicleAI/1.0 (educational project; ONNX car recognition)"

# Wikimedia chan neu goi qua nhanh (tra ve HTTP 429).
#
# Delay khoi diem 2.5s; neu van bi chan, script TU DONG tang len cho ca
# phien (toi da MAX_DELAY) thay vi chi cho lau hon o lan thu lai do.
REQUEST_DELAY = 2.5
MAX_DELAY = 12.0
MAX_RETRIES = 5

# Thoi gian cho co ban khi bi chan, nhan len theo so lan thu lai.
RATE_LIMIT_WAIT = 30

# Cho giua cac lan tai anh (nhe hon goi API vi tai anh it ton tai nguyen).
DOWNLOAD_DELAY = 0.4

# Delay hien tai, tu dieu chinh trong qua trinh chay.
_current_delay = REQUEST_DELAY

DEFAULT_LIMIT = 200
THUMB_WIDTH = 800

# Anh nho hon nguong nay thuong la icon hoac anh chat luong kem.
MIN_FILE_BYTES = 15_000


def api_request(params: dict) -> dict:
    """Goi API Wikimedia, thu lai voi thoi gian cho tang dan.

    Xu ly rieng ma 429 (Too Many Requests): Wikimedia chan kha lau khi bi
    goi don dap, cho vai giay la khong du. Phai cho hang chuc giay va ton
    trong header Retry-After neu may chu gui ve.
    """
    global _current_delay

    query = urllib.parse.urlencode({**params, "format": "json"})
    url = f"{COMMONS_API}?{query}"

    for attempt in range(MAX_RETRIES):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=40) as response:
                data = json.load(response)
            if "error" in data:
                raise ValueError(data["error"].get("info", "loi API"))
            return data

        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                raise
            # Bi chan: tang delay chung cho ca phien, khong chi lan nay.
            _current_delay = min(_current_delay * 1.5, MAX_DELAY)
            wait = int(exc.headers.get("Retry-After") or 0)
            wait = max(wait, RATE_LIMIT_WAIT * (attempt + 1))
            logger.warning(
                "  Bi gioi han toc do, cho %ds (delay moi: %.1fs)",
                wait, _current_delay,
            )
            time.sleep(wait)

        except (urllib.error.URLError, ValueError, json.JSONDecodeError):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Van bi chan sau nhieu lan thu lai.")


def polite_sleep() -> None:
    """Cho giua cac lan goi API, theo delay hien tai cua phien."""
    time.sleep(_current_delay)


def search_images(keyword: str, limit: int) -> list[str]:
    """Tim ten file anh khop tu khoa."""
    titles: list[str] = []
    offset = 0

    while len(titles) < limit:
        data = api_request({
            "action": "query", "list": "search", "srnamespace": 6,
            "srlimit": min(50, limit - len(titles)), "sroffset": offset,
            "srsearch": f"{keyword} filetype:bitmap",
        })
        hits = data.get("query", {}).get("search", [])
        if not hits:
            break
        titles.extend(hit["title"] for hit in hits)
        offset += len(hits)
        polite_sleep()

        if "continue" not in data:
            break

    return titles[:limit]


def is_relevant(title: str, keyword: str, exclude: list[str]) -> bool:
    """Kiem tra ten file co phai anh ngoai that cua dung dong xe khong.

    Loc theo TEN FILE thay vi tin ket qua tim kiem toan van: Wikimedia
    khop ca noi dung mo ta, nen tra ve nhieu anh khong lien quan.
    """
    lowered = title.lower()

    if any(word in lowered for word in exclude):
        return False

    # Moi tu trong tu khoa deu phai xuat hien trong ten file.
    # Vi du "Toyota Vios XP150" -> can ca "toyota", "vios", "xp150".
    return all(token in lowered for token in keyword.lower().split())


def fetch_image_info(titles: list[str]) -> list[dict]:
    """Lay URL va thong tin giay phep cua cac anh.

    API cho phep hoi toi da 50 file moi lan.
    """
    results: list[dict] = []

    for index in range(0, len(titles), 50):
        batch = titles[index:index + 50]
        data = api_request({
            "action": "query", "prop": "imageinfo",
            "iiprop": "url|extmetadata", "iiurlwidth": THUMB_WIDTH,
            "titles": "|".join(batch),
        })

        for page in data.get("query", {}).get("pages", {}).values():
            info = page.get("imageinfo")
            if not info:
                continue
            meta = info[0].get("extmetadata", {})
            results.append({
                "title": page["title"],
                "url": info[0].get("thumburl") or info[0]["url"],
                "license": meta.get("LicenseShortName", {}).get(
                    "value", "khong ro"
                ),
                "artist": re.sub(
                    r"<[^>]+>", "",
                    meta.get("Artist", {}).get("value", ""),
                ).strip()[:100],
                "descriptionurl": info[0].get("descriptionurl", ""),
            })
        polite_sleep()

    return results


def safe_filename(title: str) -> str:
    """Chuyen ten file Wikimedia thanh ten hop le tren Windows."""
    name = title.removeprefix("File:")
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    # Windows gioi han ~255 ky tu cho ten file.
    if len(name) > 120:
        stem, _, suffix = name.rpartition(".")
        name = f"{stem[:110]}.{suffix}"
    return name


def download_image(item: dict, out_dir: Path) -> bool:
    """Tai mot anh. Tra False neu that bai hoac anh qua nho.

    Buoc tai anh cung bi gioi han toc do nhu API, nen cung phai xu ly 429.
    """
    out_path = out_dir / safe_filename(item["title"])
    if out_path.exists():
        return True

    for attempt in range(3):
        try:
            request = urllib.request.Request(
                item["url"], headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                content = response.read()
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429:
                logger.debug("Khong tai duoc %s: %s", item["title"], exc)
                return False
            wait = int(exc.headers.get("Retry-After") or 0)
            time.sleep(max(wait, RATE_LIMIT_WAIT))
        except (urllib.error.URLError, TimeoutError) as exc:
            logger.debug("Khong tai duoc %s: %s", item["title"], exc)
            return False
    else:
        return False

    if len(content) < MIN_FILE_BYTES:
        return False

    out_path.write_bytes(content)
    # Cho nhe giua cac lan tai de khong bi chan.
    time.sleep(DOWNLOAD_DELAY)
    return True


def collect_one_class(car: dict, exclude: list[str], limit: int) -> dict:
    """Thu thap anh cho mot dong xe."""
    class_name = car["class_name"]
    out_dir = ensure_dir(OUTPUT_DIR / class_name)

    found: dict[str, dict] = {}

    for keyword in car["search"]:
        # Loi o mot tu khoa khong nen lam mat ket qua cua tu khoa khac.
        try:
            titles = search_images(keyword, limit * 3)
            relevant = [
                t for t in titles if is_relevant(t, keyword, exclude)
            ]
            logger.info(
                "  %-28s tim %3d -> hop le %3d", keyword, len(titles),
                len(relevant),
            )
            for item in fetch_image_info(relevant):
                found.setdefault(item["title"], item)
        except (urllib.error.URLError, ValueError, RuntimeError) as exc:
            logger.warning("  %-28s that bai: %s", keyword, exc)

        if len(found) >= limit:
            break

    items = list(found.values())[:limit]
    downloaded = sum(download_image(item, out_dir) for item in items)

    # Ghi nguon va giay phep de trich dan trong bao cao.
    credits_file = out_dir / "_credits.json"
    credits_file.write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("  => %s: %d anh", class_name, downloaded)
    return {"class_name": class_name, "downloaded": downloaded}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Thu thap anh xe VN tu Wikimedia Commons."
    )
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_LIMIT,
        help=f"So anh toi da moi dong xe (mac dinh: {DEFAULT_LIMIT})",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="Chi thu thap cac dong xe co ten chua chuoi nay",
    )
    args = parser.parse_args()

    setup_logging()

    config = json.loads(CLASSES_FILE.read_text(encoding="utf-8"))
    exclude = config["exclude_keywords"]
    cars = config["cars"]

    if args.only:
        needle = args.only.lower()
        cars = [c for c in cars if needle in c["class_name"].lower()]
        if not cars:
            logger.error("Khong co dong xe nao khop %r", args.only)
            return 1

    logger.info("Thu thap %d dong xe, toi da %d anh moi dong",
                len(cars), args.limit)
    ensure_dir(OUTPUT_DIR)

    summary = []
    for index, car in enumerate(cars, start=1):
        logger.info("[%d/%d] %s", index, len(cars), car["class_name"])
        try:
            summary.append(collect_one_class(car, exclude, args.limit))
        except (urllib.error.URLError, ValueError) as exc:
            logger.error("  Bo qua %s: %s", car["class_name"], exc)
            summary.append({
                "class_name": car["class_name"], "downloaded": 0,
            })

    total = sum(item["downloaded"] for item in summary)
    enough = [s for s in summary if s["downloaded"] >= 80]
    too_few = [s for s in summary if s["downloaded"] < 80]

    print()
    print("=" * 60)
    print(f"TONG: {total} anh cho {len(summary)} dong xe")
    print("=" * 60)
    print(f"\nDU DUNG (>=80 anh): {len(enough)} dong")
    for item in sorted(enough, key=lambda s: -s["downloaded"]):
        print(f"  {item['class_name']:<36} {item['downloaded']:>4}")

    if too_few:
        print(f"\nCHUA DU (<80 anh): {len(too_few)} dong")
        for item in sorted(too_few, key=lambda s: -s["downloaded"]):
            print(f"  {item['class_name']:<36} {item['downloaded']:>4}")
        print("\n  Cac dong nay nen loai khoi tap huan luyen, hoac bo sung")
        print("  tu khoa tim kiem trong data/vn_car_classes.json.")

    print(f"\nAnh luu tai: {OUTPUT_DIR}")
    print("Buoc tiep theo: python scripts/review_images.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
