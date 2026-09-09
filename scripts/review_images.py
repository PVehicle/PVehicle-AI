"""Cong cu duyet anh da thu thap — loai bo anh sai nhan.

Nhan gan tu dong theo ten file KHONG du tin cay. Vi du tim "Toyota Vios
XP150" tra ve ca ban sedan lan hatchback, ten file khong phan biet duoc.
Anh sai nhan se day mo hinh hoc nham, va loi do rat kho phat hien sau nay.

Cong cu nay mo mot trang web cuc don gian: hien luoi anh, bam vao anh sai
de danh dau loai, bam Luu. Nhanh hon gan nhan tay rat nhieu.

Cach chay:
    python scripts/review_images.py
    python scripts/review_images.py --port 8600

Anh bi loai duoc chuyen sang thu muc `_rejected/`, KHONG xoa han — co the
khoi phuc neu duyet nham.
"""

import argparse
import json
import shutil
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    RAW_DATA_DIR,
    ensure_dir,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

IMAGES_DIR = RAW_DATA_DIR / "vn_cars"
REJECTED_DIRNAME = "_rejected"

PAGE_TEMPLATE = """<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>Duyet anh — PVehicle-AI</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 16px 20px 90px;
    font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  }}
  header {{ margin-bottom: 16px; }}
  h1 {{ font-size: 20px; margin: 0 0 6px; }}
  .meta {{ opacity: .7; font-size: 13px; }}
  nav {{ margin: 12px 0; display: flex; flex-wrap: wrap; gap: 6px; }}
  nav a {{
    padding: 5px 11px; border: 1px solid #8886; border-radius: 6px;
    text-decoration: none; color: inherit; font-size: 13px;
  }}
  nav a.current {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
  nav a.done {{ border-color: #16a34a; }}
  .grid {{
    display: grid; gap: 10px;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  }}
  .tile {{ position: relative; cursor: pointer; }}
  .tile img {{
    width: 100%; aspect-ratio: 4/3; object-fit: cover;
    border-radius: 8px; border: 3px solid transparent; display: block;
  }}
  .tile.reject img {{ border-color: #dc2626; opacity: .35; }}
  .tile .mark {{
    position: absolute; top: 8px; right: 8px; font-size: 22px;
    opacity: 0; text-shadow: 0 1px 3px #000;
  }}
  .tile.reject .mark {{ opacity: 1; }}
  .tile .name {{
    font-size: 11px; opacity: .6; margin-top: 3px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  footer {{
    position: fixed; bottom: 0; left: 0; right: 0;
    padding: 12px 20px; background: Canvas;
    border-top: 1px solid #8884;
    display: flex; gap: 12px; align-items: center;
  }}
  button {{
    padding: 9px 18px; font-size: 14px; font-weight: 600;
    border-radius: 7px; border: 1px solid #8886; cursor: pointer;
  }}
  .primary {{ background: #2563eb; color: #fff; border-color: #2563eb; }}
  #count {{ opacity: .75; }}
</style>
</head>
<body>
<header>
  <h1>{class_name}</h1>
  <div class="meta">
    {total} anh &middot; Bam vao anh <b>SAI</b> de danh dau loai
  </div>
  <nav>{nav}</nav>
</header>

<div class="grid">{tiles}</div>

<footer>
  <button class="primary" onclick="save()">Luu va sang lop tiep theo</button>
  <button onclick="clearAll()">Bo chon het</button>
  <span id="count">Da chon loai: 0</span>
</footer>

<script>
const CLASS_NAME = {class_json};
const NEXT_URL = {next_json};

document.querySelectorAll('.tile').forEach(tile => {{
  tile.onclick = () => {{
    tile.classList.toggle('reject');
    updateCount();
  }};
}});

function updateCount() {{
  const n = document.querySelectorAll('.tile.reject').length;
  document.getElementById('count').textContent = 'Da chon loai: ' + n;
}}

function clearAll() {{
  document.querySelectorAll('.tile.reject')
    .forEach(t => t.classList.remove('reject'));
  updateCount();
}}

async function save() {{
  const rejected = [...document.querySelectorAll('.tile.reject')]
    .map(t => t.dataset.file);
  await fetch('/reject', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{class_name: CLASS_NAME, files: rejected}}),
  }});
  window.location = NEXT_URL;
}}
</script>
</body>
</html>
"""


def list_classes() -> list[Path]:
    """Cac thu muc lop, bo qua thu muc chua anh da loai."""
    return sorted(
        path for path in IMAGES_DIR.iterdir()
        if path.is_dir() and path.name != REJECTED_DIRNAME
    )


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


class ReviewHandler(BaseHTTPRequestHandler):
    """May chu web toi gian phuc vu viec duyet anh."""

    def log_message(self, *args):
        """Tat log mac dinh cua http.server cho do nhieu."""

    def send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/":
            classes = list_classes()
            if not classes:
                self.send_html(
                    "<h1>Chua co anh trong data/raw/vn_cars</h1>"
                )
                return
            self.send_response(302)
            self.send_header("Location", f"/review?i=0")
            self.end_headers()
            return

        if path == "/review":
            params = parse_qs(parsed.query)
            index = int(params.get("i", ["0"])[0])
            self.render_review(index)
            return

        if path.startswith("/image/"):
            self.serve_image(path.removeprefix("/image/"))
            return

        self.send_error(404)

    def render_review(self, index: int) -> None:
        classes = list_classes()
        if index >= len(classes):
            self.send_html(
                "<body style='font:16px system-ui;padding:40px'>"
                "<h1>Da duyet xong tat ca cac lop</h1>"
                "<p>Buoc tiep theo: <code>python scripts/prepare_vn_dataset.py"
                "</code></p></body>"
            )
            return

        class_dir = classes[index]
        images = list_images(class_dir)

        tiles = "".join(
            f'<div class="tile" data-file="{path.name}">'
            f'<img src="/image/{class_dir.name}/{path.name}" loading="lazy">'
            f'<div class="mark">&#10060;</div>'
            f'<div class="name">{path.name}</div></div>'
            for path in images
        )

        nav = "".join(
            f'<a href="/review?i={i}" '
            f'class="{"current" if i == index else ""}">'
            f'{c.name.split()[0]} {c.name.split()[1]}</a>'
            for i, c in enumerate(classes)
        )

        self.send_html(PAGE_TEMPLATE.format(
            class_name=class_dir.name,
            total=len(images),
            tiles=tiles,
            nav=nav,
            class_json=json.dumps(class_dir.name),
            next_json=json.dumps(f"/review?i={index + 1}"),
        ))

    def serve_image(self, relative: str) -> None:
        image_path = IMAGES_DIR / relative
        try:
            # Chan truy cap ra ngoai thu muc anh.
            image_path.resolve().relative_to(IMAGES_DIR.resolve())
            content = image_path.read_bytes()
        except (ValueError, OSError):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/reject":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length))

        class_name = payload["class_name"]
        files = payload["files"]
        moved = 0

        if files:
            reject_dir = ensure_dir(
                IMAGES_DIR / REJECTED_DIRNAME / class_name
            )
            for name in files:
                source = IMAGES_DIR / class_name / name
                if source.exists():
                    # Chuyen chu khong xoa — co the khoi phuc neu duyet nham.
                    shutil.move(str(source), str(reject_dir / name))
                    moved += 1
            logger.info("%s: loai %d anh", class_name, moved)
        else:
            logger.info("%s: giu nguyen toan bo", class_name)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"moved": moved}).encode())


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Duyet anh da thu thap, loai bo anh sai nhan."
    )
    parser.add_argument("--port", type=int, default=8600)
    args = parser.parse_args()

    setup_logging()

    if not IMAGES_DIR.exists() or not list_classes():
        logger.error(
            "Chua co anh trong %s. Chay truoc:\n"
            "  python scripts/collect_vn_images.py", IMAGES_DIR,
        )
        return 1

    classes = list_classes()
    total = sum(len(list_images(c)) for c in classes)

    print()
    print("=" * 58)
    print(f"  Duyet {total} anh cua {len(classes)} dong xe")
    print("=" * 58)
    print(f"\n  Mo trinh duyet: http://localhost:{args.port}")
    print("\n  Bam vao anh SAI de danh dau loai, roi bam Luu.")
    print("  Anh bi loai chuyen sang _rejected/, khong xoa han.")
    print("\n  Nhan Ctrl+C de dung.\n")

    server = HTTPServer(("127.0.0.1", args.port), ReviewHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDa dung.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
