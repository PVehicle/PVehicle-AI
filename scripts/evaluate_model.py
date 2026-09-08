"""Sinh so lieu danh gia mo hinh de dua vao bao cao.

Doc checkpoint da huan luyen tren Colab, xuat ra:
  - Bang so lieu tung epoch (dinh dang Markdown, dan thang vao bao cao)
  - Bieu do loss va do chinh xac
  - Cac chi so tong hop

Cach chay:
    python scripts/evaluate_model.py --checkpoint duong/dan/best.pt

Neu khong truyen --checkpoint, script tim trong models/.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import (  # noqa: E402
    MODELS_DIR,
    PROJECT_ROOT,
    get_logger,
    setup_logging,
)

logger = get_logger(__name__)

DEFAULT_CHECKPOINT = MODELS_DIR / "best.pt"
REPORT_DIR = PROJECT_ROOT / "reports"


def load_history(checkpoint_path: Path) -> tuple[dict, int, float]:
    """Doc lich su huan luyen tu checkpoint.

    Checkpoint duoc luu boi notebook Colab, chua ca lich su tung epoch.
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "Can thu vien torch de doc file .pt. Cai bang:\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install torch\n"
            "Hoac dung --history de doc tu file JSON da xuat san."
        ) from None

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    return (
        checkpoint["history"],
        int(checkpoint["epoch"]),
        float(checkpoint["best_acc"]),
    )


def write_markdown_table(history: dict, output_path: Path) -> None:
    """Xuat bang so lieu tung epoch dang Markdown."""
    lines = [
        "# Kết quả huấn luyện theo epoch",
        "",
        "| Epoch | Train loss | Train acc | Val loss | Top-1 | Top-5 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    n_epochs = len(history["train_loss"])
    for i in range(n_epochs):
        lines.append(
            f"| {i + 1} "
            f"| {history['train_loss'][i]:.4f} "
            f"| {history['train_acc'][i]:.2%} "
            f"| {history['val_loss'][i]:.4f} "
            f"| {history['val_top1'][i]:.2%} "
            f"| {history['val_top5'][i]:.2%} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Da ghi bang so lieu: %s", output_path.name)


def plot_curves(history: dict, output_path: Path) -> None:
    """Ve bieu do loss va do chinh xac."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # khong can man hinh
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning(
            "Chua cai matplotlib, bo qua buoc ve bieu do. "
            "Cai bang: pip install matplotlib"
        )
        return

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))

    ax_loss.plot(epochs, history["train_loss"], label="Train")
    ax_loss.plot(epochs, history["val_loss"], label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Loss theo epoch")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="Train top-1")
    ax_acc.plot(epochs, history["val_top1"], label="Val top-1")
    ax_acc.plot(epochs, history["val_top5"], label="Val top-5")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Do chinh xac")
    ax_acc.set_title("Do chinh xac theo epoch")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Da ve bieu do: %s", output_path.name)


def summarize(history: dict, best_epoch: int, best_acc: float) -> dict:
    """Tinh cac chi so tong hop cho bao cao."""
    top1 = history["val_top1"]
    top5 = history["val_top5"]

    return {
        "so_epoch": len(top1),
        "epoch_tot_nhat": best_epoch,
        "top1_tot_nhat": max(top1),
        "top5_tot_nhat": max(top5),
        "top1_cuoi_cung": top1[-1],
        "top5_cuoi_cung": top5[-1],
        "train_acc_cuoi": history["train_acc"][-1],
        # Chenh lech lon giua train va val la dau hieu qua khop.
        "chenh_lech_train_val": history["train_acc"][-1] - top1[-1],
        "best_acc_trong_checkpoint": best_acc,
    }


def print_summary(summary: dict) -> None:
    """In bang tom tat ra man hinh."""
    print()
    print("=" * 56)
    print("TOM TAT KET QUA HUAN LUYEN")
    print("=" * 56)
    print(f"{'So epoch da chay':<32} {summary['so_epoch']:>10}")
    print(f"{'Epoch tot nhat':<32} {summary['epoch_tot_nhat']:>10}")
    print("-" * 56)
    print(f"{'Top-1 tot nhat':<32} {summary['top1_tot_nhat']:>9.2%}")
    print(f"{'Top-5 tot nhat':<32} {summary['top5_tot_nhat']:>9.2%}")
    print(f"{'Top-1 epoch cuoi':<32} {summary['top1_cuoi_cung']:>9.2%}")
    print(f"{'Top-5 epoch cuoi':<32} {summary['top5_cuoi_cung']:>9.2%}")
    print("-" * 56)
    print(f"{'Train acc epoch cuoi':<32} {summary['train_acc_cuoi']:>9.2%}")

    gap = summary["chenh_lech_train_val"]
    print(f"{'Chenh lech train - val':<32} {gap:>9.2%}")
    if gap > 0.15:
        print()
        print("  Canh bao: chenh lech > 15% cho thay mo hinh QUA KHOP.")
        print("  Co the tang cuong du lieu manh hon hoac giam so epoch.")
    print("=" * 56)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sinh so lieu danh gia tu checkpoint huan luyen."
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
        help=f"Duong dan file .pt (mac dinh: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--history", type=Path, default=None,
        help="Doc lich su tu file JSON thay vi tu checkpoint",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPORT_DIR,
        help=f"Thu muc xuat ket qua (mac dinh: {REPORT_DIR.name}/)",
    )
    args = parser.parse_args()

    setup_logging()

    if args.history is not None:
        if not args.history.exists():
            logger.error("Khong tim thay file: %s", args.history)
            return 1
        data = json.loads(args.history.read_text(encoding="utf-8"))
        history = data["history"]
        best_epoch = data.get("epoch", len(history["val_top1"]))
        best_acc = data.get("best_acc", max(history["val_top1"]))
    else:
        if not args.checkpoint.exists():
            logger.error(
                "Khong tim thay checkpoint: %s\n"
                "Tai file best.pt tu Google Drive "
                "(MyDrive/PVehicle-AI/checkpoints/) vao thu muc models/.",
                args.checkpoint,
            )
            return 1
        history, best_epoch, best_acc = load_history(args.checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_markdown_table(history, args.output_dir / "training_table.md")
    plot_curves(history, args.output_dir / "training_curves.png")

    summary = summarize(history, best_epoch, best_acc)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print_summary(summary)
    logger.info("Ket qua da luu trong: %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
