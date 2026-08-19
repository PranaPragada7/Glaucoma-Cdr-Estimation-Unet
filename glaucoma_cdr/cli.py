"""Command-line interface for one-image segmentation and CDR estimation."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .inference import infer_image
from .model import DEFAULT_WEIGHTS


def _save_mask(mask: np.ndarray, path: Path) -> None:
    image = Image.fromarray(mask.astype(np.uint8) * 255)
    image.save(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Segment optic disc/cup masks and estimate vertical CDR."
    )
    parser.add_argument("--image", required=True, type=Path, help="fundus image")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Keras HDF5 weight file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for masks and JSON summary",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--scale-01",
        action="store_true",
        help="divide RGB values by 255 (not used by the original notebook)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 <= args.threshold < 1:
        raise SystemExit("--threshold must be in the range [0, 1)")

    result = infer_image(
        args.image,
        weights_path=args.weights,
        threshold=args.threshold,
        scale_01=args.scale_01,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.image.stem
    disc_path = args.output_dir / f"{stem}_disc_mask.png"
    cup_path = args.output_dir / f"{stem}_cup_mask.png"
    summary_path = args.output_dir / f"{stem}_cdr.json"
    _save_mask(result.disc_mask, disc_path)
    _save_mask(result.cup_mask, cup_path)

    summary = {
        "source_image": str(args.image),
        "weights": str(args.weights),
        "threshold": args.threshold,
        "disc_vertical_diameter_px": result.cdr.disc_vertical_diameter,
        "cup_vertical_diameter_px": result.cdr.cup_vertical_diameter,
        "vertical_cdr": result.cdr.ratio,
        "disclaimer": "Research use only; not a clinical diagnosis.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Vertical CDR: {result.cdr.ratio:.3f}")
    print(f"Disc mask: {disc_path}")
    print(f"Cup mask: {cup_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
