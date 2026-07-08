"""Generate compact WebP card thumbnails for Cloud Storage."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("docs/cards"))
    parser.add_argument("--output", type=Path, default=Path(".cloud-card-assets/cards"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--quality", type=int, default=72)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    sources = sorted(args.source.glob("*.jpg"))
    for index, source in enumerate(sources, 1):
        target = args.output / f"{source.stem}.webp"
        if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
            continue
        with Image.open(source) as image:
            image.thumbnail((args.width, args.width * 2), Image.Resampling.LANCZOS)
            image.convert("RGB").save(
                target, "WEBP", quality=args.quality, method=6, optimize=True
            )
        if index % 100 == 0:
            print(f"Generated {index}/{len(sources)} thumbnails")
    print(f"Card thumbnails ready: {args.output} ({len(sources)} files)")


if __name__ == "__main__":
    main()
