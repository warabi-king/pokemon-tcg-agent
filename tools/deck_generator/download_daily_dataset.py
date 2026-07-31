from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "manifest.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "daily_dataset"
KAGGLE_DOWNLOAD_URL = "https://www.kaggle.com/api/v1/datasets/download/{owner}/{slug}"


@dataclass(frozen=True)
class ManifestRow:
    date: str
    url: str
    total_bytes: int
    episode_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Kaggle daily battle episode datasets and extract JSON files by date."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--date", action="append", help="Download only this date. Can be repeated.")
    parser.add_argument("--limit", type=int, help="Download only the first N matching manifest rows.")
    parser.add_argument("--force", action="store_true", help="Redownload even if output files already exist.")
    parser.add_argument("--keep-zip", action="store_true", help="Keep downloaded zip files after extraction.")
    parser.add_argument("--zip-only", action="store_true", help="Download zip files and do not extract JSON files.")
    parser.add_argument(
        "--ignore-space-check",
        action="store_true",
        help="Start even if free space is below the manifest total_bytes estimate.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned downloads without downloading.")
    return parser.parse_args()


def read_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"date", "daily_dataset_url", "total_bytes", "episode_count"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing columns: {', '.join(sorted(missing))}")

        for raw in reader:
            rows.append(
                ManifestRow(
                    date=raw["date"],
                    url=raw["daily_dataset_url"],
                    total_bytes=int(raw["total_bytes"]),
                    episode_count=int(raw["episode_count"]),
                )
            )
    return rows


def dataset_ref(dataset_url: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(dataset_url)
    parts = [part for part in parsed.path.split("/") if part]
    try:
        datasets_index = parts.index("datasets")
        owner = parts[datasets_index + 1]
        slug = parts[datasets_index + 2]
    except (ValueError, IndexError) as exc:
        raise ValueError(f"could not parse Kaggle dataset URL: {dataset_url}") from exc
    return owner, slug


def already_done(date_dir: Path, expected_count: int) -> bool:
    if not date_dir.exists():
        return False
    json_count = sum(1 for path in date_dir.rglob("*.json") if path.is_file())
    return json_count >= expected_count


def zip_already_done(zip_path: Path, expected_count: int) -> bool:
    if not zip_path.exists():
        return False
    try:
        return count_json_in_zip(zip_path) >= expected_count
    except zipfile.BadZipFile:
        return False


def check_free_space(output_dir: Path, rows: list[ManifestRow], ignore: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(output_dir).free
    required = sum(row.total_bytes for row in rows)
    if free < required and not ignore:
        free_gb = free / 1024**3
        required_gb = required / 1024**3
        raise RuntimeError(
            f"not enough free space for all requested datasets: "
            f"{free_gb:.1f} GiB free, {required_gb:.1f} GiB estimated. "
            "Use --output-dir on a larger drive or pass --ignore-space-check."
        )


def download_zip(row: ManifestRow, zip_path: Path) -> None:
    owner, slug = dataset_ref(row.url)
    url = KAGGLE_DOWNLOAD_URL.format(
        owner=urllib.parse.quote(owner),
        slug=urllib.parse.quote(slug),
    )
    url = f"{url}?datasetVersionNumber=1"

    request = urllib.request.Request(url, headers={"User-Agent": "pokemon-tcg-agent-dataset-downloader"})
    part_path = zip_path.with_suffix(zip_path.suffix + ".part")
    with urllib.request.urlopen(request) as response:
        total = response.headers.get("Content-Length")
        total_text = f" ({int(total) / 1024**3:.1f} GiB)" if total and total.isdigit() else ""
        print(f"[{row.date}] downloading{total_text}: {url}", flush=True)
        with part_path.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024 * 16)
    part_path.replace(zip_path)


def count_json_in_zip(zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as zf:
        return sum(1 for info in zf.infolist() if not info.is_dir() and info.filename.endswith(".json"))


def extract_json(zip_path: Path, date_dir: Path) -> int:
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        json_members = [info for info in zf.infolist() if not info.is_dir() and info.filename.endswith(".json")]
        for info in json_members:
            target = date_dir / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)
            extracted += 1
    return extracted


def download_row(row: ManifestRow, output_dir: Path, force: bool, keep_zip: bool, zip_only: bool) -> None:
    date_dir = output_dir / row.date
    zip_path = date_dir / f"{row.date}.zip"
    if zip_only and not force and zip_already_done(zip_path, row.episode_count):
        print(f"[{row.date}] skipped; zip already contains JSON files", flush=True)
        return
    if not zip_only and not force and already_done(date_dir, row.episode_count):
        print(f"[{row.date}] skipped; JSON files already present", flush=True)
        return

    date_dir.mkdir(parents=True, exist_ok=True)
    download_zip(row, zip_path)
    json_count = count_json_in_zip(zip_path)
    print(f"[{row.date}] zip contains {json_count} JSON files", flush=True)
    if zip_only:
        return

    print(f"[{row.date}] extracting JSON files", flush=True)
    extracted = extract_json(zip_path, date_dir)
    print(f"[{row.date}] extracted {extracted} JSON files", flush=True)

    if not keep_zip:
        zip_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if args.date:
        wanted = set(args.date)
        rows = [row for row in rows if row.date in wanted]
    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        print("No manifest rows selected.")
        return 0

    total_bytes = sum(row.total_bytes for row in rows)
    print(
        f"Selected {len(rows)} dataset(s), estimated total {total_bytes / 1024**3:.1f} GiB.",
        flush=True,
    )
    for row in rows:
        print(f"  {row.date}: {row.url}", flush=True)

    if args.dry_run:
        return 0

    check_free_space(args.output_dir, rows, args.ignore_space_check)
    with tempfile.TemporaryDirectory(prefix="daily_dataset_", dir=args.output_dir) as tmp:
        Path(tmp).rmdir()
    for row in rows:
        download_row(row, args.output_dir, args.force, args.keep_zip, args.zip_only)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
