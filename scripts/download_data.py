"""Download CAFA 6 Kaggle files into data/raw using KaggleHub."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cafa6.io import RAW_DIR, REQUIRED_RAW_FILES, check_raw_files, format_raw_file_summary


COMPETITION = "cafa-6-protein-function-prediction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Destination directory for required raw files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in the raw directory.",
    )
    return parser.parse_args()


def copy_required_files(download_dir: Path, raw_dir: Path, overwrite: bool = False) -> list[Path]:
    """Copy required files from a KaggleHub download directory into data/raw."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []

    for file_name in REQUIRED_RAW_FILES:
        matches = sorted(download_dir.rglob(file_name))
        source = matches[0] if matches else download_dir / file_name
        destination = raw_dir / file_name

        if not source.is_file():
            continue
        if destination.exists() and not overwrite:
            continue

        shutil.copy2(source, destination)
        copied.append(destination)

    return copied


def main() -> int:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is required for downloads. Install it, then rerun this script."
        ) from exc

    args = parse_args()
    download_path = Path(kagglehub.competition_download(COMPETITION))
    copied = copy_required_files(download_path, args.raw_dir, overwrite=args.overwrite)

    print(f"KaggleHub download path: {download_path}")
    print(f"Copied files: {len(copied)}")
    for path in copied:
        print(f"- {path}")

    report = check_raw_files(args.raw_dir)
    print(format_raw_file_summary(report))

    return 0 if report["all_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
