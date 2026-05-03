#!/usr/bin/env python3
"""Remove the local Chroma persist directory (fixes KeyError '_type' / catalog skew after upgrades).

Usage:
  python scripts/reset_chroma.py --yes

Then restart the API and re-upload or run: python ingest_docs.py ...
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Delete Chroma on-disk data (CHROMA_PERSIST_DIR).")
    p.add_argument(
        "--yes",
        action="store_true",
        help="Required; deletes the directory without interactive confirm.",
    )
    args = p.parse_args()
    if not args.yes:
        print("Refusing to delete without --yes.", file=sys.stderr)
        return 2

    repo = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo))
    try:
        from dotenv import load_dotenv

        load_dotenv(repo / ".env")
    except ImportError:
        pass

    from config import settings

    target = Path(settings.chroma_persist_dir).expanduser()
    if not target.exists():
        print(f"Nothing to remove: {target} does not exist.")
        return 0
    shutil.rmtree(target)
    print(f"Removed: {target}")
    print("Restart the API (uvicorn). Re-upload documents or run your ingest pipeline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
