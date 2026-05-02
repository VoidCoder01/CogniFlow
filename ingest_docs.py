#!/usr/bin/env python3
"""CLI: chunk documents and upsert into the Chroma collection."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core.document_processor import DocumentProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SUFFIXES = {".pdf", ".md", ".markdown", ".html", ".htm"}


def _collect_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        p = p.expanduser().resolve()
        if p.is_file():
            if p.suffix.lower() in _SUFFIXES:
                files.append(p)
            else:
                logger.warning("Skipping unsupported file type: %s", p)
        elif p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file() and child.suffix.lower() in _SUFFIXES:
                    files.append(child)
        else:
            logger.error("Path not found: %s", p)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest PDF, Markdown, or HTML into ChromaDB via CogniFlow.",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Files or directories to ingest",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk only; do not write to the vector store",
    )
    parser.add_argument(
        "--session-id",
        default="__global__",
        metavar="ID",
        help="Tag all chunks with this session id (default: __global__ for shared corpus)",
    )
    parser.add_argument(
        "--user-id",
        default="",
        metavar="ID",
        help="Optional user id for metadata (enables user-scoped retrieval with --session-id).",
    )
    args = parser.parse_args(argv)

    targets = _collect_files(args.paths)
    if not targets:
        logger.error("No ingestible files found.")
        return 1

    processor = DocumentProcessor()
    store = None
    if not args.dry_run:
        from core.vector_store import VectorStore

        store = VectorStore()

    total_chunks = 0
    for path in targets:
        chunks = processor.process_file(
            path,
            session_id=args.session_id,
            user_id=args.user_id or None,
        )
        total_chunks += len(chunks)
        if not args.dry_run and chunks:
            store.add_documents(chunks)
        logger.info("%s → %s chunks", path, len(chunks))

    logger.info("Done. %s files, %s chunks total.", len(targets), total_chunks)
    if args.dry_run:
        logger.info("Dry run: nothing written to Chroma.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
