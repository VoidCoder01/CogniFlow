from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from pypdf import PdfReader

from config import settings
from core.models import DocumentChunk, DocumentMetadata
from core.text_splitting import iter_markdown_sections, recursive_chunk

logger = logging.getLogger(__name__)

_CODE_FENCE_RE = re.compile(r"^\s*```", re.MULTILINE)
_MD_TITLE_RE = re.compile(r"^\s*#\s+(.+)$", re.MULTILINE)


class DocumentProcessor:
    """Extract text from PDF / Markdown / HTML, chunk with overlap, attach metadata."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._upload_original_name: str | None = None
        self._upload_doc_id: str | None = None
        self._ingest_session_id: str = ""
        self._ingest_user_id: str = ""
        self._ingest_content_hash: str = ""

    def process_file(
        self,
        path: str | Path,
        *,
        original_filename: Optional[str] = None,
        doc_instance_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> list[DocumentChunk]:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(str(p))

        self._upload_original_name = (original_filename or "").strip() or None
        self._upload_doc_id = (doc_instance_id or "").strip() or None
        self._ingest_session_id = (session_id or "").strip()
        self._ingest_user_id = (user_id or "").strip()
        self._ingest_content_hash = (content_hash or "").strip()
        try:
            suffix = p.suffix.lower()
            if suffix == ".pdf":
                out = self._process_pdf(p)
            elif suffix in (".md", ".markdown"):
                out = self._process_markdown(p)
            elif suffix in (".html", ".htm"):
                out = self._process_html(p)
            else:
                raise ValueError(f"Unsupported document type: {suffix} ({p})")
            for c in out:
                c.metadata.session_id = self._ingest_session_id
                c.metadata.user_id = self._ingest_user_id
                c.metadata.content_hash = self._ingest_content_hash
            return out
        finally:
            self._upload_original_name = None
            self._upload_doc_id = None
            self._ingest_session_id = ""
            self._ingest_user_id = ""
            self._ingest_content_hash = ""

    def process_paths(
        self,
        paths: list[str | Path],
        *,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> list[DocumentChunk]:
        out: list[DocumentChunk] = []
        for raw in paths:
            out.extend(self.process_file(raw, session_id=session_id, user_id=user_id))
        return out

    def _process_pdf(self, path: Path) -> list[DocumentChunk]:
        reader = PdfReader(str(path))
        meta = reader.metadata or {}
        doc_title = self._clean_str(meta.get("/Title") or meta.get("title"))
        title = doc_title or path.stem

        pieces: list[DocumentChunk] = []
        for i, page in enumerate(reader.pages):
            page_num = i + 1
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            for chunk_text in recursive_chunk(text, self.chunk_size, self.chunk_overlap):
                meta_dm = DocumentMetadata(
                    source=str(path),
                    doc_type="pdf",
                    title=title,
                    section_headers=[],
                    has_code_blocks=bool(_CODE_FENCE_RE.search(chunk_text)),
                    version="",
                    page_number=page_num,
                    chunk_index=0,
                    total_chunks=1,
                )
                self._stamp_upload_metadata(path, title, meta_dm)
                pieces.append(DocumentChunk(content=chunk_text, metadata=meta_dm))

        if not pieces:
            logger.warning("No extractable text in PDF: %s", path)
            return []

        return self._finalize_indices(pieces)

    def _process_markdown(self, path: Path) -> list[DocumentChunk]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        title = self._first_markdown_title(raw) or path.stem

        pieces: list[DocumentChunk] = []
        for headers, body in iter_markdown_sections(raw):
            body = body.strip()
            if not body:
                continue
            for chunk_text in recursive_chunk(body, self.chunk_size, self.chunk_overlap):
                meta_dm = DocumentMetadata(
                    source=str(path),
                    doc_type="markdown",
                    title=title,
                    section_headers=headers,
                    has_code_blocks=bool(_CODE_FENCE_RE.search(chunk_text)),
                    version="",
                    page_number=None,
                    chunk_index=0,
                    total_chunks=1,
                )
                self._stamp_upload_metadata(path, title, meta_dm)
                pieces.append(DocumentChunk(content=chunk_text, metadata=meta_dm))

        if not pieces:
            logger.warning("No chunkable content in Markdown: %s", path)
            return []

        return self._finalize_indices(pieces)

    def _process_html(self, path: Path) -> list[DocumentChunk]:
        raw = path.read_text(encoding="utf-8", errors="replace")
        try:
            soup = BeautifulSoup(raw, "lxml")
        except Exception:
            soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        title_tag = soup.find("title")
        h1 = soup.find(["h1", "h2"])
        title = (
            self._clean_str(title_tag.get_text()) if title_tag else None
        ) or (
            self._clean_str(h1.get_text()) if h1 else None
        ) or path.stem

        body = soup.find("body") or soup
        text = body.get_text(separator="\n", strip=True)
        has_pre = bool(soup.find("pre") or soup.find("code"))

        pieces: list[DocumentChunk] = []
        for chunk_text in recursive_chunk(text, self.chunk_size, self.chunk_overlap):
            meta_dm = DocumentMetadata(
                source=str(path),
                doc_type="html",
                title=title,
                section_headers=[],
                has_code_blocks=has_pre or bool(_CODE_FENCE_RE.search(chunk_text)),
                version="",
                page_number=None,
                chunk_index=0,
                total_chunks=1,
            )
            self._stamp_upload_metadata(path, title, meta_dm)
            pieces.append(DocumentChunk(content=chunk_text, metadata=meta_dm))

        if not pieces:
            logger.warning("No chunkable content in HTML: %s", path)
            return []

        return self._finalize_indices(pieces)

    @staticmethod
    def _finalize_indices(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        total = len(chunks)
        for i, c in enumerate(chunks):
            c.metadata.chunk_index = i
            c.metadata.total_chunks = max(total, 1)
        return chunks

    def _stamp_upload_metadata(self, path: Path, title: str, meta: DocumentMetadata) -> None:
        """Replace temp paths with original filename + id so duplicate names stay distinct."""
        if not self._upload_doc_id:
            meta.source = str(path)
            return
        short = self._upload_doc_id.replace("-", "")[:8]
        orig = self._upload_original_name or path.name
        meta.original_filename = orig
        meta.doc_instance_id = self._upload_doc_id
        meta.source = f"{orig} · {short}"
        meta.title = f"{title} · {short}"

    @staticmethod
    def _first_markdown_title(text: str) -> str | None:
        m = _MD_TITLE_RE.search(text)
        return m.group(1).strip() if m else None

    @staticmethod
    def _clean_str(value: object | None) -> str | None:
        if value is None:
            return None
        s = str(value).strip()
        return s or None
