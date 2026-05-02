"""Text chunking without langchain_text_splitters (keeps imports free of sentence_transformers)."""

from __future__ import annotations

import re
def recursive_chunk(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Greedy paragraph merge with character-level fallback for oversized paragraphs.
    Overlap is applied between emitted chunks (tail of previous prepended to next start).
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    buf = paragraphs[0]

    for para in paragraphs[1:]:
        candidate = f"{buf}\n\n{para}"
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        chunks.extend(_flush_buffer(buf, chunk_size, chunk_overlap))
        buf = para

    chunks.extend(_flush_buffer(buf, chunk_size, chunk_overlap))
    return [c for c in chunks if c]


def _flush_buffer(buf: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    buf = buf.strip()
    if not buf:
        return []
    if len(buf) <= chunk_size:
        return [buf]
    out: list[str] = []
    i = 0
    n = len(buf)
    while i < n:
        end = min(i + chunk_size, n)
        piece = buf[i:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        nxt = end - chunk_overlap
        if nxt <= i:
            nxt = end
        i = nxt
    return out


_HEADER_LINE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def iter_markdown_sections(text: str) -> list[tuple[list[str], str]]:
    """
    Split markdown on # / ## / ### headers. Each section is (breadcrumb titles, body).
    """
    lines = text.splitlines()
    sections: list[tuple[list[str], str]] = []
    stack: list[str] = []
    body: list[str] = []

    def flush():
        blob = "\n".join(body).strip()
        hdrs = stack.copy()
        if blob or hdrs:
            sections.append((hdrs, blob))

    for line in lines:
        m = _HEADER_LINE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            flush()
            stack = stack[: level - 1] + [title]
            body = []
        else:
            body.append(line)

    flush()

    if not sections and text.strip():
        return [([], text.strip())]

    return sections
