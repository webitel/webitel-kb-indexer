"""Splitting an article body into the chunks that carry it into the index."""

from __future__ import annotations

import re
from dataclasses import dataclass

# The per-space strategy this module implements. `kb.space.chunking_strategy`
# is free text, so the pipeline compares it against this value and refuses a
# space it cannot chunk instead of quietly applying a different strategy.
STRATEGY = "recursive_markdown"

# How the heading path is written into a chunk. The content of a chunk is
# therefore not a verbatim slice of the body.
CRUMB_SEPARATOR = " / "

# A deep heading path never takes more than this share of a chunk: the text is
# what gets retrieved, the path is only context for it.
_PATH_SHARE = 4

_HEADING = re.compile(r"^(#{1,6})\s+(\S.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
# Where an overlap may start: a whole sentence when the tail holds one, a
# whole word otherwise.
_OVERLAP_STARTS = (re.compile(r"(?<=[.!?…])\s"), re.compile(r"\s"))

# Where a block may be cut, from the least damaging to the most. Every pattern
# is zero-width, so the parts still join back into the original text.
_BREAKS = (
    re.compile(r"(?<=\n)"),
    re.compile(r"(?<=[.!?…])(?=\s)"),
    re.compile(r"(?<=\s)"),
)


@dataclass(frozen=True, slots=True)
class Policy:
    """Chunk size, in characters."""

    max_chars: int = 1000
    overlap_chars: int = 150

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            msg = f"max_chars must be positive, got {self.max_chars}"
            raise ValueError(msg)

        if not 0 <= self.overlap_chars < self.max_chars:
            msg = f"overlap_chars must be within [0, {self.max_chars}), got {self.overlap_chars}"
            raise ValueError(msg)


DEFAULT = Policy()


@dataclass(frozen=True, slots=True)
class _Section:
    """A part of the body under one heading path."""

    path: tuple[str, ...]
    body: str


@dataclass(frozen=True, slots=True)
class _Piece:
    """A part of a section small enough to place into a chunk."""

    text: str
    code: bool


def split(text: str, *, title: str | None = None, policy: Policy = DEFAULT) -> list[str]:
    """Split a markdown body into chunks, in reading order."""
    chunks: list[str] = []
    for section in _sections(text):
        crumbs = _crumbs(title, section.path)
        opening = f"{crumbs}\n\n" if crumbs else ""
        budget = max(policy.max_chars - len(opening), policy.max_chars // _PATH_SHARE)
        pieces = _pieces(section.body, budget)
        chunks.extend(opening + body for body in _pack(pieces, budget, policy.overlap_chars))

    return chunks


def _sections(text: str) -> list[_Section]:
    """Cut the body at headings, carrying the headings above each part."""
    sections: list[_Section] = []
    # Heading level with its text: a heading closes every heading at or below
    # its own level, which keeps siblings siblings when a document skips one.
    open_headings: list[tuple[int, str]] = []
    body: list[str] = []
    fence = ""

    for line in text.splitlines():
        fence = _fence_state(fence, line)
        heading = _HEADING.match(line) if not fence else None
        if heading is None:
            body.append(line)
            continue

        sections.append(_Section(path=_path(open_headings), body="\n".join(body)))
        level = len(heading.group(1))
        while open_headings and open_headings[-1][0] >= level:
            open_headings.pop()

        open_headings.append((level, heading.group(2)))
        body = []

    sections.append(_Section(path=_path(open_headings), body="\n".join(body)))

    # A heading with nothing under it carries no text to retrieve; it stays in
    # the path of the sections below it.
    return [section for section in sections if section.body.strip()]


def _path(open_headings: list[tuple[int, str]]) -> tuple[str, ...]:
    """The headings above a section, outermost first."""
    return tuple(name for _, name in open_headings)


def _fence_state(fence: str, line: str) -> str:
    """Track an open code fence: markdown inside one is text, not structure."""
    marker = _FENCE.match(line)
    if marker is None:
        return fence

    found = marker.group("fence")
    if not fence:
        return found

    closes = found[0] == fence[0] and len(found) >= len(fence) and not marker.group("info").strip()

    return "" if closes else fence


def _crumbs(title: str | None, path: tuple[str, ...]) -> str:
    """The heading path of a chunk, article subject first.

    Every part is collapsed onto one line: the path opens the chunk, and a
    subject with a line break in it would read as body text.
    """
    named = [" ".join(part.split()) for part in (title or "", *path) if part.strip()]

    return CRUMB_SEPARATOR.join(named)


def _pieces(body: str, budget: int) -> list[_Piece]:
    """Cut a section into parts that each fit the budget, in order."""
    pieces: list[_Piece] = []
    for block, code in _blocks(body):
        parts = _split_code(block, budget) if code else _split_text(block, budget)
        pieces.extend(_Piece(text=part, code=code) for part in parts)

    return pieces


def _blocks(body: str) -> list[tuple[str, bool]]:
    """Split into markdown blocks. A fenced block stays whole, blank lines inside it and all."""
    blocks: list[tuple[str, bool]] = []
    lines: list[str] = []
    fence = ""

    for line in body.splitlines(keepends=True):
        was = fence
        fence = _fence_state(fence, line.rstrip("\n"))

        if not was and fence and lines:
            blocks.append(("".join(lines), False))
            lines = []

        lines.append(line)

        if was and not fence:
            blocks.append(("".join(lines), True))
            lines = []
        elif not fence and not line.strip():
            blocks.append(("".join(lines), False))
            lines = []

    if lines:
        blocks.append(("".join(lines), bool(fence)))

    return _joined(blocks)


def _joined(blocks: list[tuple[str, bool]]) -> list[tuple[str, bool]]:
    """Keep the blank line between two blocks with the block above it."""
    kept: list[tuple[str, bool]] = []
    for text, code in blocks:
        if text.strip():
            kept.append((text, code))
        elif kept:
            above, above_code = kept[-1]
            kept[-1] = (above + text, above_code)

    return kept


def _split_text(text: str, budget: int, level: int = 0) -> list[str]:
    """Cut at the least damaging boundary that makes the text fit."""
    if len(text) <= budget:
        return [text]

    if level >= len(_BREAKS):
        return _hard_cut(text, budget)

    parts = [part for part in _BREAKS[level].split(text) if part]
    if len(parts) <= 1:
        return _split_text(text, budget, level + 1)

    pieces: list[str] = []
    for part in parts:
        pieces.extend(_split_text(part, budget, level + 1))

    return pieces


def _split_code(block: str, budget: int) -> list[str]:
    """Cut a code fence that does not fit, reopening the fence in every part."""
    if len(block) <= budget:
        return [block]

    code = block.rstrip("\n")
    trailer = block[len(code) :]
    lines = code.splitlines(keepends=True)
    opening = lines[0]
    marker = _FENCE.match(opening)
    closed = _fence_state(_fence_state("", opening.rstrip("\n")), lines[-1]) == ""
    # A body that never closed its fence still gets closed parts: an open fence
    # would swallow whatever the retrieval places after the chunk.
    closing = lines[-1] if closed else (marker.group("fence") if marker else "")
    inner = lines[1:-1] if closed else lines[1:]

    # One character is kept for the line break a closing fence needs of its own.
    room = budget - len(opening) - len(closing) - 1
    if room <= 0:
        return _hard_cut(block, budget)

    parts = _fenced(inner, opening, closing, room)
    parts[-1] += trailer

    return parts


def _fenced(inner: list[str], opening: str, closing: str, room: int) -> list[str]:
    """Group code lines into fences of their own."""
    parts: list[str] = []
    group: list[str] = []
    size = 0

    for line in inner:
        for piece in _hard_cut(line, room) if len(line) > room else [line]:
            if group and size + len(piece) > room:
                parts.append(_fence(opening, group, closing))
                group, size = [], 0

            group.append(piece)
            size += len(piece)

    parts.append(_fence(opening, group, closing))

    return parts


def _fence(opening: str, group: list[str], closing: str) -> str:
    """One part of a split fence. The closing marker always starts a line of its own."""
    code = "".join(group)
    if closing and not code.endswith("\n"):
        code += "\n"

    return opening + code + closing


def _hard_cut(text: str, budget: int) -> list[str]:
    """The last resort: a word longer than a whole chunk."""
    return [text[at : at + budget] for at in range(0, len(text), budget)]


def _pack(pieces: list[_Piece], budget: int, overlap: int) -> list[str]:
    """Fill chunks up to the budget, opening each one with the tail of the last."""
    chunks: list[str] = []
    current = ""
    # Plain text at the end of the chunk. Code is cut on line boundaries, so an
    # overlap that reached into it would repeat lines and carry a lone fence
    # into the next chunk.
    prose = 0

    for piece in pieces:
        if current and len(current) + len(piece.text) > budget:
            chunks.append(current.strip())
            carried = "" if piece.code else _tail(current, min(overlap, prose))
            if len(carried) + len(piece.text) > budget:
                carried = ""

            current = carried
            prose = len(carried)

        current += piece.text
        prose = 0 if piece.code else prose + len(piece.text)

    chunks.append(current.strip())

    return [chunk for chunk in chunks if chunk]


def _tail(text: str, size: int) -> str:
    """The end of a chunk, from a whole sentence or word in it, to open the next one."""
    if size <= 0:
        return ""

    tail = text[-size:]
    for start in _OVERLAP_STARTS:
        at = start.search(tail)
        if at is not None:
            return tail[at.end() :]

    return tail
