from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Listing


@runtime_checkable
class Source(Protocol):
    name: str

    def fetch(self) -> list[Listing]: ...


def normalise_text(text: str) -> str:
    """Collapse whitespace; strip leading/trailing."""
    import re
    return re.sub(r"\s+", " ", text or "").strip()


def make_absolute(url: str, base: str) -> str:
    """Resolve a relative URL against a base URL."""
    from urllib.parse import urljoin
    if not url:
        return ""
    return urljoin(base, url)
