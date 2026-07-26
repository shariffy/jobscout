from __future__ import annotations


def normalise_text(text: str) -> str:
    """Collapse whitespace; strip leading/trailing."""
    import re
    return re.sub(r"\s+", " ", text or "").strip()
