from __future__ import annotations

from typing import Any

import httpx
from bs4 import BeautifulSoup

from ..config import SourceConfig
from ..models import Listing
from .base import make_absolute, normalise_text

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


class HttpSource:
    """Scrapes a job listing page using config-driven CSS selectors."""

    def __init__(self, config: SourceConfig) -> None:
        self._cfg = config
        self.name = config.name

    def fetch(self) -> list[Listing]:
        html = self._get(self._cfg.url)
        listings = self._parse(html, self._cfg.url)
        if self._cfg.selectors.fetch_detail:
            listings = [self._enrich(l) for l in listings]
        return listings

    def _get(self, url: str) -> str:
        with httpx.Client(follow_redirects=True, timeout=30) as client:
            resp = client.get(url, headers={"User-Agent": _UA})
            resp.raise_for_status()
            return resp.text

    def _enrich(self, listing: Listing) -> Listing:
        """Fetch the individual job page and extract a richer description."""
        try:
            html = self._get(listing.url)
        except Exception:
            return listing
        soup = BeautifulSoup(html, "html.parser")
        desc = _extract_jd(soup)
        if desc and len(desc) > len(listing.description or ""):
            return listing.model_copy(update={"description": desc})
        return listing

    def _parse(self, html: str, base_url: str) -> list[Listing]:
        soup = BeautifulSoup(html, "html.parser")
        sel = self._cfg.selectors
        listings: list[Listing] = []

        containers = soup.select(sel.container) if sel.container else [soup]
        for el in containers:
            title = _text(el, sel.title)
            company = _text(el, sel.company)
            location = _text(el, sel.location)
            description = _text(el, sel.description)
            url = _attr(el, sel.url, "href")
            url = make_absolute(url, base_url)

            if not title or not url:
                continue

            listings.append(
                Listing(
                    source_name=self.name,
                    title=title,
                    company=company or self.name,
                    location=location,
                    url=url,
                    description=description,
                    raw={"html_snippet": str(el)[:2000]},
                )
            )

        return listings

    def parse_html(self, html: str, base_url: str = "") -> list[Listing]:
        """Exposed for testing — parse arbitrary HTML without network."""
        return self._parse(html, base_url or self._cfg.url)


def _extract_jd(soup: BeautifulSoup) -> str:
    """Extract a job description from an individual job page using structural heuristics."""
    import json as _json
    # 1. JSON-LD structured data
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = _json.loads(script.string or "")
            if isinstance(data, dict) and data.get("description"):
                return normalise_text(data["description"])
        except Exception:
            pass
    # 2. <main> tag paragraphs / list items — broad but reliable
    main = soup.find("main")
    root = main if main else soup
    chunks = []
    for el in root.find_all(["p", "li"]):
        text = normalise_text(el.get_text())
        if len(text) > 40:
            chunks.append(text)
    if chunks:
        return " ".join(chunks)
    return ""


def _text(el: Any, selector: str) -> str:
    if not selector:
        return ""
    found = el.select_one(selector)
    return normalise_text(found.get_text()) if found else ""


def _attr(el: Any, selector: str, attr: str) -> str:
    if not selector:
        return ""
    found = el.select_one(selector)
    if not found:
        return ""
    return (found.get(attr) or "").strip()
