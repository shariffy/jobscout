from __future__ import annotations

import ipaddress
import re
import socket
import time
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from ..config import SelectorConfig, SourceConfig
from ..models import Listing
from .base import normalise_text

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"  # noqa: E501
_MAX_REDIRECTS = 5


class SSRFError(ValueError):
    """A URL (or a redirect hop) points at a host we refuse to fetch."""


def _assert_safe_url(url: str) -> None:
    """Reject non-http(s) schemes and hosts that resolve to a loopback, private,
    link-local, or otherwise reserved address (e.g. 169.254.169.254 cloud metadata,
    127.0.0.1). Listing-controlled URLs and redirect targets are attacker input —
    a hostile board could point either at an internal address, so every hop is
    checked, not just the URL a source config names."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFError(f"Refusing to fetch non-http(s) URL: {url!r}")
    host = parsed.hostname
    if not host:
        raise SSRFError(f"URL has no host: {url!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as exc:
        raise SSRFError(f"Could not resolve host {host!r}: {exc}") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise SSRFError(
                f"Refusing to fetch {url!r}: host {host!r} resolves to disallowed address {ip}"
            )


def safe_get(url: str) -> str:
    """Fetch url, following redirects manually so every hop gets the SSRF host check.

    Raises SSRFError if any hop resolves to a disallowed address, uses a
    non-http(s) scheme, or exceeds the redirect limit.  A hostile job board could
    redirect a listing URL at an internal address — this catches that on every hop,
    not just the first.
    """
    with httpx.Client(follow_redirects=False, timeout=30) as client:
        for _ in range(_MAX_REDIRECTS + 1):
            _assert_safe_url(url)
            resp = client.get(url, headers={"User-Agent": _UA})
            if resp.is_redirect and resp.headers.get("location"):
                url = urljoin(url, resp.headers["location"])
                continue
            resp.raise_for_status()
            return resp.text
    raise SSRFError(f"Too many redirects (> {_MAX_REDIRECTS}) fetching {url!r}")


class HttpSource:
    """Scrapes a job listing page using config-driven CSS selectors."""

    def __init__(self, config: SourceConfig) -> None:
        self._cfg = config
        self.name = config.name

    def fetch(self) -> list[Listing]:
        listings: list[Listing] = []
        for i, url in enumerate(page_urls(self._cfg)):
            if i:
                time.sleep(0.5)  # be polite when paginating
            page = self._parse(safe_get(url), url)
            if not page:
                break  # empty page — end of results
            listings.extend(page)
        if self._cfg.selectors.fetch_detail:
            listings = [self._enrich(lst) for lst in listings]
        return listings

    def _enrich(self, listing: Listing) -> Listing:
        """Fetch the individual job page and extract a richer description."""
        try:
            html = safe_get(listing.url)
        except Exception:
            return listing
        soup = BeautifulSoup(html, "html.parser")
        desc = _extract_jd(soup)
        if desc and len(desc) > len(listing.description or ""):
            return listing.model_copy(update={"description": desc})
        return listing

    def _parse(self, html: str, base_url: str) -> list[Listing]:
        return parse_listings(html, base_url, self._cfg.selectors, self.name)

    def parse_html(self, html: str, base_url: str = "") -> list[Listing]:
        """Exposed for testing — parse arbitrary HTML without network."""
        return self._parse(html, base_url or self._cfg.url)


def page_urls(cfg: SourceConfig):
    """Yield the sequence of page URLs to fetch for a paginated source.

    Shared by HttpSource and BrowserSource. Advances `page_param` by `page_size`
    each time, starting from its current value in `cfg.url`.
    """
    if cfg.pages <= 1 or not cfg.page_param:
        yield cfg.url
        return
    m = re.search(rf"[?&]{re.escape(cfg.page_param)}=([0-9]+)", cfg.url)
    base = int(m.group(1)) if m else 0
    for i in range(cfg.pages):
        yield _set_query_param(cfg.url, cfg.page_param, base + i * cfg.page_size)


def parse_listings(
    html: str, base_url: str, sel: SelectorConfig, source_name: str
) -> list[Listing]:
    """Parse listing cards out of a page's HTML using config-driven selectors.

    Shared by HttpSource and BrowserSource so both scrape with the same field
    extraction rules — the only difference between them is how the HTML was
    obtained (plain GET vs. a rendered browser page).
    """
    soup = BeautifulSoup(html, "html.parser")
    listings: list[Listing] = []

    containers = soup.select(sel.container) if sel.container else [soup]
    for el in containers:
        title = _text(el, sel.title)
        company = _text(el, sel.company)
        location = _text(el, sel.location)
        description = _text(el, sel.description)
        url = _attr(el, sel.url, "href")
        url = urljoin(base_url, url) if url else ""

        if not title or not url:
            continue

        listings.append(
            Listing(
                source_name=source_name,
                title=title,
                company=company or source_name,
                location=location,
                url=url,
                description=description,
                raw={"html_snippet": str(el)[:2000]},
            )
        )

    return listings


def _set_query_param(url: str, param: str, value: int) -> str:
    """Replace (or append) a numeric query parameter, leaving the rest of the
    url untouched so pre-encoded values (e.g. %2C in a location) are preserved."""
    pattern = rf"([?&]{re.escape(param)}=)[^&]*"
    if re.search(pattern, url):
        return re.sub(pattern, rf"\g<1>{value}", url)
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}{param}={value}"


def _clean_jsonld_description(raw: str) -> str:
    """JSON-LD `description` fields often embed HTML, sometimes entity-escaped
    (LinkedIn stores the whole JD as escaped HTML in a string). Unescape entities
    and strip any tags so the scorer receives clean prose rather than markup."""
    import html as _html

    text = _html.unescape(raw)  # &lt;p&gt; -> <p>
    if "<" in text and ">" in text:
        text = BeautifulSoup(text, "html.parser").get_text(" ")
    return normalise_text(text)


def _extract_jd(soup: BeautifulSoup) -> str:
    """Extract a job description from an individual job page using structural heuristics."""
    import json as _json
    # 1. JSON-LD structured data
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = _json.loads(script.string or "")
            if isinstance(data, dict) and data.get("description"):
                return _clean_jsonld_description(data["description"])
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
