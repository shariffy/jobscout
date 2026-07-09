from __future__ import annotations

import feedparser

from ..config import SourceConfig
from ..models import Listing
from .base import normalise_text


class FeedSource:
    """Parses RSS / Atom feeds using feedparser."""

    def __init__(self, config: SourceConfig) -> None:
        self._cfg = config
        self.name = config.name

    def fetch(self) -> list[Listing]:
        feed = feedparser.parse(self._cfg.url)
        return self._parse(feed)

    def _parse(self, feed: feedparser.FeedParserDict) -> list[Listing]:
        listings: list[Listing] = []
        for entry in feed.entries:
            title = normalise_text(entry.get("title", ""))
            url = entry.get("link", "")
            description = normalise_text(
                entry.get("summary", "") or entry.get("content", [{}])[0].get("value", "")
            )
            author = entry.get("author", "")
            company = author or (feed.feed.get("title", "") or self.name)

            if not title or not url or not url.startswith("http"):
                continue

            listings.append(
                Listing(
                    source_name=self.name,
                    title=title,
                    company=company,
                    location="",
                    url=url,
                    description=description[:4000],
                    raw={"feed_id": entry.get("id", "")},
                )
            )
        return listings

    def parse_feed(self, feed: feedparser.FeedParserDict) -> list[Listing]:
        """Exposed for testing."""
        return self._parse(feed)
