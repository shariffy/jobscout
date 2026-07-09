from __future__ import annotations

from ..config import SourceConfig
from .base import Source
from .feed_source import FeedSource
from .http_source import HttpSource


def build_source(config: SourceConfig) -> Source:
    if config.type == "http":
        return HttpSource(config)
    if config.type == "feed":
        return FeedSource(config)
    if config.type == "browser":
        from .browser_source import BrowserSource
        return BrowserSource(config)
    raise ValueError(f"Unknown source type: {config.type!r}")


def build_enabled_sources(configs: list[SourceConfig]) -> list[Source]:
    return [build_source(c) for c in configs if c.enabled]
