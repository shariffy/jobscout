from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import SourceConfig
from .feed_source import FeedSource
from .http_source import HttpSource

if TYPE_CHECKING:
    from .browser_source import BrowserSource


def build_source(config: SourceConfig) -> HttpSource | FeedSource | BrowserSource:
    if config.type == "http":
        return HttpSource(config)
    if config.type == "feed":
        return FeedSource(config)
    if config.type == "browser":
        from .browser_source import BrowserSource
        return BrowserSource(config)
    raise ValueError(f"Unknown source type: {config.type!r}")


def build_enabled_sources(
    configs: list[SourceConfig],
) -> list[HttpSource | FeedSource | BrowserSource]:
    return [build_source(c) for c in configs if c.enabled]
