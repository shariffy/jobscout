"""Source adapter tests — no network, no AI."""
from pathlib import Path

import feedparser

from jobscout.config import SelectorConfig, SourceConfig
from jobscout.sources.feed_source import FeedSource
from jobscout.sources.http_source import HttpSource

FIXTURES = Path(__file__).parent / "fixtures"


def make_http_source(**kwargs) -> HttpSource:
    defaults = dict(
        name="test-board",
        type="http",
        url="https://example.com/jobs",
        selectors=SelectorConfig(
            container="div.job-card",
            title="h2.job-title",
            company="span.company",
            location="span.location",
            url="a.job-link",
            description="div.description",
        ),
    )
    defaults.update(kwargs)
    return HttpSource(SourceConfig(**defaults))


# --- HttpSource ---

def test_http_parses_listings():
    html = (FIXTURES / "jobs.html").read_text()
    source = make_http_source()
    listings = source.parse_html(html, base_url="https://example.com")

    assert len(listings) == 2  # malformed card skipped


def test_http_title_and_company():
    html = (FIXTURES / "jobs.html").read_text()
    listings = make_http_source().parse_html(html, "https://example.com")

    assert listings[0].title == "Senior Python Engineer"
    assert listings[0].company == "Widgets Ltd"
    assert listings[1].title == "Staff Engineer"
    assert listings[1].company == "Acme Corp"


def test_http_resolves_relative_url():
    html = (FIXTURES / "jobs.html").read_text()
    listings = make_http_source().parse_html(html, "https://example.com")

    assert listings[0].url == "https://example.com/jobs/101"


def test_http_keeps_absolute_url():
    html = (FIXTURES / "jobs.html").read_text()
    listings = make_http_source().parse_html(html, "https://example.com")

    assert listings[1].url == "https://acme.example.com/jobs/42"


def test_http_location():
    html = (FIXTURES / "jobs.html").read_text()
    listings = make_http_source().parse_html(html, "https://example.com")

    assert listings[0].location == "London / Remote"
    assert listings[1].location == "Remote"


def test_http_description():
    html = (FIXTURES / "jobs.html").read_text()
    listings = make_http_source().parse_html(html, "https://example.com")

    assert "widget platform" in listings[0].description


def test_http_source_name():
    html = (FIXTURES / "jobs.html").read_text()
    listings = make_http_source(name="MyBoard").parse_html(html, "https://example.com")

    assert all(lst.source_name == "MyBoard" for lst in listings)


def test_http_listing_hash_stable():
    html = (FIXTURES / "jobs.html").read_text()
    source = make_http_source()
    a = source.parse_html(html, "https://example.com")
    b = source.parse_html(html, "https://example.com")

    assert a[0].hash == b[0].hash


def test_http_listings_have_unique_hashes():
    html = (FIXTURES / "jobs.html").read_text()
    listings = make_http_source().parse_html(html, "https://example.com")

    hashes = [lst.hash for lst in listings]
    assert len(set(hashes)) == len(hashes)


# --- FeedSource ---

def make_feed_source(**kwargs) -> FeedSource:
    defaults = dict(name="test-feed", type="feed", url="https://feed.example.com/rss")
    defaults.update(kwargs)
    return FeedSource(SourceConfig(**defaults))


def test_feed_parses_entries():
    xml = (FIXTURES / "feed.xml").read_text()
    feed = feedparser.parse(xml)
    listings = make_feed_source().parse_feed(feed)

    assert len(listings) == 2  # orphan (no link) skipped


def test_feed_title_and_url():
    xml = (FIXTURES / "feed.xml").read_text()
    feed = feedparser.parse(xml)
    listings = make_feed_source().parse_feed(feed)

    assert listings[0].title == "Engineering Manager, Platform"
    assert listings[0].url == "https://feed.example.com/jobs/1"
    assert listings[1].title == "Senior Backend Developer"


def test_feed_description():
    xml = (FIXTURES / "feed.xml").read_text()
    feed = feedparser.parse(xml)
    listings = make_feed_source().parse_feed(feed)

    assert "platform" in listings[0].description.lower()


def test_feed_company_from_author():
    xml = (FIXTURES / "feed.xml").read_text()
    feed = feedparser.parse(xml)
    listings = make_feed_source().parse_feed(feed)

    assert listings[0].company == "TechCo"
    assert listings[1].company == "StartupXYZ"


def test_feed_source_name():
    xml = (FIXTURES / "feed.xml").read_text()
    feed = feedparser.parse(xml)
    listings = make_feed_source(name="HN Feed").parse_feed(feed)

    assert all(lst.source_name == "HN Feed" for lst in listings)
