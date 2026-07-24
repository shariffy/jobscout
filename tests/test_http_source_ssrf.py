"""SSRF guard tests for HttpSource — no real network, DNS is mocked."""
from __future__ import annotations

import httpx
import pytest

from jobscout.config import SourceConfig
from jobscout.sources import http_source
from jobscout.sources.http_source import HttpSource, SSRFError, _assert_safe_url


def _fake_getaddrinfo(mapping: dict[str, str]):
    def fn(host, *args, **kwargs):
        ip = mapping.get(host)
        if ip is None:
            raise OSError(f"no mapping for {host!r}")
        return [(None, None, None, None, (ip, 0))]

    return fn


def make_http_source(url: str) -> HttpSource:
    return HttpSource(SourceConfig(name="t", type="http", url=url))


# --- _assert_safe_url ---

def test_rejects_private_ip(monkeypatch):
    monkeypatch.setattr(
        http_source.socket, "getaddrinfo", _fake_getaddrinfo({"internal.example.com": "10.0.0.5"})
    )
    with pytest.raises(SSRFError):
        _assert_safe_url("http://internal.example.com/x")


def test_rejects_loopback(monkeypatch):
    monkeypatch.setattr(
        http_source.socket, "getaddrinfo", _fake_getaddrinfo({"local.example.com": "127.0.0.1"})
    )
    with pytest.raises(SSRFError):
        _assert_safe_url("http://local.example.com/x")


def test_rejects_link_local_metadata_ip(monkeypatch):
    monkeypatch.setattr(
        http_source.socket,
        "getaddrinfo",
        _fake_getaddrinfo({"metadata.example.com": "169.254.169.254"}),
    )
    with pytest.raises(SSRFError):
        _assert_safe_url("http://metadata.example.com/x")


def test_rejects_non_http_scheme():
    with pytest.raises(SSRFError):
        _assert_safe_url("file:///etc/passwd")


def test_allows_public_ip(monkeypatch):
    monkeypatch.setattr(
        http_source.socket, "getaddrinfo", _fake_getaddrinfo({"example.com": "93.184.216.34"})
    )
    _assert_safe_url("http://example.com/x")  # no raise


# --- HttpSource._get follows redirects, checking every hop ---

def test_get_rejects_redirect_to_private_address(monkeypatch):
    monkeypatch.setattr(
        http_source.socket,
        "getaddrinfo",
        _fake_getaddrinfo(
            {
                "board.example.com": "93.184.216.34",
                "internal.example.com": "127.0.0.1",
            }
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "board.example.com":
            return httpx.Response(302, headers={"location": "http://internal.example.com/x"})
        return httpx.Response(200, text="should never be reached")

    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.Client
    monkeypatch.setattr(
        http_source.httpx, "Client", lambda **kwargs: real_client_cls(transport=transport)
    )

    source = make_http_source("http://board.example.com/jobs")
    with pytest.raises(SSRFError):
        source._get("http://board.example.com/jobs")


def test_get_follows_safe_redirect(monkeypatch):
    monkeypatch.setattr(
        http_source.socket,
        "getaddrinfo",
        _fake_getaddrinfo(
            {
                "board.example.com": "93.184.216.34",
                "board2.example.com": "93.184.216.35",
            }
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "board.example.com":
            return httpx.Response(302, headers={"location": "http://board2.example.com/x"})
        return httpx.Response(200, text="ok")

    transport = httpx.MockTransport(handler)
    real_client_cls = httpx.Client
    monkeypatch.setattr(
        http_source.httpx, "Client", lambda **kwargs: real_client_cls(transport=transport)
    )

    source = make_http_source("http://board.example.com/jobs")
    assert source._get("http://board.example.com/jobs") == "ok"
