"""SSRF guard tests for HttpSource — no real network, DNS is mocked."""
from __future__ import annotations

import httpx
import pytest

from jobscout.sources import http_source
from jobscout.sources.http_source import SSRFError, _assert_safe_url, safe_get


def _fake_getaddrinfo(mapping: dict[str, str]):
    def fn(host, *args, **kwargs):
        ip = mapping.get(host)
        if ip is None:
            raise OSError(f"no mapping for {host!r}")
        return [(None, None, None, None, (ip, 0))]

    return fn


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


def test_rejects_cgnat(monkeypatch):
    # 100.64.0.0/10 is CGNAT — not covered by is_private, but not is_global either.
    monkeypatch.setattr(
        http_source.socket, "getaddrinfo", _fake_getaddrinfo({"cgnat.example.com": "100.64.0.1"})
    )
    with pytest.raises(SSRFError):
        _assert_safe_url("http://cgnat.example.com/x")


# --- safe_get rejects disallowed hosts ---

def test_safe_get_rejects_private_host(monkeypatch):
    monkeypatch.setattr(
        http_source.socket,
        "getaddrinfo",
        _fake_getaddrinfo({"internal.example.com": "10.0.0.5"}),
    )
    with pytest.raises(SSRFError):
        safe_get("http://internal.example.com/x")


def test_safe_get_rejects_cgnat_host(monkeypatch):
    monkeypatch.setattr(
        http_source.socket,
        "getaddrinfo",
        _fake_getaddrinfo({"cgnat.example.com": "100.64.0.1"}),
    )
    with pytest.raises(SSRFError):
        safe_get("http://cgnat.example.com/x")


# --- safe_get follows redirects, checking every hop ---

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

    with pytest.raises(SSRFError):
        safe_get("http://board.example.com/jobs")


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

    assert safe_get("http://board.example.com/jobs") == "ok"
