from __future__ import annotations

import os
import time

from bs4 import BeautifulSoup

from ..config import SourceConfig
from ..models import Listing
from .http_source import _UA, _assert_safe_url, _extract_jd, page_urls, parse_listings

_TIMEOUT_MS = 30_000


class BrowserSource:
    """Scrapes a job listing page with a headless Chromium browser (Playwright).

    Use this for boards that render listings with JavaScript, or that require a
    login. Selectors, pagination, and JD enrichment work exactly as they do for
    HttpSource — the only difference is the page is rendered in a real browser
    before it's parsed. Reuses `page_urls`, `parse_listings`, and `_extract_jd`
    from http_source rather than re-implementing them.

    Requires `playwright install chromium` to have been run once; see `_launch`
    for the error raised when it hasn't.

    Login is a best-effort heuristic (see `_login`) — it only works for simple
    single-page username+password forms, because SourceConfig has no way to
    name the login form's own field selectors. Boards with multi-step logins,
    SSO, CAPTCHAs, or 2FA are not supported; write a source-specific login
    routine instead if you need one.
    """

    def __init__(self, config: SourceConfig) -> None:
        self._cfg = config
        self.name = config.name

    def fetch(self) -> list[Listing]:
        sync_playwright = _import_sync_playwright()
        listings: list[Listing] = []
        with sync_playwright() as p:
            browser = self._launch(p)
            try:
                page = browser.new_page(user_agent=_UA)
                page.set_default_timeout(_TIMEOUT_MS)
                self._login(page)
                for i, url in enumerate(page_urls(self._cfg)):
                    if i:
                        time.sleep(0.5)  # be polite when paginating
                    # ponytail: first-hop only; full request interception deferred —
                    # Chromium does its own DNS/redirects/subresource loads.
                    _assert_safe_url(url)
                    page.goto(url, wait_until="networkidle")
                    page_listings = parse_listings(
                        page.content(), url, self._cfg.selectors, self.name
                    )
                    if not page_listings:
                        break  # empty page — end of results
                    listings.extend(page_listings)
                if self._cfg.selectors.fetch_detail:
                    listings = [self._enrich(page, listing) for listing in listings]
            finally:
                browser.close()
        return listings

    def _launch(self, playwright):
        from playwright.sync_api import Error as PlaywrightError

        try:
            return playwright.chromium.launch(headless=True)
        except PlaywrightError as exc:
            raise RuntimeError(
                f"[{self.name}] Chromium isn't installed for Playwright. Run "
                "`playwright install chromium` once, then try again."
            ) from exc

    def _login(self, page) -> None:
        """Best-effort login for simple username+password forms.

        SourceConfig only carries `login_url` + credential env var names, not
        selectors for the login form's fields — there's nowhere in config to
        describe a bespoke login flow. So this heuristic fills the first
        visible email/text input and the first password input, then submits.
        It works for plain single-page login forms and nothing fancier
        (no SSO redirects, multi-step logins, CAPTCHAs, or 2FA). If a board
        needs more than that, this heuristic will silently fail to log in and
        the subsequent scrape will proceed unauthenticated — likely returning
        nothing or a public-only view. Write a dedicated login routine instead.
        """
        cfg = self._cfg
        if not cfg.login_url:
            return
        if not cfg.username_env or not cfg.password_env:
            raise RuntimeError(
                f"[{self.name}] login_url is set but username_env/password_env are not"
            )
        username = os.environ.get(cfg.username_env)
        password = os.environ.get(cfg.password_env)
        if not username or not password:
            raise RuntimeError(
                f"[{self.name}] login_url is set but {cfg.username_env} and/or "
                f"{cfg.password_env} are not set in the environment"
            )

        page.goto(cfg.login_url, wait_until="domcontentloaded")
        page.locator("input[type=email], input[type=text]").first.fill(username)
        password_field = page.locator("input[type=password]").first
        password_field.fill(password)

        submit = page.locator("button[type=submit]")
        if submit.count() > 0:
            submit.first.click()
        else:
            password_field.press("Enter")
        try:
            page.wait_for_load_state("networkidle")
        except Exception:
            pass  # some login pages never go fully idle (polling, websockets, ...)

    def _enrich(self, page, listing: Listing) -> Listing:
        """Open the listing's own page in the browser and pull a richer description."""
        try:
            # ponytail: first-hop only; full request interception deferred —
            # Chromium does its own DNS/redirects/subresource loads.
            _assert_safe_url(listing.url)
            page.goto(listing.url, wait_until="networkidle")
            html = page.content()
        except Exception:
            return listing
        soup = BeautifulSoup(html, "html.parser")
        desc = _extract_jd(soup)
        if desc and len(desc) > len(listing.description or ""):
            return listing.model_copy(update={"description": desc})
        return listing


def _import_sync_playwright():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ImportError(
            "The 'browser' source type requires Playwright's browser binary, which "
            "isn't downloaded automatically by pip. Run `playwright install chromium` "
            "once, then try again."
        ) from exc
    return sync_playwright
