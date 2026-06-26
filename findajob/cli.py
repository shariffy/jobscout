from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .store import Store

app = typer.Typer(name="find-a-job", help="Personal job-board analyser and tracker.")
console = Console()

_CONFIG_OPT = typer.Option(None, "--config", "-c", help="Path to config.toml")


def _load(config: str | None):
    return load_config(config)


def _notion(cfg) -> "NotionSync":
    from .notion_sync import NotionSync
    if not cfg.notion.token:
        console.print("[red]No Notion token in config.toml.[/]")
        raise typer.Exit(1)
    if not cfg.notion.database_id:
        console.print(
            "[red]No Notion database_id in config.toml.[/]\n"
            "Run [bold]find-a-job init --notion-parent <page-url>[/] to create the board."
        )
        raise typer.Exit(1)
    return NotionSync(token=cfg.notion.token, database_id=cfg.notion.database_id)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command()
def init(
    config: Optional[str] = _CONFIG_OPT,
    notion_parent: Optional[str] = typer.Option(
        None, "--notion-parent", help="Notion page URL or ID to create the pipeline database under"
    ),
) -> None:
    """Scaffold config.toml and initialise the SQLite database. Optionally create the Notion board."""
    config_path = Path(config) if config else Path("config.toml")

    if not config_path.exists():
        example = Path(__file__).parent.parent / "config.example.toml"
        if example.exists():
            shutil.copy(example, config_path)
            console.print(f"[green]Created[/] {config_path} from example — fill in your values.")
        else:
            console.print(f"[yellow]config.example.toml not found[/]; create {config_path} manually.")
    else:
        console.print(f"[dim]{config_path} already exists — skipping.[/]")

    cfg = _load(str(config_path) if config_path.exists() else None)
    with Store(cfg.store.db_path) as store:
        _ = store
    console.print(f"[green]SQLite ready:[/] {cfg.store.db_path}")

    if notion_parent:
        if not cfg.notion.token:
            console.print("[red]Set notion.token in config.toml first.[/]")
            raise typer.Exit(1)
        from .notion_sync import NotionSync
        ns = NotionSync(token=cfg.notion.token)
        console.print("Creating Notion database…")
        db_id = ns.create_database(notion_parent)
        console.print(f"\n[green]Board created![/] Database ID:\n\n  {db_id}\n")
        console.print(
            "Add this to [bold]config.toml[/]:\n\n"
            f"  [notion]\n  database_id = \"{db_id}\""
        )
    elif cfg.notion.token and cfg.notion.database_id:
        from .notion_sync import NotionSync
        ns = NotionSync(token=cfg.notion.token, database_id=cfg.notion.database_id)
        ok = ns.verify_database()
        console.print(f"[green]Notion board:[/] {'✓ reachable' if ok else '[red]unreachable — check token + database_id[/]'}")


# ---------------------------------------------------------------------------
# profile
# ---------------------------------------------------------------------------

@app.command()
def profile(
    config: Optional[str] = _CONFIG_OPT,
    force: bool = typer.Option(False, "--force", "-f", help="Rebuild even if cache exists"),
) -> None:
    """(Re)build candidate profile from CV + goals and cache to candidate_profile.json."""
    from .profile import build_profile

    cfg = _load(config)
    if not cfg.profile.cv_path:
        console.print("[red]No cv_path set in config.toml.[/]")
        raise typer.Exit(1)

    console.print(f"Reading CV from [bold]{cfg.profile.cv_path}[/]…")
    p = build_profile(cfg, force=force)
    console.print("\n[green]Profile cached.[/]\n")
    console.print(f"[bold]Seniority:[/] {p.seniority}")
    console.print(f"[bold]Summary:[/] {p.summary}")
    console.print(f"[bold]Skills:[/] {', '.join(p.skills)}")
    console.print(f"[bold]Must-haves:[/] {', '.join(p.must_haves)}")
    console.print(f"[bold]Dealbreakers:[/] {', '.join(p.dealbreakers)}")


# ---------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------

@app.command()
def scan(
    config: Optional[str] = _CONFIG_OPT,
    dry_run: bool = typer.Option(False, "--dry-run", help="Fetch but don't save to DB"),
    no_score: bool = typer.Option(False, "--no-score", help="Skip AI scoring after fetching"),
) -> None:
    """Scrape all enabled sources, save new listings, then bulk-score them with Haiku."""
    from .sources.registry import build_enabled_sources

    cfg = _load(config)
    sources = build_enabled_sources(cfg.sources)

    if not sources:
        console.print("[yellow]No enabled sources in config.toml.[/]")
        raise typer.Exit(1)

    total_new = 0
    total_seen = 0
    new_listings = []

    with Store(cfg.store.db_path) as store:
        for source in sources:
            console.print(f"[bold]Scanning[/] {source.name}…", end=" ")
            try:
                listings = source.fetch()
            except Exception as exc:
                console.print(f"[red]error:[/] {exc}")
                continue

            new_count = 0
            for listing in listings:
                if dry_run:
                    new_count += 1
                else:
                    saved, is_new = store.upsert_listing(listing)
                    if is_new:
                        new_count += 1
                        new_listings.append(saved)

            total_new += new_count
            total_seen += len(listings)
            console.print(f"[green]{new_count} new[/] / {len(listings)} fetched")

        if dry_run:
            console.print(f"[dim]Dry run — {total_seen} listings found, nothing saved.[/]")
            return

        console.print(f"\n[bold]Fetch done.[/] {total_new} new listings saved (of {total_seen} fetched).")

        if not new_listings or no_score:
            if new_listings and no_score:
                console.print("[dim]Skipping scoring (--no-score).[/]")
            return

        from .profile import build_profile
        from .scoring import BulkScorer

        if not Path("candidate_profile.json").exists():
            console.print("[yellow]No candidate_profile.json — run [bold]find-a-job profile[/] first.[/]")
            return

        profile_obj = build_profile(cfg)
        scorer = BulkScorer(cfg, profile_obj)

        console.print(f"\n[bold]Scoring[/] {len(new_listings)} new listing(s) with {cfg.ai.bulk_model}…\n")

        scored = 0
        for listing in new_listings:
            try:
                score = scorer.score(listing)
                store.insert_score(score)
                scored += 1
                color = "green" if score.fit_score >= cfg.ai.fit_threshold else (
                    "yellow" if score.fit_score >= 50 else "red"
                )
                console.print(f"  [{color}]{score.fit_score:3d}[/]  {listing.title} @ {listing.company}")
            except Exception as exc:
                console.print(f"  [red]error scoring {listing.id}:[/] {exc}")

        console.print(
            f"\n[bold]Done.[/] {scored}/{len(new_listings)} scored. "
            f"Run [bold]find-a-job list --min-fit {cfg.ai.fit_threshold}[/] to see top matches."
        )


# ---------------------------------------------------------------------------
# shortlist
# ---------------------------------------------------------------------------

@app.command()
def shortlist(
    config: Optional[str] = _CONFIG_OPT,
    min_fit: int = typer.Option(None, "--min-fit", "-f", help="Override fit threshold from config"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show candidates without pushing to Notion"),
) -> None:
    """Push roles above the fit threshold to your Notion board."""
    cfg = _load(config)
    threshold = min_fit if min_fit is not None else cfg.ai.fit_threshold

    with Store(cfg.store.db_path) as store:
        candidates = store.shortlist_candidates(threshold)

        if not candidates:
            console.print(f"[dim]No new candidates above fit {threshold}. Try a lower --min-fit.[/]")
            raise typer.Exit()

        console.print(f"[bold]{len(candidates)}[/] candidate(s) above fit {threshold}:\n")
        for r in candidates:
            console.print(f"  [green]{r['fit_score']:3d}[/]  {r['title']} @ {r['company']}")

        if dry_run:
            console.print("\n[dim]Dry run — nothing pushed to Notion.[/]")
            return

        console.print()
        if not typer.confirm(f"Push {len(candidates)} listing(s) to Notion?"):
            console.print("[dim]Aborted.[/]")
            raise typer.Exit()

        console.print("\nPushing to Notion…\n")
        ns = _notion(cfg)
        pushed = 0
        failed = 0

        for r in candidates:
            listing = store.get_listing(r["id"])
            score = store.get_best_score(r["id"])
            if not listing or not score:
                continue
            try:
                page_id = ns.push_listing(listing, score)
                from .models import Application
                app_row = Application(listing_id=listing.id, notion_page_id=page_id)
                store.upsert_application(app_row)
                pushed += 1
                console.print(f"  [green]✓[/] {listing.title} @ {listing.company}")
            except Exception as exc:
                failed += 1
                console.print(f"  [red]✗[/] {listing.title}: {exc}")

        console.print(f"\n[bold]Done.[/] {pushed} pushed to Notion" + (f", {failed} failed" if failed else "") + ".")


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------

@app.command()
def apply(
    listing_id: int = typer.Argument(..., help="Listing ID (from find-a-job list)"),
    chase_days: int = typer.Option(7, "--chase-days", help="Days until follow-up reminder"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Mark a role as applied and set a chase date."""
    cfg = _load(config)
    now = datetime.now(UTC)
    chase_at = now + timedelta(days=chase_days)

    with Store(cfg.store.db_path) as store:
        listing = store.get_listing(listing_id)
        if not listing:
            console.print(f"[red]No listing with ID {listing_id}.[/]")
            raise typer.Exit(1)

        existing = store.get_application(listing_id)
        app_row = existing or __import__("findajob.models", fromlist=["Application"]).Application(
            listing_id=listing_id
        )
        app_row = app_row.model_copy(update={
            "status": "applied",
            "applied_at": now,
            "chase_at": chase_at,
        })
        saved = store.upsert_application(app_row)

        if saved.notion_page_id:
            try:
                ns = _notion(cfg)
                ns.update_status(
                    saved.notion_page_id,
                    status="applied",
                    applied_at=now,
                    chase_at=chase_at,
                )
                console.print("[dim]Notion updated.[/]")
            except Exception as exc:
                console.print(f"[yellow]Notion update failed:[/] {exc}")

    console.print(
        f"[green]Applied:[/] {listing.title} @ {listing.company}\n"
        f"Chase reminder set for [bold]{chase_at.date()}[/]"
    )


# ---------------------------------------------------------------------------
# list / chase
# ---------------------------------------------------------------------------

@app.command(name="list")
def list_jobs(
    min_fit: int = typer.Option(0, "--min-fit", "-f", help="Minimum fit score to show"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum rows to show"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Show ranked job matches from the local database."""
    cfg = _load(config)
    with Store(cfg.store.db_path) as store:
        rows = store.list_listings(min_fit=min_fit or None, limit=limit)

    if not rows:
        console.print("[dim]No listings found. Run [bold]find-a-job scan[/] first.[/]")
        raise typer.Exit()

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("ID", style="dim", width=5)
    table.add_column("Fit", justify="right", width=4)
    table.add_column("Title", min_width=30)
    table.add_column("Company", min_width=20)
    table.add_column("Location")
    table.add_column("Source", style="dim")

    for r in rows:
        fit = str(r["fit_score"]) if r.get("fit_score") is not None else "—"
        table.add_row(
            str(r["id"]),
            fit,
            r["title"],
            r["company"],
            r.get("location") or "",
            r["source_name"],
        )

    console.print(table)
    console.print(f"\n[dim]{len(rows)} listing(s) shown[/]")


@app.command()
def rescore(
    listing_ids: list[int] = typer.Argument(..., help="Listing IDs to rescore (from find-a-job list)"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Re-score specific listings with the current profile, fetching full JD from the listing URL."""
    import httpx
    from bs4 import BeautifulSoup
    from .sources.http_source import _extract_jd
    from .profile import build_profile
    from .scoring import BulkScorer

    cfg = _load(config)

    if not Path("candidate_profile.json").exists():
        console.print("[red]No candidate_profile.json — run [bold]find-a-job profile[/] first.[/]")
        raise typer.Exit(1)

    profile_obj = build_profile(cfg)
    scorer = BulkScorer(cfg, profile_obj)

    ns = None
    if cfg.notion.token and cfg.notion.database_id:
        from .notion_sync import NotionSync
        ns = NotionSync(token=cfg.notion.token, database_id=cfg.notion.database_id)

    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    with Store(cfg.store.db_path) as store:
        for lid in listing_ids:
            listing = store.get_listing(lid)
            if not listing:
                console.print(f"[red]No listing {lid}[/]")
                continue

            # Fetch full JD from individual page
            enriched_desc = listing.description
            try:
                resp = httpx.get(listing.url, headers={"User-Agent": ua}, follow_redirects=True, timeout=20)
                soup = BeautifulSoup(resp.text, "html.parser")
                jd = _extract_jd(soup)
                if jd and len(jd) > len(enriched_desc or ""):
                    enriched_desc = jd
            except Exception as exc:
                console.print(f"  [yellow]Could not fetch {listing.url}: {exc}[/]")

            listing_with_jd = listing.model_copy(update={"description": enriched_desc})

            # Delete old scores
            store.conn.execute("DELETE FROM scores WHERE listing_id = ?", (lid,))
            store.conn.commit()

            # Re-score
            try:
                score = scorer.score(listing_with_jd)
                store.insert_score(score)
                color = "green" if score.fit_score >= cfg.ai.fit_threshold else (
                    "yellow" if score.fit_score >= 50 else "red"
                )
                console.print(
                    f"  [{color}]{score.fit_score:3d}[/]  {listing.title} @ {listing.company}"
                )
                if score.rationale:
                    console.print(f"       [dim]{score.rationale}[/]")
            except Exception as exc:
                console.print(f"  [red]error scoring {lid}:[/] {exc}")
                continue

            # Update Notion page if linked
            app_row = store.get_application(lid)
            if app_row and app_row.notion_page_id and ns:
                try:
                    from typing import Any
                    props: dict[str, Any] = {
                        "Fit": {"number": score.fit_score},
                    }
                    if score.flags:
                        props["Flags"] = {"rich_text": [{"text": {"content": ", ".join(score.flags)[:2000]}}]}
                    ns._patch(f"/pages/{app_row.notion_page_id}", {"properties": props})
                    console.print(f"       [dim]Notion updated.[/]")
                except Exception as exc:
                    console.print(f"       [yellow]Notion update failed: {exc}[/]")


@app.command()
def dismiss(
    listing_ids: list[int] = typer.Argument(..., help="Listing IDs to dismiss (from find-a-job list)"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Mark listings as not interested — hides them from future shortlists."""
    from .models import Application

    cfg = _load(config)
    ns = None
    if cfg.notion.token and cfg.notion.database_id:
        from .notion_sync import NotionSync
        ns = NotionSync(token=cfg.notion.token, database_id=cfg.notion.database_id)

    with Store(cfg.store.db_path) as store:
        for lid in listing_ids:
            listing = store.get_listing(lid)
            if not listing:
                console.print(f"[red]No listing {lid}[/]")
                continue

            existing = store.get_application(lid)
            app_row = (existing or Application(listing_id=lid)).model_copy(
                update={"status": "not_interested"}
            )
            store.upsert_application(app_row)

            if ns and app_row.notion_page_id:
                try:
                    ns.archive_page(app_row.notion_page_id)
                    console.print(f"  [dim]✓ archived in Notion[/]")
                except Exception as exc:
                    console.print(f"  [yellow]Notion archive failed:[/] {exc}")

            console.print(f"  [dim]✗[/] {listing.title} @ {listing.company} — dismissed")


@app.command()
def watch(
    interval: int = typer.Option(60, "--interval", "-i", help="Poll interval in seconds"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Poll the Notion board for pending actions and execute them."""
    import time
    import httpx as _httpx
    from bs4 import BeautifulSoup as _BS
    from .models import Application
    from .notion_sync import NotionSync
    from .profile import build_profile
    from .scoring import BulkScorer
    from .sources.http_source import _extract_jd

    cfg = _load(config)
    ns = _notion(cfg)

    console.print(f"[bold]Watching Notion board[/] (every {interval}s) — Ctrl+C to stop\n")

    # Patch schema to ensure "Not Interested" option exists
    try:
        ns.update_schema()
    except Exception:
        pass

    _ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    while True:
        try:
            pending = ns.get_pending_actions()
        except Exception as exc:
            console.print(f"[yellow]Poll error:[/] {exc}")
            time.sleep(interval)
            continue

        if pending:
            console.print(f"[bold]{len(pending)} pending action(s)[/]")

        with Store(cfg.store.db_path) as store:
            for item in pending:
                page_id = item["page_id"]
                lid = item["listing_id"]
                action = item["action"]
                listing = store.get_listing(lid)
                if not listing:
                    ns.reset_action(page_id)
                    continue

                console.print(f"  [{action}] {listing.title} @ {listing.company}")

                try:
                    if action == "Not Interested":
                        existing = store.get_application(lid)
                        app_row = (existing or Application(listing_id=lid)).model_copy(
                            update={"status": "not_interested"}
                        )
                        store.upsert_application(app_row)
                        ns.reset_action(page_id)
                        ns.archive_page(page_id)
                        console.print(f"    [dim]dismissed + archived[/]")

                    elif action == "Rescore":
                        profile_obj = build_profile(cfg)
                        scorer = BulkScorer(cfg, profile_obj)
                        try:
                            resp = _httpx.get(listing.url, headers={"User-Agent": _ua}, follow_redirects=True, timeout=20)
                            jd = _extract_jd(_BS(resp.text, "html.parser"))
                            if jd and len(jd) > len(listing.description or ""):
                                listing = listing.model_copy(update={"description": jd})
                        except Exception:
                            pass
                        store.conn.execute("DELETE FROM scores WHERE listing_id = ?", (lid,))
                        store.conn.commit()
                        score = scorer.score(listing)
                        store.insert_score(score)
                        ns._patch(f"/pages/{page_id}", {"properties": {"Fit": {"number": score.fit_score}}})
                        ns.reset_action(page_id)
                        console.print(f"    [dim]rescored → {score.fit_score}[/]")

                    else:
                        # Unknown action — clear it
                        ns.reset_action(page_id)

                except Exception as exc:
                    console.print(f"    [red]error:[/] {exc}")
                    ns.reset_action(page_id)

        time.sleep(interval)


@app.command()
def chase(
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Show applications that are due for follow-up today."""
    cfg = _load(config)
    with Store(cfg.store.db_path) as store:
        rows = store.due_for_chase()

    if not rows:
        console.print("[green]Nothing to chase today.[/]")
        raise typer.Exit()

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("ID", style="dim", width=5)
    table.add_column("Title", min_width=30)
    table.add_column("Company", min_width=20)
    table.add_column("Status")
    table.add_column("Chase by")

    for r in rows:
        table.add_row(
            str(r["id"]),
            r["title"],
            r["company"],
            r["status"],
            str(r["chase_at"])[:10] if r.get("chase_at") else "—",
        )

    console.print(table)
