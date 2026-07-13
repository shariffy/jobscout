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

app = typer.Typer(name="jobscout", help="Personal job-board analyser and tracker.")
console = Console()

_CONFIG_OPT = typer.Option(None, "--config", "-c", help="Path to config.toml")


def _load(config: str | None):
    return load_config(config)


def _verdict(score, cfg) -> str:
    """One-line rich markup for a fresh score: APPLY/NO + tier for gated
    assessments, the plain fit number for legacy additive ones."""
    if score.tier_label or score.gate_results:  # produced by the gated scorer
        if score.decision == "apply":
            tier = f" {score.tier_label}" if score.tier_label not in ("", "none") else ""
            return f"[green]APPLY{tier}[/] [dim](fit {score.fit_score})[/]"
        failed = ", ".join(
            f.removeprefix("gate-fail-") for f in score.flags if f.startswith("gate-fail-")
        )
        return f"[red]NO[/] ({failed})" if failed else "[red]NO[/]"
    color = "green" if score.fit_score >= cfg.ai.fit_threshold else (
        "yellow" if score.fit_score >= 50 else "red"
    )
    return f"[{color}]{score.fit_score:3d}[/]"


def _notion(cfg) -> "NotionSync":
    from .notion_sync import NotionSync
    if not cfg.notion.token:
        console.print("[red]No Notion token in config.toml.[/]")
        raise typer.Exit(1)
    if not cfg.notion.database_id:
        console.print(
            "[red]No Notion database_id in config.toml.[/]\n"
            "Run [bold]jobscout init --notion-parent <page-url>[/] to create the board."
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

        from .assess import build_scorer
        from .profile import build_profile

        if not Path("candidate_profile.json").exists():
            console.print("[yellow]No candidate_profile.json — run [bold]jobscout profile[/] first.[/]")
            return

        profile_obj = build_profile(cfg)
        scorer = build_scorer(cfg, profile_obj, store)

        console.print(f"\n[bold]Scoring[/] {len(new_listings)} new listing(s) with {cfg.ai.bulk_model}…\n")

        scored = 0
        for listing in new_listings:
            try:
                score = scorer.score(listing)
                store.insert_score(score)
                scored += 1
                console.print(f"  {_verdict(score, cfg)}  {listing.title} @ {listing.company}")
            except Exception as exc:
                console.print(f"  [red]error scoring {listing.id}:[/] {exc}")

        console.print(
            f"\n[bold]Done.[/] {scored}/{len(new_listings)} scored. "
            f"Run [bold]jobscout list --min-fit {cfg.ai.fit_threshold}[/] to see top matches."
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

def _apply_to_store(store, listing_id: int, chase_days: int = 7):
    """Mark a listing as applied in the local store and set a chase date.

    Returns (listing, saved_application, chase_at), or (None, None, None) if the
    listing does not exist. Notion syncing is left to the caller.
    """
    from .models import Application

    listing = store.get_listing(listing_id)
    if not listing:
        return None, None, None

    now = datetime.now(UTC)
    chase_at = now + timedelta(days=chase_days)
    existing = store.get_application(listing_id)
    app_row = (existing or Application(listing_id=listing_id)).model_copy(update={
        "status": "applied",
        "applied_at": now,
        "chase_at": chase_at,
    })
    saved = store.upsert_application(app_row)
    return listing, saved, chase_at


@app.command(name="mark-applied")
def mark_applied(
    listing_id: int = typer.Argument(..., help="Listing ID (from jobscout list)"),
    chase_days: int = typer.Option(7, "--chase-days", help="Days until follow-up reminder"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Mark a role as applied and set a chase date."""
    cfg = _load(config)

    with Store(cfg.store.db_path) as store:
        listing, saved, chase_at = _apply_to_store(store, listing_id, chase_days)
        if not listing:
            console.print(f"[red]No listing with ID {listing_id}.[/]")
            raise typer.Exit(1)

        if saved.notion_page_id:
            try:
                ns = _notion(cfg)
                ns.update_status(
                    saved.notion_page_id,
                    status="applied",
                    applied_at=saved.applied_at,
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
# prep
# ---------------------------------------------------------------------------

@app.command()
def prep(
    listing_id: int = typer.Argument(..., help="Listing ID (from jobscout list)"),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Generate an application prep brief for a role and append it to its Notion page."""
    import httpx
    from bs4 import BeautifulSoup
    from .prep import generate_prep
    from .profile import build_profile
    from .sources.http_source import _extract_jd

    cfg = _load(config)

    with Store(cfg.store.db_path) as store:
        listing = store.get_listing(listing_id)
        if not listing:
            console.print(f"[red]No listing {listing_id}[/]")
            raise typer.Exit(1)
        score = store.get_best_score(listing_id)
        app_row = store.get_application(listing_id)

    # Enrich description from live URL
    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    try:
        resp = httpx.get(listing.url, headers={"User-Agent": ua}, follow_redirects=True, timeout=20)
        jd = _extract_jd(BeautifulSoup(resp.text, "html.parser"))
        if jd and len(jd) > len(listing.description or ""):
            listing = listing.model_copy(update={"description": jd})
    except Exception as exc:
        console.print(f"[yellow]Could not fetch full JD: {exc}[/]")

    profile_obj = build_profile(cfg)

    console.print(f"[bold]Generating prep brief for[/] {listing.title} @ {listing.company}…\n")
    content = generate_prep(cfg, listing, profile_obj, score)
    console.print(content)

    if app_row and app_row.notion_page_id:
        try:
            ns = _notion(cfg)
            ns.append_prep_content(app_row.notion_page_id, content)
            console.print("\n[dim]Appended to Notion page.[/]")
        except Exception as exc:
            console.print(f"\n[yellow]Notion append failed: {exc}[/]")


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
        console.print("[dim]No listings found. Run [bold]jobscout scan[/] first.[/]")
        raise typer.Exit()

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("ID", style="dim", width=5)
    table.add_column("Fit", justify="right", width=4)
    table.add_column("Decision", width=9)
    table.add_column("Tier", width=4)
    table.add_column("Title", min_width=30)
    table.add_column("Company", min_width=20)
    table.add_column("Location")
    table.add_column("Source", style="dim")

    for r in rows:
        fit = str(r["fit_score"]) if r.get("fit_score") is not None else "—"
        decision = r.get("decision") or "—"
        if decision == "apply":
            decision = "[green]apply[/]"
        elif decision == "no":
            decision = "[red]no[/]"
        table.add_row(
            str(r["id"]),
            fit,
            decision,
            r.get("tier_label") or "—",
            r["title"],
            r["company"],
            r.get("location") or "",
            r["source_name"],
        )

    console.print(table)
    console.print(f"\n[dim]{len(rows)} listing(s) shown[/]")


@app.command()
def rescore(
    listing_ids: list[int] = typer.Argument(default=None, help="Listing IDs to rescore (from jobscout list)"),
    all_listings: bool = typer.Option(False, "--all", help="Rescore every listing in the database"),
    min_fit: int = typer.Option(0, "--min-fit", "-f", help="With --all, only rescore listings scoring at or above this"),
    force: bool = typer.Option(
        False, "--force",
        help="With --all, also rescore listings already scored under the current config",
    ),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Re-score listings with the current profile, fetching full JD from the listing URL."""
    import httpx
    from bs4 import BeautifulSoup
    from .assess import build_scorer
    from .config import assessment_config_hash
    from .profile import build_profile
    from .sources.http_source import _extract_jd

    cfg = _load(config)

    if not all_listings and not listing_ids:
        console.print("[red]Provide listing IDs or pass --all.[/]")
        raise typer.Exit(1)

    if not Path("candidate_profile.json").exists():
        console.print("[red]No candidate_profile.json — run [bold]jobscout profile[/] first.[/]")
        raise typer.Exit(1)

    profile_obj = build_profile(cfg)
    current_version = assessment_config_hash(cfg)

    ns = None
    if cfg.notion.token and cfg.notion.database_id:
        from .notion_sync import NotionSync
        ns = NotionSync(token=cfg.notion.token, database_id=cfg.notion.database_id)

    ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

    with Store(cfg.store.db_path) as store:
        # Passing store lets the gated scorer serve/save Stage-1 extractions from
        # the extractions cache instead of re-calling the LLM for a listing whose
        # model/prompt/reasoning haven't changed since the last rescore.
        scorer = build_scorer(cfg, profile_obj, store)

        if all_listings:
            sql = """
                SELECT l.id FROM listings l
                LEFT JOIN applications a ON a.listing_id = l.id
                LEFT JOIN scores s ON s.listing_id = l.id
                WHERE (a.status IS NULL OR a.status != 'not_interested')
            """
            params: list = []
            if min_fit:
                sql += " AND s.fit_score >= ?"
                params.append(min_fit)
            if not force:
                # Skip listings already scored under this exact config — the common
                # case of re-running rescore --all with nothing changed is then free.
                sql += " AND (s.assessment_version IS NULL OR s.assessment_version != ?)"
                params.append(current_version)
            sql += " ORDER BY l.id"
            listing_ids = [r[0] for r in store.conn.execute(sql, params).fetchall()]
            console.print(f"[bold]Rescoring {len(listing_ids)} listings…[/]\n")
            if not listing_ids:
                console.print(
                    "[dim]Nothing to do — every listing already matches the current config. "
                    "Use --force to rescore anyway.[/]"
                )
                return

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

            # Persist the enriched JD so a future rescore (gate tweak, model swap)
            # skips this HTTP fetch and scores from the richer text already on file.
            if enriched_desc != listing.description:
                store.update_listing_description(lid, enriched_desc)
            listing_with_jd = listing.model_copy(update={"description": enriched_desc})

            # Score first, then replace — so a crash mid-flight leaves the old score intact
            try:
                score = scorer.score(listing_with_jd)
                store.insert_score(score)
                console.print(f"  {_verdict(score, cfg)}  {listing.title} @ {listing.company}")
                if score.rationale:
                    console.print(f"       [dim]{score.rationale}[/]")
            except Exception as exc:
                console.print(f"  [red]error scoring {lid}:[/] {exc}")
                continue

            # Update Notion page if linked and not dismissed
            app_row = store.get_application(lid)
            if app_row and app_row.notion_page_id and ns and app_row.status != "not_interested":
                try:
                    ns.update_score(app_row.notion_page_id, score)
                    console.print(f"       [dim]Notion updated.[/]")
                except Exception as exc:
                    console.print(f"       [yellow]Notion update failed: {exc}[/]")


@app.command()
def dismiss(
    listing_ids: list[int] = typer.Argument(..., help="Listing IDs to dismiss (from jobscout list)"),
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
    from .assess import build_scorer
    from .models import Application
    from .notion_sync import NotionSync
    from .prep import generate_prep
    from .profile import build_profile
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
                        update = {"status": "not_interested"}
                        if item.get("notes"):
                            update["notes"] = item["notes"]
                        app_row = (existing or Application(listing_id=lid)).model_copy(update=update)
                        store.upsert_application(app_row)
                        ns.reset_action(page_id)
                        ns.archive_page(page_id)
                        console.print(f"    [dim]dismissed + archived[/]")

                    elif action == "Rescore":
                        profile_obj = build_profile(cfg)
                        scorer = build_scorer(cfg, profile_obj, store)
                        try:
                            resp = _httpx.get(listing.url, headers={"User-Agent": _ua}, follow_redirects=True, timeout=20)
                            jd = _extract_jd(_BS(resp.text, "html.parser"))
                            if jd and len(jd) > len(listing.description or ""):
                                store.update_listing_description(lid, jd)
                                listing = listing.model_copy(update={"description": jd})
                        except Exception:
                            pass
                        score = scorer.score(listing)
                        store.insert_score(score)
                        ns.update_score(page_id, score)
                        ns.reset_action(page_id)
                        console.print(f"    [dim]rescored → {_verdict(score, cfg)}[/]")

                    elif action == "Prep":
                        profile_obj = build_profile(cfg)
                        score = store.get_best_score(lid)
                        try:
                            resp = _httpx.get(listing.url, headers={"User-Agent": _ua}, follow_redirects=True, timeout=20)
                            jd = _extract_jd(_BS(resp.text, "html.parser"))
                            if jd and len(jd) > len(listing.description or ""):
                                listing = listing.model_copy(update={"description": jd})
                        except Exception:
                            pass
                        content = generate_prep(cfg, listing, profile_obj, score)
                        ns.append_prep_content(page_id, content)
                        # Mark as prepping so the role stays visible on the board
                        existing = store.get_application(lid)
                        app_row = (existing or Application(listing_id=lid)).model_copy(update={"status": "prepping"})
                        store.upsert_application(app_row)
                        ns._patch(f"/pages/{page_id}", {"properties": {"Status": {"select": {"name": "prepping"}}}})
                        ns.reset_action(page_id)
                        console.print(f"    [dim]prep brief appended → status: prepping[/]")

                    elif action == "Mark as Applied":
                        _, saved, chase_at = _apply_to_store(store, lid)
                        ns.update_status(
                            page_id,
                            status="applied",
                            applied_at=saved.applied_at,
                            chase_at=chase_at,
                        )
                        ns.reset_action(page_id)
                        console.print(f"    [dim]marked applied → chase {chase_at.date()}[/]")

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


@app.command()
def prune(
    min_fit: int = typer.Option(70, "--min-fit", help="Archive Notion pages scoring below this threshold."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be archived without doing it."),
    config: Optional[str] = _CONFIG_OPT,
) -> None:
    """Archive Notion pages for listings that scored below the threshold (or were never scored)."""
    cfg = _load(config)
    if not cfg.notion.token or not cfg.notion.database_id:
        console.print("[red]Notion not configured.[/]")
        raise typer.Exit(1)

    from .models import Application
    from .notion_sync import NotionSync
    ns = NotionSync(token=cfg.notion.token, database_id=cfg.notion.database_id)

    with Store(cfg.store.db_path) as store:
        # Find Notion-linked applications where current score < threshold or no score
        rows = store.conn.execute("""
            SELECT a.listing_id, a.notion_page_id, l.title, l.company,
                   COALESCE(s.fit_score, -1) AS fit_score, a.status
            FROM applications a
            JOIN listings l ON l.id = a.listing_id
            LEFT JOIN scores s ON s.listing_id = a.listing_id
            WHERE a.notion_page_id IS NOT NULL
              AND (a.status IS NULL OR a.status NOT IN ('applied', 'interviewing', 'offer'))
              AND (s.fit_score IS NULL OR s.fit_score < ?)
        """, (min_fit,)).fetchall()

        if not rows:
            console.print(f"[green]Nothing to prune (all Notion pages score ≥ {min_fit}).[/]")
            return

        console.print(f"[bold]{'Would archive' if dry_run else 'Archiving'} {len(rows)} Notion pages below {min_fit}:[/]\n")
        archived = 0
        for row in rows:
            lid, page_id, title, company, fit, status = row
            label = f"  {fit:3d}  {title[:45]} @ {company}"
            if dry_run:
                console.print(f"[dim]{label}[/]")
            else:
                try:
                    ns.archive_page(page_id)
                    # Mirror the archive locally so it doesn't come back as a stale
                    # "shortlisted" row on the next rescore/shortlist pass — a page
                    # archived here without this used to desync silently forever.
                    existing = store.get_application(lid)
                    app_row = (existing or Application(listing_id=lid)).model_copy(
                        update={"status": "not_interested"}
                    )
                    store.upsert_application(app_row)
                    console.print(f"[dim]{label}[/]")
                    archived += 1
                except Exception as exc:
                    console.print(f"[yellow]  skip {lid}: {exc}[/]")

        if not dry_run:
            console.print(f"\n[green]Archived {archived} pages.[/]")
