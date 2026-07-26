from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .store import Store, _chmod_owner_only

app = typer.Typer(name="jobscout", help="Personal job-board analyser and tracker.")
console = Console()

_CONFIG_OPT = typer.Option(None, "--config", "-c", help="Path to config.toml")


def _load(config: str | None):
    return load_config(config)


def _verdict(score) -> str:
    """One-line rich markup: APPLY + tier, or NO (+ failed gates)."""
    if score.decision == "apply":
        tier = f" {score.tier_label}" if score.tier_label not in ("", "none") else ""
        pri = f" [dim]p{score.priority}[/]" if score.priority is not None else ""
        return f"[green]APPLY{tier}[/]{pri}"
    failed = ", ".join(
        f.removeprefix("gate-fail-") for f in score.flags if f.startswith("gate-fail-")
    )
    return f"[red]NO[/] ({failed})" if failed else "[red]NO[/]"


def _notion(cfg):
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
    config: str | None = _CONFIG_OPT,
    notion_parent: str | None = typer.Option(
        None, "--notion-parent", help="Notion page URL or ID to create the pipeline database under"
    ),
) -> None:
    """Scaffold config.toml and initialise the SQLite database. Optionally create the Notion board."""
    config_path = Path(config) if config else Path("config.toml")

    if not config_path.exists():
        example = Path(__file__).parent.parent / "config.example.toml"
        if example.exists():
            shutil.copy(example, config_path)
            _chmod_owner_only(config_path)
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
    config: str | None = _CONFIG_OPT,
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
    config: str | None = _CONFIG_OPT,
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

        from .parallel import map_bounded

        scored = 0

        def _on_scored(done, total, listing, score, error):
            nonlocal scored
            if error is not None:
                console.print(f"  [red]error scoring {listing.id}:[/] {error}")
                return
            # Runs on the calling thread (as each score completes), so the store
            # write and console print are serialised and safe.
            store.insert_score(score)
            scored += 1
            console.print(f"  {_verdict(score)}  {listing.title} @ {listing.company}")

        map_bounded(new_listings, scorer.score, _on_scored,
                    max_workers=cfg.ai.scorer_concurrency)

        console.print(
            f"\n[bold]Done.[/] {scored}/{len(new_listings)} scored. "
            f"Run [bold]jobscout list --apply[/] to see top matches."
        )


# ---------------------------------------------------------------------------
# shortlist
# ---------------------------------------------------------------------------

@app.command()
def shortlist(
    config: str | None = _CONFIG_OPT,
    dry_run: bool = typer.Option(False, "--dry-run", help="Show candidates without pushing to Notion"),
) -> None:
    """Push apply-decision roles to your Notion board."""
    cfg = _load(config)

    with Store(cfg.store.db_path) as store:
        candidates = store.shortlist_candidates()

        if not candidates:
            console.print("[dim]No new apply candidates. Run scan/rescore first.[/]")
            raise typer.Exit()

        console.print(f"[bold]{len(candidates)}[/] apply candidate(s):\n")
        for r in candidates:
            tier = r.get("tier_label") or "—"
            pri = r.get("priority")
            pri_s = f"p{pri}" if pri is not None else "p—"
            console.print(f"  [green]{tier:4}[/] {pri_s:>5}  {r['title']} @ {r['company']}")

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
# prep
# ---------------------------------------------------------------------------

@app.command()
def prep(
    listing_id: int = typer.Argument(..., help="Listing ID (from jobscout list)"),
    config: str | None = _CONFIG_OPT,
) -> None:
    """Generate an application prep brief for a role and append it to its Notion page."""
    from bs4 import BeautifulSoup

    from .prep import generate_prep
    from .profile import build_profile
    from .sources.http_source import _extract_jd, safe_get

    cfg = _load(config)

    with Store(cfg.store.db_path) as store:
        listing = store.get_listing(listing_id)
        if not listing:
            console.print(f"[red]No listing {listing_id}[/]")
            raise typer.Exit(1)
        score = store.get_best_score(listing_id)
        app_row = store.get_application(listing_id)

    # Enrich description from live URL
    try:
        jd = _extract_jd(BeautifulSoup(safe_get(listing.url), "html.parser"))
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
    apply_only: bool = typer.Option(False, "--apply", help="Show only apply-decision listings"),
    limit: int = typer.Option(50, "--limit", "-n", help="Maximum rows to show"),
    config: str | None = _CONFIG_OPT,
) -> None:
    """Show ranked job matches from the local database."""
    cfg = _load(config)
    with Store(cfg.store.db_path) as store:
        rows = store.list_listings(apply_only=apply_only, limit=limit)

    if not rows:
        console.print("[dim]No listings found. Run [bold]jobscout scan[/] first.[/]")
        raise typer.Exit()

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("ID", style="dim", width=5)
    table.add_column("Pri", justify="right", width=5)
    table.add_column("Decision", width=9)
    table.add_column("Tier", width=4)
    table.add_column("Title", min_width=30)
    table.add_column("Company", min_width=20)
    table.add_column("Location")
    table.add_column("Source", style="dim")

    for r in rows:
        pri = str(r["priority"]) if r.get("priority") is not None else "—"
        decision = r.get("decision") or "—"
        if decision == "apply":
            decision = "[green]apply[/]"
        elif decision == "no":
            decision = "[red]no[/]"
        table.add_row(
            str(r["id"]),
            pri,
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
    apply_only: bool = typer.Option(False, "--apply", help="With --all, only rescore apply-decision listings"),
    force: bool = typer.Option(
        False, "--force",
        help="With --all, also rescore listings already scored under the current config",
    ),
    config: str | None = _CONFIG_OPT,
) -> None:
    """Re-score listings with the current profile, fetching full JD from the listing URL."""
    from bs4 import BeautifulSoup

    from .assess import build_scorer
    from .config import assessment_config_hash
    from .profile import build_profile
    from .sources.http_source import _extract_jd, safe_get

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
            if apply_only:
                sql += " AND s.decision = 'apply'"
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

        listings = []
        for lid in listing_ids:
            listing = store.get_listing(lid)
            if not listing:
                console.print(f"[red]No listing {lid}[/]")
                continue
            listings.append(listing)

        def _rescore_work(listing):
            # Worker thread: only the thread-safe network work (JD fetch + score).
            # The fetch failure is returned, not printed, so the console stays on the
            # main thread. Persisting the JD and the score happens in _on_rescored.
            enriched_desc = listing.description
            fetch_err = None
            try:
                jd = _extract_jd(BeautifulSoup(safe_get(listing.url), "html.parser"))
                if jd and len(jd) > len(enriched_desc or ""):
                    enriched_desc = jd
            except Exception as exc:
                fetch_err = exc
            listing_with_jd = listing.model_copy(update={"description": enriched_desc})
            score = scorer.score(listing_with_jd)
            return {"enriched_desc": enriched_desc, "score": score, "fetch_err": fetch_err}

        def _on_rescored(done, total, listing, result, error):
            # Calling thread: store writes, console, and Notion are all safe here.
            if error is not None:
                console.print(f"  [red]error scoring {listing.id}:[/] {error}")
                return
            if result["fetch_err"] is not None:
                console.print(f"  [yellow]Could not fetch {listing.url}: {result['fetch_err']}[/]")
            # Persist the enriched JD so a future rescore skips the HTTP fetch.
            if result["enriched_desc"] != listing.description:
                store.update_listing_description(listing.id, result["enriched_desc"])
            score = result["score"]
            store.insert_score(score)
            console.print(f"  {_verdict(score)}  {listing.title} @ {listing.company}")
            if score.rationale:
                console.print(f"       [dim]{score.rationale}[/]")
            app_row = store.get_application(listing.id)
            if app_row and app_row.notion_page_id and ns and app_row.status != "not_interested":
                try:
                    ns.update_score(app_row.notion_page_id, score)
                    console.print("       [dim]Notion updated.[/]")
                except Exception as exc:
                    console.print(f"       [yellow]Notion update failed: {exc}[/]")

        from .parallel import map_bounded

        map_bounded(listings, _rescore_work, _on_rescored,
                    max_workers=cfg.ai.scorer_concurrency)


_VALID_STATUSES = [
    "shortlisted", "prepping", "applied", "interviewing",
    "offer", "rejected", "withdrawn", "not_interested",
]
# Statuses that also archive the Notion page once synced — not_interested is
# pre-application noise; rejected/withdrawn are terminal outcomes you've asked
# to have off the active board too. Anything else (applied, interviewing,
# offer, prepping, shortlisted) stays visible.
_ARCHIVE_ON_STATUS = {"not_interested", "rejected", "withdrawn"}


def _apply_status(
    store, ns, lid: int, status: str, notes: str = "", chase_days: int = 7,
    notion_page_id: str | None = None,
):
    """Set a listing's status locally, then best-effort mirror it to Notion —
    computing chase_at on a transition into "applied" and archiving on a
    terminal status (see _ARCHIVE_ON_STATUS). Shared by the `set-status`
    command and watch's board-drift sync so a manual CLI change and a manual
    Notion edit end up in exactly the same state.

    notion_page_id lets a caller that already knows the page (board-drift
    sync) link it even if no local Application row existed yet — otherwise
    the Notion push is skipped for lack of anywhere to send it.

    Returns (listing, saved_application, notion_error); listing is None if
    the listing_id doesn't exist locally. notion_error is None on success or
    when there's no linked Notion page to update.
    """
    from .models import Application

    listing = store.get_listing(lid)
    if not listing:
        return None, None, None

    existing = store.get_application(lid)
    update: dict = {"status": status}
    if notes:
        update["notes"] = notes
    if notion_page_id and not (existing and existing.notion_page_id):
        update["notion_page_id"] = notion_page_id

    chase_at = None
    if status == "applied":
        now = datetime.now(UTC)
        chase_at = now + timedelta(days=chase_days)
        update["applied_at"] = now
        update["chase_at"] = chase_at

    app_row = (existing or Application(listing_id=lid)).model_copy(update=update)
    saved = store.upsert_application(app_row)

    notion_error = None
    if ns and saved.notion_page_id:
        try:
            ns.update_status(
                saved.notion_page_id, status=status,
                applied_at=saved.applied_at if status == "applied" else None,
                chase_at=chase_at, notes=notes,
            )
            if status in _ARCHIVE_ON_STATUS:
                ns.archive_page(saved.notion_page_id)
        except Exception as exc:
            notion_error = str(exc)

    return listing, saved, notion_error


@app.command(name="set-status")
def set_status(
    listing_ids: list[int] = typer.Argument(..., help="Listing IDs (from jobscout list)"),
    to: str = typer.Option(..., "--to", "-t", help=f"New status: {', '.join(_VALID_STATUSES)}"),
    notes: str = typer.Option("", "--notes", help="Optional note to attach"),
    chase_days: int = typer.Option(
        7, "--chase-days", help="Days until follow-up reminder (only used with --to applied)"
    ),
    config: str | None = _CONFIG_OPT,
) -> None:
    """Set one or more listings' status locally and in Notion.

    Archives the Notion page for a terminal status (not_interested, rejected,
    withdrawn); computes a chase-date reminder for --to applied. Replaces the
    old separate mark-applied / dismiss commands — both were just this with
    the target status hardcoded.
    """
    if to not in _VALID_STATUSES:
        console.print(f"[red]Unknown status {to!r}. Choose one of: {', '.join(_VALID_STATUSES)}[/]")
        raise typer.Exit(1)

    cfg = _load(config)
    ns = None
    if cfg.notion.token and cfg.notion.database_id:
        from .notion_sync import NotionSync
        ns = NotionSync(token=cfg.notion.token, database_id=cfg.notion.database_id)

    with Store(cfg.store.db_path) as store:
        for lid in listing_ids:
            listing, saved, notion_error = _apply_status(
                store, ns, lid, to, notes=notes, chase_days=chase_days
            )
            if not listing:
                console.print(f"[red]No listing {lid}[/]")
                continue

            suffix = f" — chase {saved.chase_at.date()}" if to == "applied" else ""
            console.print(f"  [dim]→ {to}[/]{suffix}  {listing.title} @ {listing.company}")
            if ns and saved.notion_page_id:
                if notion_error:
                    console.print(f"    [yellow]Notion update failed:[/] {notion_error}")
                else:
                    archived = " + archived" if to in _ARCHIVE_ON_STATUS else ""
                    console.print(f"    [dim]Notion updated{archived}.[/]")


def _sync_status_drift(store, ns, console, board_statuses: list[dict]) -> int:
    """Pull manual Status edits made directly on the Notion board into SQLite.

    Action is reserved for operations with no status equivalent (Rescore,
    Prep) — everything else, including future statuses, is just "change the
    dropdown" and gets picked up here via the same _apply_status the
    `set-status` command uses, instead of needing a matching Action.

    Prints exactly what changed for each listing (prior status -> new status,
    plus chase date / archive / carried-over notes / any Notion push failure).
    Returns the count synced.
    """
    drifted = []
    for item in board_statuses:
        app_row = store.get_application(item["listing_id"])
        prior_status = app_row.status if app_row else None
        if item["status"] != prior_status:
            drifted.append((item, prior_status))

    synced = 0
    for item, prior_status in drifted:
        notion_status = item["status"]
        listing, saved, notion_error = _apply_status(
            store, ns, item["listing_id"], notion_status, notes=item.get("notes", ""),
            notion_page_id=item["page_id"],
        )
        if not listing:
            continue

        synced += 1
        from_label = prior_status or "(unlinked)"
        console.print(
            f"  [status] {listing.title} @ {listing.company}: "
            f"[dim]{from_label}[/] → [bold]{notion_status}[/]"
        )
        detail = []
        if notion_status == "applied" and saved.chase_at:
            detail.append(f"chase {saved.chase_at.date()}")
        if notion_status in _ARCHIVE_ON_STATUS:
            detail.append("archived in Notion")
        if item.get("notes"):
            detail.append(f'notes: "{item["notes"]}"')
        if detail:
            console.print(f"    [dim]{' · '.join(detail)}[/]")
        if notion_error:
            console.print(f"    [yellow]Notion follow-up failed: {notion_error}[/]")
    return synced


@app.command()
def watch(
    interval: int = typer.Option(60, "--interval", "-i", help="Poll interval in seconds"),
    config: str | None = _CONFIG_OPT,
) -> None:
    """Poll the Notion board for Status changes and pending Actions, and sync them."""
    import time

    from bs4 import BeautifulSoup as _BS

    from .assess import build_scorer
    from .models import Application
    from .prep import generate_prep
    from .profile import build_profile
    from .sources.http_source import _extract_jd
    from .sources.http_source import safe_get as _safe_get

    cfg = _load(config)
    ns = _notion(cfg)

    console.print(f"[bold]Watching Notion board[/] (every {interval}s) — Ctrl+C to stop\n")

    # Patch schema to pick up any property/option changes since the board was created.
    try:
        ns.update_schema()
    except Exception:
        pass

    while True:
        try:
            board_statuses = ns.list_active_statuses()
        except Exception as exc:
            console.print(f"[yellow]Status poll error:[/] {exc}")
            board_statuses = []

        try:
            pending = ns.get_pending_actions()
        except Exception as exc:
            console.print(f"[yellow]Action poll error:[/] {exc}")
            pending = []

        with Store(cfg.store.db_path) as store:
            _sync_status_drift(store, ns, console, board_statuses)

            if pending:
                console.print(f"[bold]{len(pending)} pending action(s)[/]")

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
                    if action == "Rescore":
                        profile_obj = build_profile(cfg)
                        scorer = build_scorer(cfg, profile_obj, store)
                        try:
                            jd = _extract_jd(_BS(_safe_get(listing.url), "html.parser"))
                            if jd and len(jd) > len(listing.description or ""):
                                store.update_listing_description(lid, jd)
                                listing = listing.model_copy(update={"description": jd})
                        except Exception:
                            pass
                        score = scorer.score(listing)
                        store.insert_score(score)
                        ns.update_score(page_id, score)
                        ns.reset_action(page_id)
                        console.print(f"    [dim]rescored → {_verdict(score)}[/]")

                    elif action == "Prep":
                        profile_obj = build_profile(cfg)
                        score = store.get_best_score(lid)
                        try:
                            jd = _extract_jd(_BS(_safe_get(listing.url), "html.parser"))
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
                        console.print("    [dim]prep brief appended → status: prepping[/]")

                    else:
                        # Unknown action — clear it
                        ns.reset_action(page_id)

                except Exception as exc:
                    console.print(f"    [red]error:[/] {exc}")
                    ns.reset_action(page_id)

        time.sleep(interval)


@app.command()
def chase(
    config: str | None = _CONFIG_OPT,
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
    dry_run: bool = typer.Option(False, "--dry-run", help="Print what would be archived without doing it."),
    config: str | None = _CONFIG_OPT,
) -> None:
    """Archive Notion pages for listings that are not apply (or were never scored)."""
    cfg = _load(config)
    if not cfg.notion.token or not cfg.notion.database_id:
        console.print("[red]Notion not configured.[/]")
        raise typer.Exit(1)

    from .models import Application
    from .notion_sync import NotionSync
    ns = NotionSync(token=cfg.notion.token, database_id=cfg.notion.database_id)

    with Store(cfg.store.db_path) as store:
        rows = store.conn.execute("""
            SELECT a.listing_id, a.notion_page_id, l.title, l.company,
                   COALESCE(s.decision, '') AS decision, a.status
            FROM applications a
            JOIN listings l ON l.id = a.listing_id
            LEFT JOIN scores s ON s.listing_id = a.listing_id
            WHERE a.notion_page_id IS NOT NULL
              AND (a.status IS NULL OR a.status NOT IN ('applied', 'interviewing', 'offer'))
              AND (s.decision IS NULL OR s.decision != 'apply')
        """).fetchall()

        if not rows:
            console.print("[green]Nothing to prune (all Notion pages are apply).[/]")
            return

        console.print(f"[bold]{'Would archive' if dry_run else 'Archiving'} {len(rows)} non-apply Notion page(s):[/]\n")
        archived = 0
        for row in rows:
            lid, page_id, title, company, decision, status = row
            label = f"  {(decision or '—'):5}  {title[:45]} @ {company}"
            if dry_run:
                console.print(f"[dim]{label}[/]")
            else:
                try:
                    # Set Status before archiving so an unarchive later shows the
                    # right state instead of whatever it was before pruning.
                    ns.update_status(page_id, status="not_interested")
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
