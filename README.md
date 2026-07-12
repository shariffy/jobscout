# jobscout

An AI-assisted job-search pipeline. It scrapes job boards and feeds, scores each
listing against **your** CV and goals using Claude, pushes the strong matches to a
Notion board, and can draft a tailored application prep brief for any role.

It's built to be personal but not personalised in code: what you want, what you're
qualified for, and what you won't accept all come from your CV plus a few lines of
goals — nothing about your field or seniority is hardcoded. It works for software
engineers of any level, and for non-engineering roles too.

## How it works

```
sources ──▶ scan ──▶ score (Claude) ──▶ shortlist ──▶ Notion board ──▶ prep brief
 (feeds/     (save    (fit 0–100 vs      (push above    (act on it:      (tailored,
  HTTP/       new      your profile)      threshold)     applied, etc.)   per-role)
  browser)    jobs)
```

1. **Profile** — your CV (PDF/Markdown) + a goals blurb are distilled into a
   structured profile (skills, seniority, must-haves, dealbreakers) by Claude.
2. **Scan** — each configured source is scraped; new listings are saved to a local
   SQLite database and bulk-scored against your profile.
3. **Score** — every listing gets a 0–100 fit score with a breakdown, rationale, and
   flags. The rubric's weights and thresholds are configurable (see [Scoring](#scoring)).
4. **Shortlist / Notion** — roles above your fit threshold are pushed to a Notion
   board you can work from. Actions you set in Notion (Mark as Applied, Not
   Interested, Prep) are polled and executed by `watch`.
5. **Prep** — generate a per-role prep brief (CV tweaks, cover-letter angles, likely
   interview questions, people to contact) that's appended to the Notion page.

## Install

Requires Python 3.12+.

```bash
git clone <your-fork-url>
cd jobscout
python -m venv .venv && source .venv/bin/activate
pip install -e .
playwright install chromium   # only needed for browser-type sources
```

## Quickstart

```bash
jobscout init                 # scaffold config.toml from the example
# edit config.toml: set your CV path, goals, API key, and sources
jobscout profile              # build your candidate profile from CV + goals
jobscout scan                 # scrape sources and score new listings
jobscout list                 # see ranked matches in the terminal
```

To use the Notion board:

```bash
# create a Notion integration (https://www.notion.so/my-integrations),
# share a page with it, then:
jobscout init --notion-parent <notion-page-url>   # creates the board, prints its ID
# paste the database_id into config.toml, then:
jobscout shortlist            # push strong matches to Notion
jobscout watch                # poll Notion for actions and execute them
jobscout prep <listing-id>    # generate a prep brief for a role
```

Run `jobscout --help` for the full command list.

## Configuration

Copy `config.example.toml` to `config.toml` (done for you by `init`) and fill it in.
`config.toml` is git-ignored — it holds your keys and personal goals.

- `[profile]` — `cv_path` (PDF or `.md`) and a free-text `goals` blurb.
- `[ai]` — one `openrouter_api_key` (or `OPEN_ROUTER_API_KEY`) powers everything:
  bulk scoring, profile extraction, and prep briefs. `bulk_model` / `deep_model` /
  `prep_model` are OpenRouter slugs (e.g. `anthropic/claude-haiku-4-5`,
  `google/gemini-3.5-flash`); plus `fit_threshold` (push to Notion at/above this score).
- `[notion]` — integration `token` (or `NOTION_TOKEN`) and `database_id`.
- `[scoring]` — the tunable rubric (below).
- `[[sources]]` — the boards and feeds to scrape (see [docs/sources.md](docs/sources.md)).

### Scoring

The scoring rubric is data-driven, so you tune it without touching code. The *content*
of what matters to you comes from your profile; the `[scoring]` section holds the hard
numbers:

```toml
[scoring]
salary_floor = 100000            # deduct hard below this; omit to ignore pay entirely
salary_currency = "$"            # wording only
max_office_days = 3              # more than this in office is a hard negative; omit to ignore
specialist_domain_ceiling = 100  # cap roles needing specialist knowledge you lack; 100 = off

[scoring.weights]                # max points each dimension can add/subtract
title = 20                       # match to your target job titles
scope = 20                       # responsibility / seniority / autonomy fit
company = 15                     # size, stage, and type of company
stack = 10                       # how well the required skills/tools match yours
domain = 10                      # penalty when the industry needs specialist knowledge
salary = 3                       # penalty when pay is unstated
office = 3                       # penalty when at your office-day limit
dealbreakers = 50                # penalty when one of your stated dealbreakers is confirmed
```

All fields are optional with sensible defaults. Leaving `salary_floor` and
`max_office_days` unset makes the scorer ignore pay and location — useful if your CV
and goals already express those preferences, or for non-engineering searches.

Your dealbreakers, must-haves, and preferred titles/domains are read from your profile
(derived from your CV + goals), not from this file — so the same rubric works for any
role. Edit your `goals` and re-run `jobscout profile` to change them.

### Gated assessment (experimental)

Set `scorer = "gated"` under `[ai]` to switch from the single additive fit score to a
two-part decision: hard eligibility **gates** answer *"should I apply?"* and tiered
**priority** answers *"which first?"*. The LLM only extracts facts from the listing
(with confidence and a supporting quote); pure Python applies your policy, so decisions
are stable, testable, and explainable.

- `[[gates]]` — non-compensatory vetoes: any failed gate means `decision = "no"`, no
  matter how good the rest is. Each is either a structured predicate over a canonical
  extracted feature (`feature`/`op`/`value` — see `jobscout/features.py`) or a
  natural-language `rule` ("Reject only if …"). `on_unknown` controls what happens when
  the listing doesn't state the fact (`pass` / `fail` / `pass_flag`), and a gate can
  only hard-fail on a high-confidence detection.
- `[[priority.tiers]]` + `[priority]` — rank the apply set by tier (matched on title
  or extracted role substance — substance wins), with per-unknown-gate penalties
  breaking ties within a tier.

`fit_score` is still written (derived: apply ⇒ 70–100 by tier, no ⇒ below 70), so the
shortlist threshold, Notion `Fit` column, and existing tooling keep working during the
migration. See `config.example.toml` for the full config surface and
`validate_gate.py --scorer gated` for regression-testing gates against roles you
actually applied to.

For how the gated pipeline came to be and how bulk scoring moved off Sonnet 4.6 to a
cheaper model without losing quality, see
[docs/llm-optimization-journey.md](docs/llm-optimization-journey.md).

## Cost

Every listing is sent to an LLM for scoring, so scans cost money. All calls route
through OpenRouter, so pick a cheap `bulk_model` slug (e.g. `anthropic/claude-haiku-4-5`,
`google/gemini-3.5-flash`, or a `:free` model) since it runs on every listing, and
reserve stronger `deep_model` / `prep_model` slugs for profile extraction and prep
briefs. Use `bakeoff.py` to compare model cost and quality before you commit. Watch your
volume when adding high-traffic sources.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check jobscout/
```

## License

MIT — see [LICENSE](LICENSE).
