# jobscout

An AI-assisted job-search pipeline. It scrapes job boards and feeds, scores each
listing against **your** CV and goals using a configurable LLM (Gemini by default
for bulk scoring; Claude via OpenRouter for deep analysis and prep), pushes the
strong matches to a Notion board, and can draft a tailored application prep brief
for any role.

It's built to be personal but not personalised in code: what you want, what you're
qualified for, and what you won't accept all come from your CV plus a few lines of
goals — nothing about your field or seniority is hardcoded. It works for software
engineers of any level, and for non-engineering roles too.

## How it works

```
sources ──▶ scan ──▶ score (LLM) ──▶ shortlist ──▶ Notion board ──▶ prep brief
 (feeds/     (save    (fit 0–100 vs      (push above    (act on it:      (tailored,
  HTTP/       new      your profile)      threshold)     applied, etc.)   per-role)
  browser)    jobs)
```

1. **Profile** — your CV (PDF/Markdown) + a goals blurb are distilled into a
   structured profile (skills, seniority, must-haves, dealbreakers) by an LLM
   (the `prep_model` route — Claude Sonnet via OpenRouter by default).
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
# edit config.toml: set your CV path, goals, API keys, and sources
# (see [ai] section in config.example.toml — defaults need GEMINI_API_KEY
#  and OPEN_ROUTER_API_KEY; Notion board also needs NOTION_TOKEN)
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
- `[ai]` — `bulk_model` / `deep_model` / `prep_model` are each `"provider:model"`
  (e.g. `openrouter:anthropic/claude-haiku-4-5`, `google:gemini-3.1-flash-lite`),
  so different model slots can hit different providers/gateways — a bare slug
  with no prefix defaults to `openrouter`. Each provider resolves its key from
  its own env var (`OPEN_ROUTER_API_KEY`, `REQUESTY_API_KEY`, `GEMINI_API_KEY`,
  `ANTHROPIC_API_KEY`), or override any of them under `[ai.api_keys]`.
- `[notion]` — integration `token` (or `NOTION_TOKEN`) and `database_id`.
- `[[gates]]` / `[priority]` — eligibility and ranking (below).
- `[[sources]]` — the boards and feeds to scrape (see [docs/sources.md](docs/sources.md)).

### Assessment

Hard eligibility **gates** answer *"should I apply?"* and tiered **priority** answers
*"which first?"*. The LLM only extracts facts from the listing (with confidence and a
supporting quote); pure Python applies your policy, so decisions are stable, testable,
and explainable.

- `[[gates]]` — non-compensatory vetoes: any failed gate means `decision = "no"`, no
  matter how good the rest is. Each is either a structured predicate over a canonical
  extracted feature (`feature`/`op`/`value` — see `jobscout/features.py`) or a
  natural-language `rule` ("Reject only if …"). `on_unknown` controls what happens when
  the listing doesn't state the fact (`pass` / `fail` / `pass_flag`), and a gate can
  only hard-fail on a high-confidence detection.
- `[[priority.tiers]]` + `[priority]` — rank the apply set by tier (matched on title
  or extracted role substance — substance wins), with per-unknown-gate penalties
  breaking ties within a tier.

`jobscout shortlist` pushes `decision = apply` roles to Notion (Decision / Priority /
Tier). See `config.example.toml` for the full config surface and `validate_gate.py`
for regression-testing gates against roles you actually applied to.

For how the gated pipeline came to be and how bulk scoring moved off Sonnet 4.6 to a
cheaper model without losing quality, see
[docs/llm-optimization-journey.md](docs/llm-optimization-journey.md).

## Cost

Every listing is sent to an LLM for scoring, so scans cost money. Pick a cheap
`bulk_model` route (e.g. `google:gemini-3.1-flash-lite` on a free tier, or
`openrouter:anthropic/claude-haiku-4-5`) since it runs on every listing, and
reserve stronger `deep_model` / `prep_model` slugs for profile extraction and prep
briefs. Use `validate_gate.py` to check a model change against roles you’ve applied to.
Watch your volume when adding high-traffic sources.

## Privacy

jobscout is a thin client over third-party LLM APIs and Notion — nothing is kept
private by default that those services don't already see:

- **Your CV, goals, and derived candidate profile** (`candidate_profile.json`) are
  sent to whichever LLM providers your `[ai]` routes point at — OpenRouter,
  Requesty, Google, or Anthropic — on **profile build** and **prep**. The default
  gated scoring run (`scan` / `rescore`) sends only the listing text plus a gate
  schema — **not** your profile — to the bulk model on every listing.
- **Every scraped listing's full text** (title, company, description) is sent to
  those same providers for scoring, and again for `prep` briefs.
- **Strong matches are written to your Notion workspace**: title, company, score,
  decision/tier, and the LLM's rationale, via the token in `[notion]`/`NOTION_TOKEN`.
- **`prep` uses server-side web search** (where the provider supports it — see
  `jobscout/llm.py`) and explicitly asks the model to name real people at the
  target company (from LinkedIn, the company site, Crunchbase, press) to draft
  outreach messages. That's a deliberate feature, not incidental — know that it's
  happening before you run it against a company you'd rather not surface yourself to.
- **No telemetry.** jobscout doesn't phone home or report usage anywhere beyond the
  providers and services you've configured.

**Secrets:** prefer environment variables (`OPEN_ROUTER_API_KEY`, `GEMINI_API_KEY`,
`ANTHROPIC_API_KEY`, `REQUESTY_API_KEY`, `NOTION_TOKEN`, …) over the config-file
`[ai.api_keys]` / `[notion].token` fields — env vars keep credentials out of a file
that could be committed or read by another local user. If you do put a secret in
`config.toml`, it's created with owner-only (`0600`) permissions, as are the
profile cache and SQLite database — but env vars are still the safer default.

## Development

```bash
pip install -e '.[dev]'
pytest
ruff check jobscout/
```

`requirements.lock` pins the exact versions this project is tested against
(`pip install -r requirements.lock` installs deps only — the editable install
`pip install -e .` is still required to get the `jobscout` CLI on your path);
`pyproject.toml` holds the loose ranges.

## License

MIT — see [LICENSE](LICENSE).
