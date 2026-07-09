# Sources cookbook

A **source** is one job board or feed to scrape. You add them as `[[sources]]` blocks
in `config.toml`. Each `scan` fetches every enabled source, saves new listings, and
scores them.

There are three types: `feed`, `http`, and `browser`.

Common fields for every source:

```toml
[[sources]]
name = "A human-readable label"   # shown in the UI and stored on each listing
type = "feed"                     # "feed" | "http" | "browser"
url = "https://..."               # the page or feed to fetch
enabled = true                    # set false to skip without deleting the block
```

---

## `feed` — RSS / Atom

The simplest type. Point it at an RSS/Atom feed and every entry becomes a listing. No
selectors needed.

```toml
[[sources]]
name = "Hacker News Who's Hiring"
type = "feed"
url = "https://hnrss.org/whoishiring"
enabled = true
```

Good sources of feeds: [hnrss.org](https://hnrss.org) (HN "Who's Hiring" and jobs),
company career pages that expose RSS, and aggregators with feed export. If a board has
a feed, prefer it — it's more stable than scraping HTML.

---

## `http` — scrape a listing page with CSS selectors

For boards without a feed. You give CSS selectors that point at the listing card and
its fields. The scraper finds every `container`, then reads each field from within it.

```toml
[[sources]]
name = "Example board – senior backend"
type = "http"
url = "https://example.com/jobs?q=senior+backend"
enabled = true
selectors.container = "div.job-card"     # repeated element, one per listing
selectors.title = "h2.job-title"         # required — a listing with no title is skipped
selectors.company = "span.company"
selectors.location = "span.location"
selectors.url = "a.job-link"             # required — its href is the listing URL
selectors.description = "div.snippet"
selectors.fetch_detail = false           # see below
```

Notes:

- **`container`, `title`, and `url` are the important ones.** A card is skipped unless
  it yields both a title and a URL. If you omit `container`, the whole page is treated
  as a single listing (useful for a single-job page).
- `title`, `company`, `location`, `description` read the matched element's **text**.
  `url` reads the `href` **attribute** of its match, resolved to an absolute URL.
- **`fetch_detail = true`** makes the scraper open each listing's own page and pull a
  fuller job description from it (it tries JSON-LD structured data first, then falls
  back to the `<main>` region's paragraphs). Use this when the listing cards only show
  a short snippet — it gives the scorer much more to work with, at the cost of one
  extra request per listing.

### Finding the selectors

1. Open the board's search results in a browser.
2. Right-click a job card → **Inspect**.
3. Find the repeating wrapper element for one card → that's `container`.
4. Inside it, find the title link, company, location text → those are your field
   selectors, written relative to the whole page (the scraper scopes them to each
   container automatically).

Selectors are the brittle part: sites change their markup, so expect to re-check a
scraping source occasionally if it stops returning results.

### Worked example: Built In

[Built In](https://builtin.com) (and its city sites like builtinlondon.uk) works well.
Change the `search=` query and, if needed, the city subdomain:

```toml
[[sources]]
name = "Built In London – search"
type = "http"
url = "https://www.builtinlondon.uk/jobs?search=your+role+here"
enabled = true
selectors.container = "div[data-id='job-card']"
selectors.title = "a[data-id='job-card-title']"
selectors.company = "a[data-id='company-title'] span"
selectors.url = "a[data-id='job-card-title']"
selectors.location = "span.font-barlow.text-gray-04"
selectors.description = ".bounded-attribute-section"
selectors.fetch_detail = true
```

To track several searches, add one block per query (e.g. one per job title you're
targeting). Give each a distinct `name`.

### Worked example: LinkedIn (no login)

LinkedIn's normal job pages need a login, but its **guest job-search endpoint** returns
job-card HTML without authentication, so it works as an `http` source. Put your search
terms in `keywords=` and your location in `location=` (URL-encoded), and keep the rest:

```toml
[[sources]]
name = "LinkedIn – Head of Engineering (London)"
type = "http"
url = "https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords=Head%20of%20Engineering&location=London%2C%20England%2C%20United%20Kingdom&f_TPR=r604800&start=0"
enabled = true
pages = 3
page_param = "start"
page_size = 10
selectors.container = "li"
selectors.title = "h3.base-search-card__title"
selectors.company = "h4.base-search-card__subtitle a"
selectors.location = "span.job-search-card__location"
selectors.url = "a.base-card__full-link"
selectors.fetch_detail = true
```

- `f_TPR=r604800` restricts results to jobs posted in the last 7 days (the number is
  seconds; `r86400` = last 24h). Omit it for all results.
- **Pagination:** the endpoint returns ~10 cards per request, so fetch only the first 10
  without it. `pages = 3` with `page_param = "start"` and `page_size = 10` fetches
  `start=0,10,20` (the top 30); fetching stops early if a page comes back empty. See
  the pagination note below.
- `fetch_detail = true` pulls the full job description from each posting's JSON-LD
  (LinkedIn embeds it), which the scraper decodes to clean text.
- **Rate limits:** LinkedIn throttles scraping. Each page is one list request, plus one
  request per job for the detail fetch, so `pages × ~10` detail fetches per block. A few
  keyword blocks at `pages = 3` is usually fine; higher `pages` or many blocks will start
  returning `429`. If that happens, lower `pages`, drop `fetch_detail`, or scan less
  often. Respect LinkedIn's terms and keep volume light.

**Pagination (any http source).** Set `pages` to fetch more than one page. Each page
advances the `page_param` query parameter by `page_size`, starting from its current value
in the `url`. For offset-based endpoints like LinkedIn use `page_param = "start"`,
`page_size = 10`; for page-number boards use `page_param = "page"`, `page_size = 1`.
Fetching stops as soon as a page returns no listings.

---

## `browser` — pages that need JavaScript or a login

For boards that render listings with JavaScript, or that require you to be logged in.
This uses a headless Chromium via Playwright, so run `playwright install chromium`
once first.

```toml
[[sources]]
name = "Example login-required board"
type = "browser"
url = "https://example.com/jobs"
enabled = true
login_url = "https://example.com/login"
username_env = "EXAMPLE_USERNAME"   # credentials come from env vars, never config
password_env = "EXAMPLE_PASSWORD"
selectors.container = "div.job-card"
selectors.title = "h2.title"
selectors.company = "span.company"
selectors.location = "span.location"
selectors.url = "a[href]"
selectors.description = "div.body"
```

The selectors work exactly as for `http`. Set the credential env vars before scanning:

```bash
export EXAMPLE_USERNAME=you@example.com
export EXAMPLE_PASSWORD=…
jobscout scan
```

Keep credentials in environment variables (or a secrets manager) — never commit them.
Only scrape boards whose terms permit it, and be respectful of rate limits.

---

## Tips

- Start with `feed` sources; they need no maintenance.
- Add `http` sources for the specific searches you care about, one block per query.
- Use `fetch_detail = true` when cards are sparse — it materially improves scoring.
- If a source suddenly returns nothing, its selectors have probably gone stale;
  re-inspect the page and update them.
