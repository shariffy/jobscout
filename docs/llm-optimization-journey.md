# LLM optimisation journey

How jobscout's bulk scoring went from a single Claude Sonnet 4.6 "fit score" to a
gated, config-driven decision extracted by **gemini-3.1-flash-lite + majority-of-3**
— matching Sonnet's quality at roughly a third of the cost. This is the reasoning
trail: where we started, what we tried, what the numbers said, and what each result
changed.

> **TL;DR.** Splitting *eligibility* (a veto) from *preference* (a ranking) turned the
> LLM's job from holistic judgment into fact extraction. That reopened the model
> question — the earlier "stay on Sonnet" verdict was about judgment, which now lives
> in Python. A structured bake-off on the *extraction* task, plus a self-consistency
> vote to buy back stability, landed on flash-lite: **0% decision flip, 100% agreement
> with Sonnet, 17/17 ground-truth roles, ~$4.5 vs ~$14 per full-corpus scan.**

---

## 1. Where we started

The bulk scorer defaulted to **Claude Haiku 4.5** — cheap enough to run on every
listing. In practice it was **erratic**: the same listing scored differently between
runs, dealbreakers were missed or invented, and the shortlist wasn't trustworthy. The
first fix was to **bump `bulk_model` to Sonnet 4.6**, which steadied things — but only to
*passable*, and because bulk scoring runs on every listing of every scan, the **daily
cost was high** (~$11 to score the corpus once, repeated per scan).

Under the hood both models ran the same design: a single additive **fit score** (0–100)
from a rubric prompt — sum weighted dimensions (title, scope, company, stack, domain,
salary, office, dealbreakers), cap to 0–100, shortlist at ≥ 70.

Two structural problems sat beneath the model choice:

1. **It fused two things that want opposite treatment.** *Eligibility* ("would I ever
   apply?") is a veto — non-compensatory, one dealbreaker should sink a role no matter
   how good everything else is. *Preference* ("which eligible one first?") is graded.
   Averaging them into one number made scoring knife-edge sensitive to weights and prone
   to *false dealbreakers*.
2. **The decision lived inside the model's judgment.** A 0–100 "vibe" drifts run-to-run
   and between models.

"Passable but expensive" set up the next step: a bake-off to find out whether a cheaper
model could match Sonnet — or whether the problem wasn't the model at all.

---

## 2. The pre-gates bake-off

Could a cheaper model replace Sonnet on the additive task? We built `bakeoff.py` (plus a
label-free scorecard) and ran it in several iterations. *(The raw report still on disk is
the final Sonnet-vs-DeepSeek run, 30 listings × 4 repeats; the earlier multi-model
numbers survive as summarised findings.)*

**Iteration 1 — cheap-model cost sweep.** Six OpenRouter models (Sonnet 5, Gemini 3.5
Flash, DeepSeek v4 Pro, Nemotron-3, Qwen3-next, Hy3) plus an OpenAI arm (GPT-5.6
Terra/Luna), scored against Sonnet's stored scores. All rejected:
- **GPT-5.6 (Terra/Luna)** — too lenient on dealbreakers (scored a sales role and a
  consultancy dealbreaker as ~60% fits), with no real cost win.
- **Sonnet 5** — its reasoning tokens made it the *most expensive* option, not the cheap
  upgrade it looked like.
- **Haiku 4.5** — only ~20% cheaper than Sonnet 4.6 on the full prompt, still emitting
  malformed JSON. (The very model we'd started on.)

**Iteration 2 — a label-free Tier-1 scorecard.** With no gold labels, screen each model
for a *disqualifying flaw* using pure post-processors on the JSON it already returned:
rubric-internal consistency (does `fit = 50 + Σ breakdown`? are terms in range? do flags
cohere with scores?) and test–retest reliability (score each listing K times, measure
variance). The data (Sonnet 4.6 vs DeepSeek v4 Pro, 30 listings × 4 repeats):

| model | mean fit | Δ vs stored | rubric-clean % | σ retest | $/corpus |
|---|--:|--:|--:|--:|--:|
| Sonnet 4.6 | 41.3 | 13.2 | **26%** | 2.74 | $11.04 |
| DeepSeek v4 Pro | 32.7 | 23.5 | 15% | **16.13** | $2.17 |

**Iteration 3 — the DeepSeek cost-challenger deep-dive.** DeepSeek was ~5× cheaper, so it
got a focused look — and was *wildly* unstable: on identical input it scored the same
listing anywhere from **11 to 81** (σ 16.13 vs Sonnet's 2.74) — e.g. a Founding Engineer
role at 44 (range 11–81) across four runs. A shortlist you can't reproduce is worse than
a slightly worse one you can. Rejected.

**Iteration 4 — rubric hardening.** The scorecard's real lesson was that **the lever was
the rubric, not the provider.** We tightened the additive prompt: function/track mismatch
treated as dealbreaker-class, in-office detection from prose (not just the scraped
location tag), a concrete over-scope rule, and "minimum N days" at the candidate's limit
treated as borderline rather than an auto-reject.

**Bake-off verdict: stay on Sonnet 4.6, with a tighter rubric.** The cost problem wasn't
the provider; no cheaper model was both accurate and stable enough.

---

## 3. The reframe that reopened the model question

Staying on Sonnet plus a tighter rubric helped — but the scorecard had surfaced something
no model choice or prompt could fix. **Even Sonnet contradicted its own rubric 74% of the
time** (only 26% of outputs "clean"), and the violations were overwhelmingly *arithmetic*:
it reported a `fit` well below `50 + Σ breakdown`. The breakdown terms summed high, and
the model quietly **clamped the total down** to force a rejection it had no additive term
to express:

- Engineering Manager @ Light — breakdown summed to **83**, reported **62**.
- Founding Engineer @ Goodley — summed to **87**, reported **57**.
- Engineering Manager @ Deliveroo — summed to **71**, reported **55**.

The model *knew* these were worse than the sum (off-track, or an unstated dealbreaker) but
broke its own formula to say so. That is not a prompt bug or a model bug — it is the
additive design fighting a **non-compensatory** reality. No weight tuning makes "one veto
overrides everything" expressible as a weighted sum.

So we replaced the additive score with a **gated + tiered assessment**
(`jobscout/assess.py`):

- **The LLM does perception, not judgment.** One call per listing extracts a canonical
  set of facts (`jobscout/features.py`) plus a pass/fail/unknown verdict on each
  natural-language gate rule — each with a **confidence** and a **verbatim evidence
  quote**.
- **Python does policy.** Pure-Python engines apply the config's `[[gates]]` (hard,
  non-compensatory vetoes → `decision = apply | no`) and `[priority]` tiers (a
  bucketed-lexicographic rank of the apply set). Only a *high-confidence* detection can
  fail a gate; anything uncertain is "unknown" and handled by an explicit `on_unknown`
  policy — which directly fixes the false-dealbreaker collapse.

This is more stable, testable, and explainable. But it also **changed what the model is
for**: narrow fact extraction, not a 0–100 opinion. Every prior rejection of a cheap
model had been about *judgment quality* — and judgment is now Python's job. So the
model question was worth reopening, on the new task.

---

## 4. How we measured "good" on an extraction task

A judgment score has no ground truth; an extraction does. That let us measure the axes
that actually matter, without gold labels for most of it:

| axis | what it means | how |
|---|---|---|
| **decision flip** | same listing, K identical repeats — does `apply/no` change? | the extraction-task analogue of score variance; **the** reliability metric |
| **agreement vs Sonnet** | does the model reach Sonnet's decision (and feature values)? | Sonnet as a *silver standard* — the right target for a drop-in replacement |
| **grounding** | does each evidence quote actually appear in the listing? | a direct hallucination check (the evidence field is verbatim) |
| **schema coverage** | how many of the canonical features does it populate? | catches models that can't hold the schema |
| **cost & latency** | real OpenRouter charged cost + per-call time | projected to the full corpus |

The guiding principle: **a model can be perfectly stable and stably wrong**, so cost and
consistency alone are not enough — agreement with a trusted reference is the accuracy proxy.

---

## 5. The investigations, in order

### 5a. Sonnet 4.6 vs DeepSeek v4 Pro on the gated task

`scorecard_gated.py`, 12-listing sample:

| model | decision flip | tier flip | proj. corpus |
|---|--:|--:|--:|
| Sonnet 4.6 | **0%** | 0% | $14.18 |
| DeepSeek v4 Pro (reasoning off) | 17% | 25% | $2.80 |

DeepSeek was ~80% cheaper but flipped the decision on 2–3 of 12 identical inputs. **For a
pipeline whose value is a stable yes/no, that's disqualifying regardless of cost.**

### 5b. A fairness correction (raised by the user)

The first DeepSeek run forced reasoning *off* (`{"enabled": false}`), which unequally
handicaps a reasoning-native model versus Sonnet's default. We re-ran DeepSeek at
`{"effort": "low"}` — a within-model off-vs-low isolation (`deepseek_reasoning_probe.py`):

- off: 25% flip · low: 18% flip — a **one-listing** difference at n=12, i.e. within noise.
- Reasoning did **not** close the gap to Sonnet's 0%, and cost rose 2.3×.
- Tellingly, the *same* config measured 17% then 25% flip across two runs — the
  instability showed up even in the instability metric.

**Corrected verdict:** not "reasoning-off DeepSeek is unreliable" (confounded), but
"DeepSeek flips a meaningful fraction of identical inputs in *every* config tested,
while Sonnet flipped none."

### 5c. The broad extraction sweep

With the task now extraction, the field reopened — including current-gen cheap models
the earlier judgment-era bake-off never fairly tested. `extraction_sweep.py`, 6 models
vs a Sonnet reference:

| model | schema cover | decision flip ↓ | agrees w/ Sonnet ↑ | ~speed | proj. corpus |
|---|--:|--:|--:|--:|--:|
| **gemini-3.1-flash-lite** | 100% | **8%** | **100%** | ~3s | **$1.54** |
| gpt-5.6-luna | 100% | 9% | 100% | ~4s | $5.05 |
| haiku-4.5 | 100% | 17% | 83% | ~7s | $5.59 |
| gpt-4o-mini | 100% | 33% | 75% | ~10s | $0.55 |
| llama-3.1-8b | 90% | 55% | 33% | ~40s | $0.05 |
| *Sonnet 4.6 (ref)* | — | *~0%* | *100%* | *~10s* | *$14.22* |

Dropped mid-run once the streaming logs exposed them: **gemma-4-31b (free)** — ~16 min
per pass and 0 features returned (couldn't hold the schema); **qwen3.5-flash** — ~24 s/call
(untenable for per-listing extraction). Older models suggested externally (gpt-4o-mini,
llama-3.1-8b) confirmed their age — poor agreement, and the 8B was also ~40 s/call.

flash-lite led on agreement, speed, parsing, and cost. Its 77% *feature* agreement
despite 100% *decision* agreement was the ideal failure shape: where it differs from
Sonnet, it's on decision-irrelevant fields.

### 5d. Chasing haiku's paradox → the office gate

Haiku extracted the most *complete and stable* feature set (5.4/9 features, lowest
run-to-run spread) yet had the **worst** decision flip (17%) and lowest agreement (83%).
Reconstructing it from cache showed why: the count was stable, but the *decision-relevant*
`office` verdict wobbled. Every one of haiku's flips/disagreements was the **office gate
firing on founding-engineer roles** — reading in-office presence into JDs more
aggressively than Sonnet.

The bigger finding: **the office gate was the #1 flip driver across *every* cheap model**,
not a per-model quirk. Founding-role JDs are genuinely ambiguous about office presence,
and models inconsistently inferred a day-count from soft signals ("based in London",
"work alongside the founders", a scraped "In-Office" card tag).

**Fix:** tightened the `office_days_per_week` extraction contract (`features.py`) to
return `null` unless a policy or day-count is *explicit*, and to never infer one from
soft co-location signals. Re-measuring flash-lite: all four founding roles went to a
**stable apply**, 0 office-gate fails. The residual flip *relocated* (to one Senior EM
role, via a subtler org-structure rule) rather than vanishing — a cleaner, more
defensible place to be uncertain.

### 5e. Self-consistency: majority-of-3

flash-lite's remaining ~8% single-run flip was bought out with a **majority-of-3 vote**
(`ai.scorer_repeats`, `consensus_assessment()`), which the model is cheap and fast enough
to afford. Final head-to-head under the tightened prompt (`compare_scorers.py`):

| config | flip ↓ | agree w/ Sonnet | proj. corpus | ~s/score |
|---|--:|--:|--:|--:|
| flash-lite single | 8% | 100% | $1.50 | 3.0s |
| **flash-lite maj-of-3** | **0%** | **100%** | **$4.49** | 9.0s |
| luna single | 17% | 100% | $5.08 | 4.4s |
| *Sonnet 4.6 (ref)* | *~0%* | *100%* | *$14.22* | *~10s* |

Majority-of-3 took flash-lite to **Sonnet-level 0% flip** while keeping 100% agreement,
at ~1/3 the corpus cost and comparable latency (the 3 calls are independent, so
parallelising them returns to ~3 s). It also degrades gracefully: flash-lite errored on
~6.5% of calls, and the vote still stood on the survivors.

### 5g. Early-stop voting (free ~30% off the vote)

A fixed majority-of-3 always pays for the third call — but once two of three runs
agree on the decision *and* tier, the third mathematically cannot change either
(`consensus_locked` in `jobscout/assess.py`): a 2-vs-1 majority is already settled, so
the remaining call is pure cost. `GatedScorer.score` now stops as soon as decision+tier
are locked. Replaying the production `score()` path over the on-disk extraction cache
(119 listings × 3 stored repeats, **$0** in new calls) confirmed the equivalence:

| strategy | decision+tier vs fixed-3 | calls/listing | proj. corpus |
|---|--:|--:|--:|
| fixed majority-of-3 | — | 3.00 | $4.49 |
| **early-stop (decision+tier lock)** | **119/119 identical** | **2.09** | **~$3.13** |

This is *exact* on the two fields that drive the pipeline, not an approximation — the
lock condition is provable, so there is no accuracy trade to weigh. Only the derived
representative fields (`fit_score`, within-tier `priority`) can shift when a listing stops
at two runs, the same latitude `consensus_assessment` already takes picking a median
representative. A *conservative* variant that also waited for the full policy signature
(flags/priority/fit) to agree cost more (~2.35–2.52 calls) yet still couldn't guarantee
those secondary fields — so it paid a premium to stabilise the fields that matter least.
Rejected in favour of the decision+tier lock.

**Parallelising the mandatory calls.** A K-way majority can't be settled by fewer than
floor(K/2)+1 agreeing runs, so those first votes are *always* made — `GatedScorer.score`
now issues them concurrently (a `ThreadPoolExecutor`), then runs the conditional
remainder serially with the same lock. For K=3 that is two calls at once, then a third
only on disagreement: latency for the ~85% two-call case drops from ~2×call to ~1×call
(a mocked 0.3 s/call run finishes in 0.30 s, not 0.60 s) with **no** change to call count
or decision — parallelism buys latency, the lock buys the calls. The vote's worker
threads reach the sqlite extraction cache through a lock on `Store` (the only table
touched concurrently); everything else stays single-threaded. Fanning out *across
listings* — `score_batch` is still a serial loop — is the larger remaining win and reuses
the same thread-safe cache, but it interacts with source rate limits (LinkedIn 429s) and
was left as a separate step.

### 5f. The real acceptance test

`validate_gate.py --scorer gated` re-scores every role actually applied to or prepping
(17 human-labelled, dealbreaker-free) and asserts `decision == apply` for all:

```
✅ GATE PASSED — 17/17 roles → apply, zero run-to-run disagreement, sensible tiers.
```

Stronger than the 12-listing agreement, and it passed.

---

## 6. What it informed

- **Production config** (`config.toml`): `bulk_model = "google/gemini-3.1-flash-lite"`,
  `bulk_reasoning = { effort = "low" }`, `scorer = "gated"`, `scorer_repeats = 3`.
  Sonnet 4.6 remains `deep_model` / `prep_model`.
- **The gated pipeline** shipped as the scoring model of record; `fit_score` is retained
  as a derived compatibility number during the Notion/DB migration.
- **The office contract fix** stabilised every candidate at once — higher-leverage than
  the model choice itself.
- **Self-consistency** (`scorer_repeats`) is now a first-class, config-driven capability,
  not a benchmark trick — the lever that lets a cheaper model match a stronger one.
- **Early-stop voting** (`consensus_locked`) makes the majority vote cost ~2.09 calls
  instead of a fixed 3 — a free ~30% off, provably identical on decision+tier (§5g).

---

## 7. Methodology lessons (worth keeping)

- **Measure decision stability, not mean score.** On a decision pipeline, flip rate is
  the metric; cost and mean-fit hide the failure that matters.
- **Extraction has ground truth — use it.** Agreement with a trusted silver-standard
  (Sonnet) catches "stable but wrong," which consistency alone cannot.
- **Reasoning settings must be fair.** Forcing reasoning off on a reasoning-native model
  handicaps it unequally; pin an explicit, task-appropriate effort per model.
- **Verify settings before the big run.** Smoke-test each model slug *and its reasoning
  config* on one listing, so config bugs cost cents, not dollars.
- **Checkpoint expensive runs.** Persist every result as it completes and reuse it on
  re-run; a kill or a one-model config change should never re-pay completed work
  (especially the Sonnet reference).
- **Stream per-item progress.** A slow or rate-limited model must read as *crawling*, not
  *hung* — that's how gemma and qwen were caught and dropped.

---

## 8. Reproducing

Production tooling (committed): `validate_gate.py` (ground-truth regression gate),
`bakeoff.py` (cross-provider harness), `migrate_notion.py` (one-shot board migration).

The exploratory analysis scripts behind §5 (`extraction_sweep.py`, `compare_scorers.py`,
`scorecard_gated.py`, `deepseek_reasoning_probe.py`, `remeasure_office.py`) were one-off
and are intentionally **not** committed; their findings are captured here. All ran on a
12-listing stratified sample and through OpenRouter, keyed by an on-disk result cache so
re-runs never re-paid completed calls.

> **Caveats.** The headline numbers rest on a 12-listing sample plus the 17-role ground
> truth — strong signal, not proof. "Agreement" is measured against Sonnet as a silver
> standard, so flash-lite inherits any Sonnet error (acceptable for a drop-in
> replacement, which was the goal). A larger, gate-boundary-focused validation and a set
> of negative (known-dealbreaker) labels are the natural next steps before fully trusting
> the cheaper model corpus-wide.
