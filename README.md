# Support Inbox Assistant

First-pass triage for a support inbox. Each inbound ticket gets a category, a
priority, a one-line summary, a draft reply, and a confidence score, then lands
in a queue where a human reviews it, edits the reply, and approves or rejects.

**Nothing is ever sent to a customer.** There is no send endpoint anywhere in
the codebase — that safety property is enforced by absence rather than by a
runtime check.

---

## Quick start

**Prerequisites.** Python 3.10+, and [Ollama](https://ollama.com/download)
installed and running — the triage step talks to it over HTTP and every command
below except `pytest` needs it up.

Set two Ollama variables at user scope before the first run. Ollama reads them
at startup, so quit and relaunch it afterwards (on Windows, quit from the tray
rather than killing the process):

```powershell
setx OLLAMA_CONTEXT_LENGTH 8192
setx OLLAMA_KEEP_ALIVE 30m
```

`OLLAMA_CONTEXT_LENGTH` matters more than it looks: Ollama truncates from the
*front* of the prompt silently, so an undersized context drops the system prompt
and leaves the model blind rather than broken. Both variables are explained
under *Environment variables* below.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # POSIX: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env              # POSIX: cp .env.example .env

ollama pull llama3.2:3b

python scripts/run_triage.py        # triages all 30, ~2.3 min
python eval/run_eval.py             # writes eval/results.json
uvicorn app.api:app                 # review queue at http://localhost:8000
pytest                              # 17 tests, ~0.2s, no model needed
```

Fill in the four required values in `.env` before running anything — see
*Environment variables*. The three optional knobs are commented out in the
template and can stay that way.

Run everything from the repo root — `.env` is resolved relative to the working
directory.

### Where to read results

| File | What it holds |
|---|---|
| `eval/results.json` | The graded artefact: the two metrics plus one prediction per ticket, all 30, in exactly the shape the brief specifies |
| `eval/diagnostics.json` | Per-ticket detail for the error analysis: model vs final confidence, signals, escalation reasons, retries, labels, match flags |

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `LLM_BASE_URL` | yes | OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1` |
| `LLM_MODEL` | yes | Model tag as the endpoint reports it |
| `LLM_API_KEY` | yes | Ignored by Ollama; a real credential against a hosted provider |
| `LLM_TIMEOUT_SECONDS` | no | Seconds before a single attempt is abandoned |
| `LLM_MAX_TOKENS` | no | Upper bound on generated tokens per call |
| `LLM_MAX_RETRIES` | no | Retries *after* the first attempt |
| `CONFIDENCE_THRESHOLD` | yes | Final confidence below which a ticket escalates |

Defaults for the optional three live in `app/config.py`, the single source of
truth for them. The four required variables have no defaults, so no endpoint,
model name, credential, or policy value is embedded in tracked source. Pointing
this at OpenAI or any other OpenAI-compatible provider is a `.env` edit, not a
code change.

`CONFIDENCE_THRESHOLD` is `0.8` in my `.env` and deliberately empty in
`.env.example` — see *Choosing the threshold*.

Two Ollama variables are set at user scope. Ollama reads them at startup, so it
must be fully quit and relaunched for a change to take effect.

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` | `8192` | Ollama truncates from the *front* of the prompt silently, so an undersized context drops the system prompt and leaves the model blind rather than broken |
| `OLLAMA_KEEP_ALIVE` | `30m` | The default unloads after 5 minutes idle; a cold reload costs 17.3s against a 4.6s warm call |

---

## How it works

```
data/tickets.json
      |
      v
  triage pipeline          batch, offline
      |
      v
   SQLite                  triage + review state
      |
      +---> FastAPI ---> review queue     (edit, approve, reject)
      |
      +---> eval ------> eval/results.json
```

Triage is a **batch step, not a request-time one**. The model takes ~4.6s per
ticket, so triaging on page load would make the queue unusable. Running it once
and persisting also means the eval and the UI report the same predictions rather
than two independent runs that quietly disagree.

A second benefit that turned out to matter: the escalation threshold is a
post-processing decision over stored records, so sweeping six threshold values
cost arithmetic rather than 180 model calls.

```
app/
  config.py      env-driven settings
  schemas.py     the contract: what the model may claim, what the system stores
  prompts.py     system prompt, delimiters, synthetic few-shot examples
  llm_client.py  transport only: retry, backoff, failure classification
  triage.py      the six stations plus the policy layer
  storage.py     SQLite and the review state machine
  api.py         four endpoints, and no fifth
scripts/run_triage.py
eval/run_eval.py
static/index.html
tests/
```

---

## Results

Measured over the 16 labeled tickets. Predictions are written for all 30.

| Metric | Score | Baseline | Gain |
|---|---|---|---|
| `category_accuracy` | **0.9375** (15/16) | 0.375 | +0.5625 |
| `priority_agreement` | **0.6250** (10/16) | 0.375 | +0.2500 |
| `priority_within_one` | 0.9375 | — | — |

The baseline is majority-class: always answer `bug`, always answer `medium`.
Both score 37.5%. Without that comparison neither number above has a scale.

Pipeline behaviour over the full 30: 138s total (4.6s per ticket), 1 fallback,
2 repair retries, 8 escalations — 4 hard-triggered, 4 score-triggered.

### Read the category number carefully

**93.75% is not a clean generalization result, and I don't want it read as
one.** The category boundaries in my system prompt were derived from these same
16 labeled tickets — the billing/account split from the six labeled tickets in
those classes, the blocking tie-breaker from T-005, the tone rule from T-012.
Measuring rules against the tickets they were derived from is close to circular.

The other 14 tickets are the real held-out set and they carry no labels, so
**I have no clean generalization number at all.** I would rather say that than
let 93.75% imply one.

What I can claim more narrowly, and still usefully: the pipeline reliably
produces schema-valid structured output from a model that produced none in 8 of
8 naive attempts, and the classification errors it makes are visible and
explainable rather than silent.

---

## Error analysis

### One category error, and it is instructive

T-019 — a webhook HMAC signature failing verification — is labeled `bug`. The
system classified it `security` at 0.992 confidence. It is the only off-diagonal
cell in the entire confusion matrix, and it costs a point on both metrics.

The system prompt carries an explicit exclusion for exactly this: *a
security-adjacent thing that is merely malfunctioning — a check that fails, a
certificate that expired — is a bug.* Prose alone did not hold.

**I chose not to fix it.** T-019 is one of the 16 labeled tickets; tuning the
prompt against it would make my accuracy measure memorisation rather than
generalization, which is the thing the brief warns about. The right fix is a
few-shot example teaching the exclusion, built from a scenario outside the 30.

### The priority errors are not random — they are a rightward shift

The aggregate 62.5% hides the shape. Per class:

| Labeled priority | Correct | Accuracy |
|---|---|---|
| `high` | 5 / 5 | **100%** |
| `medium` | 4 / 6 | 67% |
| `urgent` | 1 / 2 | 50% |
| `low` | **0 / 3** | **0%** |

**Every ticket labeled `low` was over-estimated.** The system has a working
concept of `high` and effectively no concept of `low`.

All six mismatches:

| Ticket | Predicted | Labeled | Distance | Category | Final conf | Signals |
|---|---|---|---|---|---|---|
| T-009 | high | low | **+2** | correct | 0.985 | `[]` |
| T-003 | medium | low | +1 | correct | 0.970 | `[]` |
| T-013 | medium | low | +1 | correct | 0.970 | `[]` |
| T-017 | high | medium | +1 | correct | 0.985 | `[]` |
| T-024 | high | medium | +1 | correct | 0.985 | `[]` |
| T-019 | high | urgent | −1 | **wrong** | 0.992 | `[]` |

Five over-estimates to one under-estimate, and all five over-estimates sit on
tickets the model categorised correctly. It reads the category well and then
inflates urgency.

**The direction is the operationally important part.** Over-estimation is queue
noise — a ticket read earlier than it needed to be. Under-estimation is the
failure that costs something, because a ticket gets buried. There is exactly
one, off by a single level, and nothing lands two or more levels low.

### Why: my tone rule is asymmetric

The rubric tells the model that anger does not raise priority — T-012 shouts
"THIRD time" and is labeled `high`, not `urgent`, and the system gets it right.

But nothing tells it that an explicit *de*-prioritization cue should lower one.
T-003 contains the words "No rush" in the ticket body and was still classified
`medium` against a `low` label. **The model treats everything as consequential
until proven otherwise, and I gave it no way to prove otherwise.**

A second pattern runs through the over-estimates: organisational context reads
as urgency. T-009 is a routine invoice request that became `high` because "our
accounting team needs it" and a VAT number appear — bureaucratic vocabulary, not
urgency vocabulary. T-024 mentions "timeline is this quarter"; T-013 mentions a
colleague and a workspace.

### The confidence layer is blind to this entire failure mode

Every one of the six priority errors carries an **empty `confidence_signals`**
and sits between 0.97 and 0.99. The system made six confident errors and had no
mechanism to suspect any of them.

This is structural, not bad luck. None of the five signals looks at priority at
all — they watch the summary, the input, the category, the schema, and the
transport. I built doubt signals for category correctness and none for priority.

This is the content layer reappearing somewhere I had not instrumented: a
schema-valid, confidently-wrong answer that no parser and no rule can see.

---

## Design decisions

### The LLM is an untrusted component

Model output crosses a trust boundary before anything is stored, passing six
stations: transport → extract → parse → normalize → validate → policy.

Failures fall into four layers, and only three are catchable in code:

| Layer | What broke | Caught by |
|---|---|---|
| Transport | No response at all | Retry, then fallback |
| Syntactic | Response isn't parseable | Extraction, then repair retry |
| Semantic | Parses, but out of schema | Pydantic |
| **Content** | Schema-valid, wrong judgement | **Nothing in code — the eval and the human reviewer** |

The fourth layer is why this project has an evaluation harness and a review
queue rather than just a parser. For code to know that `security` is wrong for
T-019, it would have to understand the ticket — and if it understood the ticket,
the model wouldn't be here.

Worth stating plainly: once extraction and schema validation are in place, the
logs go quiet. **That is not evidence the system is right.** Fixing layer 2
hides layer 4 behind the appearance of success. T-019 parses, validates, and is
wrong; the six priority errors all pass every check at 0.97+.

### What I observed before designing anything

Before writing any validation, I ran the model against the hardest tickets with
a deliberately naive prompt — no system message, no `response_format`, no
schema, no examples — five times on the same input at `temperature=0`. Every
constraint in `app/schemas.py` traces back to something in this run.

- **0 of 8 runs produced parseable JSON.** Every response was prose preamble + a
  code fence + trailing commentary. Extraction is the normal path for this
  model, not an edge case.
- **`finish_reason` was `stop` on all 8.** Transport reported complete success
  on eight consecutive unusable responses. Response metadata is not a health
  signal; only parsing is.
- **`temperature=0` is not determinism.** Five identical calls on T-005: the
  first returned `account/medium`, runs 2–5 returned `other/low`
  byte-identically. The divergent one was the first call after model load. A
  single sample would have told me the model got this ticket right — it gets it
  wrong four times in five.
- **The sharpest failure was semantic.** Four of five T-005 runs argued `other`
  was correct because the ticket "doesn't fit into one of the predefined
  categories (billing, bug, feature_request, account, security)" — while
  `account` sits in the list they just recited. Fluent, confident, and wrong.
- **Self-reported confidence carries almost no signal.** 0.8 for a correct call,
  a wrong call, and a six-word ticket alike; 0 for the empty body; 100 —
  out of range entirely — for the injection.
- **Empty body means invention.** T-030 has no body. The model announced it
  would "create a sample JSON object", then classified it `feature_request` from
  the subject line, with `suggested_reply: null`.
- **Latency:** 17.3s cold, 2.7–7.0s warm.

### Two schemas, not one

`LLMTriageOutput` is what the model may claim. `TriageRecord` is what the system
stores. They are deliberately asymmetric, which one model with optional fields
cannot express:

- The model may **never** claim an empty summary. That is a validation error.
- The system must **always** be able to store one, because on a ticket with an
  empty body there is genuinely nothing to summarize.

A placeholder like "No summary available" would be text the system authored
implying knowledge the ticket does not contain. The empty string states the
absence honestly.

`TriageRecord` also separates `confidence_signals` — what was observed — from
`escalation_reason` — why it escalated. A ticket can carry doubt without
crossing the threshold, and that is worth keeping: T-004 records
`['body is only 4 words']` at 0.652, and at a threshold of 0.6 it would not have
escalated at all.

### Escalation is a system decision, not a model output

`escalate` is not a field the model is allowed to emit. In the Phase 0 run the
injected ticket set it to `false` — exactly what the attacker asked for. **A
field that does not exist cannot be injected.**

Even without an adversary it makes no sense on the model's side: the model does
not know the configured threshold, whether the call needed a repair retry, or
whether a rule signal conflicted with its own answer.

Unknown keys are dropped rather than rejected. All 8 Phase 0 responses emitted
`escalate` unprompted, so forbidding extras would fail every ticket, exhaust the
retry budget, and land everything in the fallback — which sets `escalate=true`
anyway. The protection comes from the system computing escalation itself, not
from refusing the key. `unexpected_fields()` reports what was dropped so the
habit stays measurable.

### Normalization of form, never of meaning

`normalize_llm_payload()` strips, lowercases, and maps underscores to spaces on
the two enum fields only. That reconciles `labels.json`'s `feature_request` with
the canonical `feature request` and accepts `"Billing"`.

It does no substring matching and maps no near-misses. `"not billing"` and
`"billing issue"` fail validation. Coercing `"not billing"` to `billing` would
convert a rejection the system can see and repair into a confident wrong answer
it cannot — and a wrong category is the most expensive error here, because a
human routes from it.

The eval loads labels through the same normaliser, so there is one
canonicalisation rule rather than two. A corrupted spelling fails loudly rather
than silently scoring three tickets wrong.

### One retry budget, never multiplied

Two kinds of retry exist and they must add rather than multiply:

- **transport retry** — the *same* messages re-sent, because nothing arrived
- **repair retry** — *different* messages, because what arrived was invalid

Nesting them gives (N+1)² calls per ticket, or 9 at N=2. Instead one per-ticket
budget is threaded through: `call_llm` takes `max_attempts` and reports
`attempts` consumed, and the caller spends what remains on repairs.

The `openai` SDK retries internally with `max_retries=2` by default. Left alone
that would multiply against my own policy, and the extra attempts would never
reach my logs, so my reported retry rate would be wrong. It is set to `0`.

The timeout bounds one attempt, not the sequence. Worst case per ticket is three
attempts at 60s, which across 30 tickets would be ~92 minutes — but that only
happens if the model is down, and that is a condition to stop for, not to
degrade through. Bounding it would convert a loud failure into a quiet one.

A truncated response is never repair-retried. `finish_reason="length"` means my
`max_tokens` cut it off, and re-sending truncates identically.

### The confidence score is a doubt detector, not a confidence score

25 of 30 records sit between 0.97 and 1.0. The score subtracts penalties from a
ceiling, so a 1.0 means **"I found no reason to doubt this"**, not "I am
confident this is right". Absence of a doubt signal is not evidence of
correctness.

Three of the five signals never fired:

| Signal | Penalty | Fired | Note |
|---|---|---|---|
| summary repeats the subject line | −0.25 | **0** | Retired by the prompt fixing the behaviour it detected — it fired 7 of 8 times in Phase 0 |
| body under 10 words | −0.25 | 3 | T-004, T-018, T-023 |
| security keyword conflict | −0.35 | **0** | Cannot fire — the model classifies every keyword-bearing ticket as security itself |
| needed a repair retry | −0.20 | 2 | T-008, T-023 |
| response was truncated | −0.30 | **0** | No truncation occurred at 512 tokens |

So the score is driven by two live signals and discriminates "did something
fire", not a gradient.

**T-015 shows a second limit of the keyword signal, and I found it rather than
designed for it.** T-015 is a phishing message — "Dear Winner, you have been
selected to receive a $500 gift card" with a link. It never uses the word
phishing, or scam, or any other term on my list, so the keyword rule would have
missed it completely. The model classified it `security`/`high` on its own and
it escalated. Keyword matching only catches security problems that describe
themselves, which is a real limit of rule-based signals over free text.

I want to be exact about what that does and does not show. The model catching it
was luck relative to my design, not behaviour I built or can rely on: T-015 is
unlabeled, it is a single example, and one success is not a measurement. It is
evidence about the rule's blind spot, not evidence about the model's reliability.

The model's own number is weighted at 0.15 — small but non-zero, because it does
carry signal at the extremes. It returned 0 on the empty body, 0 on T-023 (the
customer had already solved their problem), and 100 on the injection; on T-004
its reported 0.1 pulled the blended score down from 0.75 to 0.652, in the right
direction. In the middle it is worthless, which is where 0.8 covered a correct
call, a wrong call, and a six-word ticket alike.

Rule signals **lower** confidence and never overwrite the model's category. Two
systems overruling each other is worse than one flagging doubt.

### Choosing the threshold

The sweep is arithmetic over stored records — no model calls — and the
reconstruction reproduces all 30 stored `escalate` flags exactly.

| Threshold | Escalations |
|---|---|
| 0.50 | 5 |
| 0.60 | 5 |
| 0.70 | 7 |
| **0.80** | **8** |
| 0.85 | 8 |
| 0.90 | 8 |

**0.8 sits on a plateau, not an edge.** Nothing scores between 0.757 and 0.97,
so anything from 0.8 to 0.9 escalates the identical eight tickets. The
insensitivity is itself the finding: the metric is not graded, so the exact
placement doesn't matter much. I picked the low end of the plateau.

What 0.8 buys over 0.6 is T-004, T-008 and T-018 — the gibberish ticket, the
injection, and the six-word ticket all now reach a human. At 0.6 all three
cleared.

The value lives in `.env` and is left empty in `.env.example`, so nobody
inherits a threshold they did not choose. Which tickets reach a person is a
policy decision, not a technical default.

### `security` escalates unconditionally

Regardless of score. The labeled set has **zero** security examples, so category
accuracy is silent about the one category where a miss is worst — T-014 is a
genuine vulnerability disclosure. Escalating every security classification costs
three tickets out of thirty and means no vulnerability report is ever
auto-cleared on a confidence number I have no evidence to trust.

### Prompt injection: containment, not prevention

Prevention is not achievable with a 3B model, so the design contains it:

1. **Secrets never enter the prompt** — the strongest protection here, and why
   nothing leaked when the injection asked for API keys
2. Decision fields are enum-constrained, so an injection has nowhere to escape to
3. Ticket text is wrapped in delimiters and declared as data, and any delimiter
   the text itself contains is defanged before interpolation
4. Free-text renders with `textContent`, never `innerHTML`
5. Nothing is auto-sent, and a human sees every ticket

**I measured this rather than assuming it held.** T-008 asked for `priority: low`
and got it — enum constraints bound what the model can *say*, not which allowed
value it *picks*. At threshold 0.8 the ticket escalates on score, so a human sees
it; at 0.6 it did not. The worst realised outcome is a mis-prioritized ticket in
a queue a human is already reading, which is the blast radius I designed for, now
confirmed rather than asserted.

I did not add an injection detector. T-008 is the only injection in the set, so
any detector would be fitted to a single example — the same reason I rejected a
length-based short-circuit rule that separated T-004 from T-023 by seven
characters and two words.

### Human work is never overwritten

Re-running `scripts/run_triage.py` refreshes untouched pending rows with new
predictions, and **preserves any row a human has touched** — edited or decided.
A stale prediction is recoverable; a discarded reviewer edit is not.

Demonstrated: after editing one reply, approving one ticket and rejecting
another, a re-run reported `{'preserved': 3, 'updated': 27}`.

`suggested_reply` and `edited_reply` are separate columns and never merge. The
pair — what the model proposed next to what a person actually wrote — is a
feedback dataset that normal use produces for free.

Status transitions are constrained: only `pending` moves, and only to `approved`
or `rejected`. Re-deciding is refused with a 409, and undo is a 422 because
`pending` is not in the API's `Decision` enum at all — the conflict code is
reserved for conflicts with reality.

### What the labeled set influenced

Both the category boundaries and the priority rubric were derived from the
labeled subset. I wrote them as general rules rather than per-ticket answers,
and did not iterate wording against the score, but neither metric is a fully
clean held-out number.

The few-shot examples are synthetic. Longest run of consecutive words shared
between any example and any labeled ticket is 3; zero runs of 6 or more, and no
example text appears verbatim in any of the 30.

---

## Tests

17 cases, ~0.2s. Every one mocks the LLM at `app.llm_client.get_client` — the
first boundary I don't own — so the real retry loop, backoff, exception
classification and truncation detection all execute. None requires Ollama, a
database, or a `.env`; verified by running the suite from a temp directory
containing only `app/` and `tests/`, with `LLM_BASE_URL` pointed at a dead port
so any test reaching the network fails loudly rather than passing quietly.

These exist to answer one question: **how do I know the reliability layer
works?** In the 30-ticket run only one ticket fell back and two needed repair,
so most of that layer never executed on real data.

**My first budget test was vacuous, and I found it by mutation.** I edited
`triage.py` in a scratch copy so each repair requested a fresh full transport
budget — the exact multiplication the test exists to prevent — and all eight
tests still passed. Each script exercised only one of the two budgets, so
neither could observe them multiply. Adding a mixed script — one schema failure,
then transport failures inside the repair — makes the same mutation fail with
*expected 3 calls, got 4*. I don't trust a passing test I haven't seen fail.

One more test came out of a mistake in my own reasoning: I had assumed removing
`max_retries=0` would surface as extra calls, but patching `get_client` replaces
the function containing that line, so no mocked test at either seam can see it.
Hence a separate three-line assertion against the real, unpatched client.

---

## Limitations

- The labeled subset is 16 of 30 and contains **zero `security` examples**, so
  category accuracy says nothing about the category with the highest cost of
  error. Both security-relevant tickets — T-014 and T-008 — are unlabeled.
- At 16 labels, **one ticket is worth 6.25 percentage points**. I do not read
  differences smaller than one or two tickets as signal.
- The category rules were derived from the labeled set, so 93.75% is not a
  held-out number and the 14 unlabeled tickets cannot substitute for one.
- `temperature=0` removes intentional sampling randomness but does not guarantee
  identical output. Phase 0 showed one call in five diverging.
- The confidence score runs on two live signals out of five and **has no
  visibility into priority at all** — all six priority errors passed unflagged
  at 0.97+.
- Latency figures are specific to this machine: partial GPU offload, 32% CPU /
  68% GPU at an 8192 context.
- `static/tickets.json` is a copy of `data/tickets.json`. Mounting `data/` over
  HTTP would also serve `labels.json` — the eval ground truth — from the
  reviewer UI. The duplicate is safe because the file is fixed provided input
  that is never regenerated.
- The model is 4-bit quantized and 3B. Schema adherence is unreliable, which is
  precisely why the validation layer exists.

---

## What I deliberately did not build

The brief asked for sharp prioritization and clear notes on corners cut, so
these are choices rather than omissions.

| Not built | Why |
|---|---|
| RAG or a vector database | Nothing to retrieve. Triage is classification over the ticket text itself |
| Agents or tool calling | I want structured judgement, not execution. Tool calling would hand the model control flow, which is the thing I spent this project keeping away from it |
| A classical ML classifier | 16 labels across 6 classes, with zero in `security`. Any precision/recall from that is noise, and synthetic training data would mean training on the errors of the same untrusted model |
| Fine-tuning | Same data problem, plus the brief specifies the model |
| An ORM | One table. `sqlite3` plus seven functions |
| Docker, Kubernetes, Redis, Celery | A 30-row batch job |
| Auth | Single-reviewer local tool. Real deployment needs it; this doesn't |
| A frontend framework or build step | One page, served from the same origin. No npm, no CORS, no bundle |
| Database indexes | 30 rows scan faster than a B-tree traverses |
| An injection detector | T-008 is the only injection in the set; any detector would be fitted to one example |

---

## Next steps

**Fix the priority floor.** The rubric only guards against inflation. A ticket
that states no urgency, asks a how-to question, or requests a routine document
is `low`, and the model has no rule saying so — T-003 says "No rush" and was
still `medium`. This must be validated against tickets outside the 30, since
T-003, T-009 and T-013 are the evidence.

**Give the confidence layer visibility into priority.** Nothing currently
watches it, which is why six confident errors passed unflagged. A cheap first
version: flag disagreement between the assigned priority and simple textual
urgency cues, as a doubt signal only — never as an override.

**Teach the security exclusion by example rather than prose.** T-019 shows the
written rule doesn't hold. A few-shot example built from a scenario outside the
30 would, without tuning against the eval set.

**Label the remaining 14 tickets**, prioritising the security cases, so there is
finally a held-out number and so `security` stops being unmeasurable.

**Add self-consistency on ambiguous tickets.** Running the low-confidence subset
twice and comparing gives a doubt signal the model cannot influence. It costs
double the time on a small slice rather than on all 30.

**Version the prompt alongside eval runs**, so an accuracy change can be
attributed to a specific edit rather than inferred.

**Measure the few-shot examples rather than assuming they help.**
`build_messages(..., few_shot=[])` collapses to two messages, so the ablation is
one argument and one eval run.
