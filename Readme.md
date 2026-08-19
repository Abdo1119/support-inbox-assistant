# Support Inbox Assistant

First-pass triage for a support inbox. Each inbound ticket gets a category, a
priority, a one-line summary, a draft reply, and a confidence score, then lands
in a queue where a human reviews it.

Nothing is ever sent to a customer. There is no send path in the codebase.

> **Status: in progress.** Phases 0–5 complete — evidence gathering,
> configuration, schemas, prompts, transport, and the triage pipeline. Storage,
> API, frontend, tests, and the evaluation harness are not built yet. Sections
> marked TODO are unwritten, not omitted.

---

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
copy .env.example .env            # then fill it in
```

The app reads `.env` relative to the working directory, so run it from the repo
root.

Ollama must be running with the model pulled:

```bash
ollama pull llama3.2:3b
```

Two Ollama environment variables are set at user scope. Ollama reads them at
startup, so it must be fully quit and relaunched for a change to take effect —
on Windows that means quitting from the tray, not killing the process.

| Variable | Value | Why |
|---|---|---|
| `OLLAMA_CONTEXT_LENGTH` | `8192` | The runtime default is not something I control, and Ollama truncates from the front of the prompt silently — an undersized context would drop the system prompt and leave the model blind rather than broken |
| `OLLAMA_KEEP_ALIVE` | `30m` | The default unloads after 5 minutes idle; a cold reload costs 17.3s against a 4.6s warm call |

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `LLM_BASE_URL` | yes | OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1` |
| `LLM_MODEL` | yes | Model tag as the endpoint reports it, e.g. `llama3.2:3b` |
| `LLM_API_KEY` | yes | Ignored by Ollama; a real credential against a hosted provider |
| `LLM_TIMEOUT_SECONDS` | no | Seconds before a single attempt is abandoned |
| `LLM_MAX_TOKENS` | no | Upper bound on generated tokens per call |
| `LLM_MAX_RETRIES` | no | Retries *after* the first attempt |
| `CONFIDENCE_THRESHOLD` | yes | Final confidence below which a ticket escalates |

Defaults for the optional three live in `app/config.py`, which is the single
source of truth for them. The four required variables have no defaults, so no
endpoint, model name, credential, or policy value is embedded in tracked
source. Pointing this at a different provider is a `.env` edit, not a code
change.

`CONFIDENCE_THRESHOLD` is `0.8` in my `.env` and left empty in `.env.example`.
The justification lives in this README rather than in the template, so nobody
inherits a threshold they did not choose — see *Choosing the threshold*.

### Why `static/tickets.json` exists

It is a copy of `data/tickets.json`. The API serves `TriageRecord`, which
holds no ticket subject or body, and only `static/` is mounted — so the review
page has no other way to show a reviewer the customer's original message.

The alternative was mounting `data/` in `app/api.py`, which would also serve
`data/labels.json` — the evaluation ground truth — over HTTP from the reviewer
UI. The copy is deliberate, not an accident: `data/tickets.json` is fixed
provided input that is never regenerated, so the two cannot drift. It is
tracked in the repository because the page does not work without it.


## Running the eval

<!-- TODO: after the evaluation harness exists -->

---

## How it works

```
data/tickets.json
      |
      v
  triage pipeline          batch, offline
      |
      v
   storage                 triage + review status
      |
      +---> API ---> review queue     (human edits, approves, rejects)
      |
      +---> eval ---> eval/results.json
```

Triage is a batch step, not a request-time one. A 3B model takes several seconds
per ticket, so doing it on page load would make the review queue unusable.
Running it once and persisting the results also means the eval and the UI read
exactly the same predictions.

---

## Design decisions

### The LLM is an untrusted component

Model output crosses a trust boundary before anything is stored. It passes six
stations: transport, extraction, parsing, normalization, validation, policy.

Failures fall into four layers, and only three are catchable in code:

| Layer | What broke | Caught by |
|---|---|---|
| Transport | No response at all | Retry, then fallback |
| Syntactic | Response isn't parseable | Extraction, then repair retry |
| Semantic | Parses, but out of schema | Pydantic |
| **Content** | Schema-valid, wrong judgement | **Nothing in code — the eval and the human reviewer** |

The fourth layer is why this project has an evaluation harness and a review
queue rather than just a parser. For code to know that `other` is the wrong
category for a ticket, it would have to understand the ticket — and if it
understood the ticket, the model wouldn't be here.

A consequence worth stating plainly: once extraction and schema validation are
in place, the logs go quiet. That is not evidence the system is right. Fixing
layer 2 hides layer 4 behind the appearance of success. T-019 is the concrete
proof — it parses, it validates, and it is wrong.

### What I observed before designing anything

Before writing any validation, I ran the model against the hardest tickets with
a deliberately naive prompt — no system message, no `response_format`, no
schema, no examples — five times on the same input at `temperature=0`. The point
was to see the real failure surface rather than design against an imagined one.
Every constraint in `app/schemas.py` traces back to something in this run.

- **0 of 8 runs produced parseable JSON.** Every response was prose preamble + a
  code fence + a trailing paragraph of unrequested commentary. `json.loads` died
  identically each time at character 0. Extraction is the normal path for this
  model, not an edge case.
- **`finish_reason` was `stop` on all 8.** The transport layer reported complete
  success on eight consecutive unusable responses. Response metadata is not a
  health signal; only parsing is.
- **`temperature=0` is not determinism.** Five identical calls on T-005: the
  first returned `account/medium`, runs 2–5 returned `other/low`
  byte-identically. The divergent one was the first call after model load. A
  single sample would have told me the model got this ticket right — it gets it
  wrong four times out of five.
- **The most interesting failure was semantic, not syntactic.** Four of five
  T-005 runs argued that `other` was correct because the ticket "doesn't fit
  into one of the predefined categories (billing, bug, feature_request, account,
  security)" — while `account` sits in the list it just recited. The human label
  is `account`. The model reasons itself out of a valid enum and then justifies
  it fluently. Fluency is not correctness, and no parser will ever see this.
- **Self-reported confidence carries almost no signal.** `0.8` for a correct
  call, `0.8` for a wrong one, `0.8` for a six-word ticket. It returned `0` on
  the empty-body ticket and `100` — outside the 0–1 range entirely — on the
  injection. It discriminates at the two extremes my code already catches
  deterministically, and not in the range where I would need it.
- **Empty body means invention.** T-030 has no body. The model announced it
  would "create a sample JSON object", then classified it `feature_request` from
  the subject line alone, with `suggested_reply: null`.
- **The injection was contained by the constrained fields.** T-008 asked for the
  system prompt, API keys, and `priority: low` / resolved. It got the disposition
  it asked for, and its fabricated `YOUR_API_KEY_HERE` landed in
  `suggested_reply`. No real secret leaked, because none was in the prompt to
  leak. `category` stayed inside the allowed set.
- **Latency:** 17.3s on the first call after model load, 2.7–7.0s once warm.

### Two schemas, not one

`app/schemas.py` defines `LLMTriageOutput` — what the model is permitted to
claim — and `TriageRecord` — what the system stores. They are deliberately
asymmetric, which a single model with optional fields cannot express:

- The model may **never** claim an empty summary. That is a validation error.
- The system must **always** be able to store one, because on a ticket with an
  empty body there is genuinely nothing to summarize.

Writing a placeholder like "No summary available" would be text the system
authored implying knowledge the ticket does not contain. The empty string states
the absence honestly, and `escalation_reason` carries the explanation.

`TriageRecord` also separates `confidence_signals` — what the system observed —
from `escalation_reason` — why it escalated. A ticket can carry doubt without
crossing the threshold, and that observation is worth keeping: T-004 scores
0.652 with `['body is only 4 words']` recorded, and at a threshold of 0.6 it
would not have escalated at all.

### Escalation is a system decision, not a model output

`escalate` is not a field the model is allowed to emit. In the Phase 0 run the
injected ticket set it to `false` — exactly what the attacker asked for. A field
that does not exist cannot be injected.

Even without an adversary the field makes no sense on the model's side: the
model does not know the configured threshold, whether the call needed a repair
retry, or whether a rule signal conflicted with its own answer. The model
produces signals; the policy layer turns signals into actions.

Unknown keys are dropped rather than rejected. All 8 Phase 0 responses emitted
`escalate` unprompted, so forbidding extras would fail every ticket, exhaust the
retry budget, and land everything in the fallback — which sets `escalate=true`
anyway. The protection comes from the system computing escalation itself, not
from refusing the key. `unexpected_fields()` reports what was dropped so the
habit stays measurable rather than silent.

### Normalization of form, never of meaning

`normalize_llm_payload()` strips whitespace, lowercases, and maps underscores to
spaces on the two enum fields only. That reconciles `labels.json`'s
`feature_request` with the canonical `feature request` and accepts `"Billing"`
as `billing`.

It does no substring matching and maps no near-misses. `"not billing"` and
`"billing issue"` pass through unchanged and fail validation. Coercing
`"not billing"` to `billing` would convert a rejection the system can see and
repair into a confident wrong answer it cannot — and a wrong category is the
most expensive error here, because a human routes from it.

### Minimum lengths sit below the shortest real output

`summary >= 5` and `suggested_reply >= 20`, both below the shortest genuine
output observed (11 and 62 characters).

A floor above the real minimum would reject valid output, burn a repair retry,
and end at the fallback — which replaces the summary with a truncated ticket
body. Trading a mediocre summary for a truncated body makes the record worse, so
these floors reject absence dressed as presence (`n/a`, `none`, `TBD`, `-`)
rather than mediocrity.

What they do not catch: 7 of 8 Phase 0 runs returned the ticket subject verbatim
as the summary, all of them 11 characters or more. That is a content failure,
invisible to a length con straint.

### Category boundaries as decision rules

The definitions are rules for deciding, not topic labels. The one that does the
most work:

- **billing** — money movement: charges, refunds, invoices, and subscription
  changes that start or stop payment
- **account** — access and identity: signing in, permissions, members, and the
  customer's own data being exported or deleted

Checked against all six labeled tickets in those classes. T-017 ("cancel my
subscription") is the one that tests it: it reads like an account request, but
it is labeled billing because the intent is to stop being charged. Category
follows intent, not the noun in the message.

`security` carries an explicit exclusion: a security-adjacent thing that is
merely malfunctioning — a check that fails, a certificate that expired — is a
bug. It is security only when someone could actually reach data or take action
they should not.

### What the labeled set influenced

Both the category boundaries and the priority rubric were derived from the
labeled subset — the billing/account split from the six labeled tickets in those
classes, the blocking tie-breaker from T-005, and the tone rule from T-012. I
wrote them as general rules rather than per-ticket answers, and did not iterate
wording against the score, but neither metric is a fully clean held-out number
and I'd rather say so than imply otherwise.

The few-shot examples are synthetic. The longest run of consecutive words shared
between any example and any of the 16 labeled tickets is 3; zero runs of 6 or
more, and no example text appears verbatim in any of the 30.

### One retry budget, never multiplied

Two kinds of retry exist and they must add rather than multiply:

- **transport retry** — the *same* messages re-sent, because nothing arrived
- **repair retry** — *different* messages, because what arrived was invalid

Nesting them would give (N+1)² calls per ticket, or 9 at N=2. Instead one
per-ticket budget is threaded through: `call_llm` takes `max_attempts` and
reports `attempts` consumed, and the caller spends what remains on repairs.
Verified with a mocked client — the sequence is `[3, 2, 1]` across a call and two
repairs, never exceeding 3 model calls per ticket.

The `openai` SDK retries internally with `max_retries=2` by default. Left alone
that would multiply against my own policy, and the extra attempts would never
reach my logs, so my reported retry rate would be wrong. It is set to `0` so
there is exactly one retry policy in the system.

The timeout bounds one attempt, not the sequence. Worst case per ticket is three
attempts at 60s, which across 30 tickets would be about 92 minutes — but that
only happens if the model is down, and that is a condition to stop for, not to
degrade through. A hung run is obvious within a minute. Bounding it would
convert a loud failure into a quiet one.

A truncated response is never repair-retried. `finish_reason="length"` means my
`max_tokens` cut it off, and re-sending truncates identically — a deterministic
failure, and retrying a deterministic failure spends budget on no chance of
success.

### The confidence score is a doubt detector, not a confidence score

25 of 30 records sit between 0.97 and 1.0. The score subtracts penalties from a
ceiling, and on most tickets nothing fires, so a 1.0 means "I found no reason to
doubt this" — not "I am confident this is right". Absence of a doubt signal is
not evidence of correctness, and I'd rather name the thing accurately than imply
a precision it doesn't have.

Three of the five signals never fired at all:

| Signal | Penalty | Fired | Note |
|---|---|---|---|
| summary repeats the subject line | −0.25 | **0** | Retired by the prompt fixing the behaviour it detected — 7 of 8 in Phase 0 |
| body under 10 words | −0.25 | 3 | T-004, T-018, T-023 |
| security keyword conflict | −0.35 | **0** | Cannot fire — the model classifies every keyword-bearing ticket as security itself |
| needed a repair retry | −0.20 | 2 | T-008, T-023 |
| response was truncated | −0.30 | **0** | No truncation occurred at 512 tokens |

So the score is driven by two live signals, and its discrimination is "did
something fire", not a gradient. The model's own number is weighted at 0.15 —
small but non-zero, because it does carry signal at the extremes: on T-004 its
reported 0.1 pulled the blended score down from 0.75 to 0.652, moving it in the
right direction, and it returned 0 on the empty body in Phase 0. In the middle
it is worthless, which is where 0.8 covered a correct call, a wrong call, and a
six-word ticket alike.

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
placement doesn't matter much, and I picked the low end of the plateau.

What 0.8 buys over 0.6 is T-004, T-008 and T-018 — the gibberish ticket, the
injection, and the six-word ticket all now reach a human. At 0.6 all three
cleared.

This also shows why triage is a batch step: the threshold is a post-processing
decision, so sweeping six values costs arithmetic rather than 180 model calls.

### `security` escalates unconditionally

Regardless of score. The labeled set has zero security examples, so category
accuracy is silent about the one category where a miss is worst — T-014 is a
genuine vulnerability disclosure. Escalating every security classification costs
three tickets out of thirty and means no vulnerability report is ever
auto-cleared on a confidence number I have no evidence to trust.

### Prompt injection: containment, not prevention

Prevention is not achievable with a 3B model, so the design contains it instead:

1. Secrets never enter the prompt — the strongest protection here, and the
   reason nothing leaked when the injection asked for API keys
2. Decision fields are enum-constrained, so an injection has nowhere to escape to
3. Ticket text is wrapped in delimiters and declared as data, and any delimiter
   the text itself contains is defanged before interpolation
4. Free-text fields render as text, never as HTML
5. Nothing is auto-sent, and a human sees every ticket

**I measured this rather than assuming it held.** T-008 asked for priority low
and got it — enum constraints bound what the model can *say*, not which allowed
value it *picks*. At threshold 0.8 the ticket escalates on score, so a human
sees it; at 0.6 it did not. The worst realised outcome is a mis-prioritized
ticket sitting in a queue a human reads, which is the blast radius I designed
for, now confirmed rather than asserted.

I did not add an injection detector. T-008 is the only injection in the set, so
any detector would be fitted to a single example — the same reason I rejected a
length-based short-circuit rule that separated T-004 from T-023 by seven
characters and two words.

---

## Measured on the full 30

Configuration in effect: `llama3.2:3b`, `CONFIDENCE_THRESHOLD=0.8`,
`LLM_MAX_RETRIES=2`, `LLM_MAX_TOKENS=512`, `LLM_TIMEOUT_SECONDS=60`.
Hardware: partial GPU offload, 32% CPU / 68% GPU at
`OLLAMA_CONTEXT_LENGTH=8192` — raising the context enlarges the KV cache and
pushes more onto CPU, so latency figures are specific to this machine.

| | |
|---|---|
| Runtime | 138s total, 4.6s per ticket |
| Fallbacks | 1 / 30 |
| Repair retries | 2 / 30 |
| Escalations | 8 / 30 — 4 hard-triggered, 4 score-triggered |
| Hard-triggered | T-014, T-015, T-019, T-030 |
| Score-triggered | T-004, T-008, T-018, T-023 |
| Confidence ≥ 0.97 | 25 / 30 |
| Confidence < 0.8 | 5 / 30 |

**Category distribution across all 30:** bug 8, billing 5, feature request 5,
other 5, account 4, security 3.

**Priority distribution across all 30:** high 16, medium 9, low 4, urgent 1 —
against a labeled distribution of 5 high, 6 medium, 3 low, 2 urgent.

More than half the tickets landed on `high` where the labeled share is under a
third, while `urgent` came out *under*-represented. The model clusters in the
upper middle without reaching the top, which is priority inflation in the literal
sense: if half the queue is high, nothing is. This will show up in
`priority_agreement` once the eval computes it.

### Token budget

| | |
|---|---|
| System prompt | ~711 tokens (estimated) |
| Few-shot examples | ~664 tokens (estimated) |
| Full request, measured | 1313 tokens across 10 messages |
| Plus 512 generation | 1825 against a context of 8192 — 22% used |

The component rows are `chars/4` estimates and ran about 11% high against the
measured total, which is why they don't sum. Ollama caches the shared
system+few-shot prefix, so only the first call after a prompt edit pays for it:
10.1s cold against a 4.6s mean.

### Transport verification

| Condition | Result |
|---|---|
| Bad model name | `NOT_FOUND`, `retryable=False`, 1 attempt of 3 — no budget spent where success is impossible |
| Dead port | `CONNECTION`, all 3 attempts, 15.3s including 1s + 2s backoff |
| Empty message list | `EMPTY_MESSAGES` at 0 attempts, no HTTP call made |
| `max_tokens=8` | `finish_reason=length`, `truncated=True`, `ok=True`, partial content retained |
| Log hygiene | 18 log lines captured containing no API key, no ticket body, no subject |

Extraction recovers all 8 Phase 0 raw responses — the ones that were 0 of 8
parseable end to end — with first-brace-to-last-brace and no regex.

---

## Known issues

### T-019: a miss I chose not to fix

T-019 (HMAC signature verification failing) is labeled `bug`. My system
classified it `security` at 0.992 confidence, and the security hard trigger
escalated it — so it costs a category point *and* a review.

The prose exclusion in the system prompt — "a check that fails is a bug" — did
not hold against it. I could fix this by tightening the prompt against that
ticket, but T-019 is one of the 16 labeled tickets. Tuning against it would make
my accuracy measure memorisation rather than generalization, which is exactly
what the brief warns about. The right fix is a few-shot example teaching the
exclusion, built from a scenario outside the 30.

### The security keyword conflict is a soft signal, not a hard trigger

My approved plan listed the keyword conflict among the hard escalation triggers.
The implementation has only `category == security` and `truncated`; the conflict
is a −0.35 penalty and nothing more. It fired 0 times, so no reported number
changes either way, but the code does not match the plan and I would rather
record that than quietly reconcile it.

The signal is also structurally unable to fire as written: it triggers when
security keywords are present *and* the category is not `security`, but the model
classifies every keyword-bearing ticket as `security` on its own — which is the
same over-triggering that produced the T-019 miss.

### The keyword list only catches self-describing security problems

T-015 is phishing but never says so. It was caught by the model rather than by
my keyword list — the blind spot I flagged was covered from the other direction,
which is luck rather than design.

---

## Results

<!-- TODO: after the eval computes category_accuracy and priority_agreement -->

## Error analysis

<!-- TODO: after the eval -->

---

## Limitations

- The labeled subset is 16 of 30 tickets and contains **zero `security`
  examples**, so category accuracy says nothing at all about the category with
  the highest cost of error. The two security-relevant tickets — T-014, a
  vulnerability disclosure, and T-008, the injection — are both unlabeled.
- At 16 labels, **one ticket is worth 6.25 percentage points**. I do not read
  differences smaller than one or two tickets as signal.
- A majority-class baseline scores **37.5% on both metrics** (`bug` and
  `medium`). Any accuracy figure should be read against that, not against zero.
- `temperature=0` removes intentional sampling randomness but does not guarantee
  identical output across runs. Phase 0 showed one call in five diverging, and
  the divergent one was the first after model load.
- The model is a 4-bit quantized 3B model — small and lossy. Schema adherence is
  unreliable, which is precisely why the validation layer exists.
- The confidence score is driven by two live signals out of five, so it
  discriminates far less than the 0–1 range suggests.
- `.env` is resolved relative to the working directory, so the app must be run
  from the repo root.

## What I deliberately did not build

<!-- TODO -->

## Next steps

<!-- TODO -->
