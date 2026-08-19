# Support Inbox Assistant

First-pass triage for a support inbox. Each inbound ticket gets a category,
a priority, a one-line summary, a draft reply, and a confidence score, then
lands in a queue where a human reviews it.

Nothing is ever sent to a customer. There is no send path in the codebase.

> **Status: in progress.** Phases 0–3 complete (evidence gathering,
> configuration, schemas, prompts). The triage pipeline, API, frontend, and eval are
> not built yet. Sections below marked TODO are unwritten, not omitted.

---

## Setup

<!-- TODO: fill in after Phase 6, once there is something to run -->

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
cp .env.example .env              # then fill it in
```

The app reads `.env` relative to the working directory, so run it from the
repo root.

### Environment variables

| Variable               | Required | Purpose                                                        |
| ---------------------- | -------- | -------------------------------------------------------------- |
| `LLM_BASE_URL`         | yes      | OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1`   |
| `LLM_MODEL`            | yes      | Model tag as the endpoint reports it, e.g. `llama3.2:3b`       |
| `LLM_API_KEY`          | yes      | Ignored by Ollama; a real credential against a hosted provider |
| `LLM_TIMEOUT_SECONDS`  | no       | Seconds before a single call is abandoned                      |
| `LLM_MAX_TOKENS`       | no       | Upper bound on generated tokens per call                       |
| `LLM_MAX_RETRIES`      | no       | Retries _after_ the first attempt                              |
| `CONFIDENCE_THRESHOLD` | yes      | Final confidence below which a ticket escalates                |

Defaults for the optional three live in `app/config.py`, which is the single
source of truth for them. The four required variables have no defaults, so no
endpoint, model name, credential, or policy value is embedded in tracked
source. Pointing this at a different provider is a `.env` edit, not a code
change.

### Ollama server settings

`OLLAMA_CONTEXT_LENGTH=8192` and `OLLAMA_KEEP_ALIVE=30m` are set as user
environment variables.

Ollama reads them at startup, so it must be fully quit and relaunched for a
change to take effect — on Windows that means quitting from the tray, not
killing the process.

## Running the eval

<!-- TODO: after Phase 7 -->

---

## Design decisions

### The LLM is an untrusted component

Model output crosses a trust boundary before anything is stored. It passes six
stations: transport, extraction, parsing, normalization, validation, policy.

Failures fall into four layers, and only three of them are catchable in code:

| Layer       | What broke                    | Caught by                                             |
| ----------- | ----------------------------- | ----------------------------------------------------- |
| Transport   | No response at all            | Retry, then fallback                                  |
| Syntactic   | Response isn't parseable      | Extraction, then retry                                |
| Semantic    | Parses, but out of schema     | Pydantic                                              |
| **Content** | Schema-valid, wrong judgement | **Nothing in code — the eval and the human reviewer** |

The fourth layer is why this project has an evaluation harness and a review
queue rather than just a parser. For code to know that `other` is the wrong
category for a ticket, it would have to understand the ticket — and if it
understood the ticket, the model wouldn't be here.

A consequence worth stating plainly: once extraction and schema validation are
in place, the logs go quiet. That is not evidence the system is right. Fixing
layer 2 hides layer 4 behind the appearance of success.

### What I observed before designing anything

Before writing any validation, I ran the model against the hardest tickets with
a deliberately naive prompt — no system message, no `response_format`, no
schema, no few-shot examples — five times on the same input at `temperature=0`.
The point was to see the real failure surface rather than design against an
imagined one. Every constraint in `app/schemas.py` traces back to something in
this run.

- **0 of 8 runs produced parseable JSON.** Every response was prose preamble +
  a code fence + a trailing paragraph of unrequested commentary. `json.loads`
  died identically each time at character 0. Extraction is the normal path for
  this model, not an edge case to handle later.
- **`finish_reason` was `stop` on all 8.** The transport layer reported
  complete success on eight consecutive unusable responses. Response metadata
  is not a health signal; only parsing is.
- **`temperature=0` is not determinism.** Five identical calls on T-005: the
  first returned `account/medium`, runs 2–5 returned `other/low`
  byte-identically. The divergent one was the first call after model load. A
  single sample would have told me the model got this ticket right — it gets it
  wrong four times out of five.
- **The most interesting failure was semantic, not syntactic.** Four of five
  T-005 runs argued that `other` was correct because the ticket "doesn't fit
  into one of the predefined categories (billing, bug, feature_request,
  account, security)" — while `account` sits in the list it just recited. The
  human label is `account`. The model reasons itself out of a valid enum and
  then justifies it fluently. Fluency is not correctness, and no parser will
  ever see this.
- **Self-reported confidence carries almost no signal.** `0.8` for a correct
  call, `0.8` for a wrong one, `0.8` for a six-word ticket. It returned `0` on
  the empty-body ticket and `100` — outside the 0–1 range entirely — on the
  injection. It discriminates at the two extremes my code already catches
  deterministically, and not in the range where I would need it.
- **Empty body means invention.** T-030 has no body. The model announced it
  would "create a sample JSON object", then classified it `feature_request`
  from the subject line alone, with `suggested_reply: null`. This ticket must
  never reach the model.
- **The injection was contained by the constrained fields.** T-008 asked for
  the system prompt, API keys, and `priority: low` / resolved. It got the
  disposition it asked for — `priority: low`, `escalate: false` — and its
  fabricated `YOUR_API_KEY_HERE` landed in `suggested_reply`. No real secret
  leaked, because none was in the prompt to leak. `category` stayed inside the
  allowed set. Free-text fields are where injection lives; constrained fields
  held.
- **Latency:** 17.3s on the first call after model load, 2.7–7.0s once warm
  (mean 5.3s across seven steady-state calls).

### Two schemas, not one

`app/schemas.py` defines `LLMTriageOutput` — what the model is permitted to
claim — and `TriageRecord` — what the system stores. They are deliberately
asymmetric, which a single model with optional fields cannot express:

- The model may **never** claim an empty summary. That is a validation error.
- The system must **always** be able to store one, because on a ticket with an
  empty body there is genuinely nothing to summarize.

Writing a placeholder like "No summary available" would be text the system
authored implying knowledge the ticket does not contain. The empty string
states the absence honestly, and `escalation_reason` carries the explanation.

### Escalation is a system decision, not a model output

`escalate` is not a field the model is allowed to emit. In the Phase 0 run the
injected ticket set it to `false` — exactly what the attacker asked for. A
field that does not exist cannot be injected.

Even without an adversary the field makes no sense on the model's side: the
model does not know the configured threshold, whether the call needed a repair
retry, or whether a rule signal conflicted with its own answer. The model
produces signals; the policy layer turns signals into actions.

Unknown keys are dropped rather than rejected. All 8 Phase 0 responses emitted
`escalate` unprompted, so forbidding extras would fail every ticket, exhaust
the retry budget, and land everything in the fallback — which sets
`escalate=true` anyway. The protection comes from the system computing
escalation itself, not from refusing the key. `unexpected_fields()` reports
what was dropped so the habit stays measurable rather than silent.

### Normalization of form, never of meaning

`normalize_llm_payload()` strips whitespace, lowercases, and maps underscores
to spaces on the two enum fields only. That reconciles `labels.json`'s
`feature_request` with the canonical `feature request` and accepts `"Billing"`
as `billing`.

It does no substring matching and maps no near-misses. `"not billing"` and
`"billing issue"` pass through unchanged and fail validation. Coercing
`"not billing"` to `billing` would convert a rejection the system can see and
repair into a confident wrong answer it cannot — and a wrong category is the
most expensive error here, because a human routes from it.

Verified: `"not billing"` is rejected; `"  Billing  "` normalizes to `billing`;
`feature_request` normalizes to `feature request`.

### Minimum lengths sit below the shortest real output

`summary >= 5` and `suggested_reply >= 20`, both below the shortest genuine
output observed (11 and 62 characters respectively).

A floor above the real minimum would reject valid output, burn a repair retry,
and end at the fallback — which replaces the summary with a truncated ticket
body. Trading a mediocre summary for a truncated body makes the record worse,
so these floors reject absence dressed as presence (`n/a`, `none`, `TBD`, `-`)
rather than mediocrity.

What they do not catch: 7 of 8 runs returned the ticket subject verbatim as the
summary, all of them 11 characters or more. That is a content failure, invisible
to a length constraint, and belongs with the confidence signals.

### Canonical category spelling

The brief spells it `feature request`; `labels.json` spells it
`feature_request`. The canonical internal form is the brief's, because
`eval/results.json` is the graded artefact and is compared against the brief —
so the canonical form matches the output form and the final write needs no
conversion. A conversion that does not exist cannot be the one that is
forgotten.

Conversion happens in two inbound places only: `normalize_llm_payload()` and
label loading in the eval. Nothing outbound converts.

### Prompt construction

`app/prompts.py` holds the system prompt, the ticket wrapper, and four
synthetic few-shot examples delivered as user/assistant message turns.

| Component                  | Approx. tokens |
| -------------------------- | -------------- |
| System prompt              | ~711           |
| Few-shot examples (4)      | ~664           |
| Full request (10 messages) | ~1418          |

With 512 generation tokens that is ~1930 against a context of 8192 — 24%
used, 76% headroom. Verified with `ollama ps` rather than assumed.

Ollama truncates from the **front** of the prompt silently, so an undersized
context would drop the system prompt and leave the model blind rather than
broken. The number was verified rather than assumed.

The latency figures and the retry budget derived from them were measured on a
partial GPU offload (20% CPU / 80% GPU), and are specific to this machine.

Leakage verification: the longest run of consecutive words shared between any
few-shot example and any of the 16 labeled tickets is 3; there are zero runs
of 6 or more. No example text appears verbatim in any of the 30 tickets.

`build_messages(..., few_shot=[])` collapses to 2 messages, so the examples
can be ablated and their effect measured rather than assumed.

### What the labeled set influenced

Both the category boundaries and the priority rubric were derived
from the labeled subset — the billing/account split from the six
labeled tickets in those classes, the blocking tie-breaker from
T-005, and the tone rule from T-012. I wrote them as general rules
rather than per-ticket answers, and did not iterate wording against
the score, but neither metric is a fully clean held-out number and
I'd rather say so than imply otherwise.

### Confidence

<!-- TODO: after Phase 3, once the signals exist -->

### Prompt injection: containment, not prevention

<!-- TODO -->

---

## Results

<!-- TODO: after Phase 7 -->

## Error analysis

<!-- TODO: after Phase 7 -->

---

## Limitations

- The labeled subset is 16 of 30 tickets and contains **zero `security`
  examples**, so category accuracy says nothing at all about the category with
  the highest cost of error. The two security-relevant tickets (T-014, a
  vulnerability disclosure; T-008, the injection) are both unlabeled.
- At 16 labels, **one ticket is worth 6.25 percentage points**. I do not read
  differences smaller than one or two tickets as signal.
- `temperature=0` removes intentional sampling randomness but does not
  guarantee identical output across runs. Phase 0 showed one call in five
  diverging, and the divergent one was the first after model load.
- The model is a 4-bit quantized 3B model — small and lossy. Schema adherence
  is unreliable, which is precisely why the validation layer exists.
- `.env` is resolved relative to the working directory, so the app must be run
  from the repo root.

## What I deliberately did not build

<!-- TODO -->

## Next steps

<!-- TODO -->

### Category boundaries as decision rules

The definitions are written as rules for deciding, not as topic labels.
The one that does the most work:

- **billing** = money movement — charges, refunds, invoices, and
  subscription changes that start or stop payment
- **account** = access and identity — signing in, permissions, members,
  and the customer's own data being exported or deleted

Checked against all six labeled tickets in those two classes. T-017
("cancel my subscription") is the one that tests it: it reads like an
account request, but it is labeled billing because the intent is to stop
being charged. Category follows intent, not the noun in the message.

`security` carries an explicit exclusion: a security-adjacent thing that
is merely malfunctioning — a check that fails, a certificate that
expired — is a bug. It is security only when someone could actually
reach data or take action they should not. Without that exclusion the
model would over-trigger on security vocabulary appearing in ordinary
defects.
