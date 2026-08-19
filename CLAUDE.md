# Support Inbox Assistant — Working Agreement

## Context

Two-day take-home. A support team is drowning in inbound messages. This app
does **first-pass triage** on 30 tickets and puts them in a review queue for a
human. It never sends anything to a customer.

The model is `llama3.2:3b` running locally through Ollama's OpenAI-compatible
endpoint. It is small, 4-bit quantized, and unreliable by design — the whole
point of the exercise is how the system behaves when the model fails.

**This repo will be walked through live in an interview.** I have to be able to
defend every line. Code I cannot explain is worse than code that does not exist.

---

## How to work with me

- **One component at a time.** Never scaffold the whole project. Never write the
  next file because it seems like the obvious next step — wait until I ask.
- **Before writing code, compare.** List the realistic alternatives, say what
  each trades off, recommend one, and explain why. Then write it.
- **Explain non-obvious lines.** If a line needs a comment to be understood by
  someone reading it cold, write the comment.
- **No new dependency without justification** against the standard library and
  against what is already installed.
- **Boring and readable beats clever.** No metaprogramming, no decorators I did
  not ask for, no abstraction layers with one implementation.
- **Small, honest commits** with messages that say why, not what.
- If I ask for something that contradicts this file, say so before doing it.

---

## Architecture decisions (already made — do not relitigate)

| Decision | Reason |
|---|---|
| Triage runs as a **batch step**, not at request time | 10–20s per ticket on CPU; 30 tickets would make the UI unusable. Also keeps eval and UI reading identical predictions. |
| **FastAPI** backend | Pydantic is native to it — the same schema validates LLM output and documents the API. |
| **SQLite** for storage | Stdlib, no install. There are writes (approve/reject/edit), and a single JSON file corrupts under concurrent writes. |
| **Single static HTML** frontend served by FastAPI | No npm, no build step, no CORS. The brief says "functional over fancy"; the time saved goes to eval and README. |
| **`openai` SDK** as the LLM client | It is the reference implementation of the OpenAI-compatible contract. Swapping providers is a config change, not a code change. |
| **No send endpoint exists anywhere** | Safety enforced by absence, not by a runtime check. |

---

## Hard constraints

**Never add:** RAG, vector databases, embeddings, agents, multi-agent,
LangChain, LlamaIndex, fine-tuning, Docker, Kubernetes, Redis, Celery,
authentication, RBAC, WebSockets, a design system, or a CSS framework beyond
plain CSS.

**Never do:**
- `except Exception: pass` — silent failure is the worst outcome in this project.
- Substring matching to coerce an enum (`"not billing"` must not become `billing`).
- Compute any metric with the LLM. Metrics are Python comparing strings.
- Hardcode `LLM_BASE_URL`, `LLM_MODEL`, or `LLM_API_KEY`.
- Commit `.env`.
- Let the model generate a summary in the fallback path — truncate the ticket
  body instead. Failure means claiming less, not inventing a substitute.
- Retry a deterministic failure (`model not found`, empty ticket body, prompt
  over the context window).

---

## The reliability model

LLM output is **untrusted input**. Every response passes six stations:

```
transport -> extract -> parse -> normalize -> validate -> policy
```

Four layers of failure, and only three are catchable in code:

1. **Transport** — no response at all (timeout, connection refused) → retry, then fallback.
2. **Syntactic** — not parseable (fences, prose around it, truncated) → extract, then retry.
3. **Semantic** — parses but out of schema (bad enum, out-of-range confidence) → retry **with the validation error as feedback**.
4. **Content** — schema-valid but the judgement is wrong → **not catchable in code.** This is what the eval and the human reviewer exist for.

Retries: **one or two maximum**, with backoff. 30 tickets × 3 attempts × 15s is
a 22-minute eval run, and the eval has to be re-runnable while I iterate.

Fallback result: `category=other`, `priority=medium`, `confidence=0.0`,
`escalate=true`, with a recorded reason. `medium` rather than `low` because
`low` means "defer" and we do not actually know.

---

## Confidence

The model's self-reported confidence is **not a measurement** — it is a
generated token shaped like a number, and it is poorly calibrated on a 3B model.
Do not escalate on it alone.

Final confidence combines signals the model cannot influence:

- input quality (empty body, very short, gibberish)
- rule conflicts (security keywords present but category is not `security`)
- schema health (needed a repair retry? fell back?)
- the model's own number, weighted low
- optionally, self-consistency across two runs on ambiguous tickets

Rule signals **lower confidence**; they never overwrite the model's decision.
Two systems overruling each other is worse than one system flagging doubt.

Escalation should carry a **reason string**, not just a boolean.

---

## Prompt injection

Containment, not prevention — prevention is not achievable with a 3B model.

- Secrets never enter the prompt.
- Decision fields (`category`, `priority`, `confidence`) are enum/range-constrained,
  so an injection has nowhere to escape to.
- Ticket text is wrapped in explicit delimiters and declared as data.
- Free-text fields render as text, never as HTML (`textContent`, not `innerHTML`;
  never `dangerouslySetInnerHTML`).
- Nothing is auto-sent; a human sees every ticket.

Worst realistic outcome: a mis-prioritized ticket in a queue a person is already reading.

---

## Data gotchas

- `labels.json` uses `feature_request` (underscore). The brief says
  `feature request` (space). **Normalize at the boundary** or every ticket in
  that class scores as wrong.
- The labeled subset has **zero `security` examples**, so category accuracy says
  nothing about the highest-cost category. Say this in the README.
- Only **16 of 30** tickets are labeled. Metrics are computed over those 16.
  `predictions` must contain **all 30**.
- Majority-class baseline is **37.5%** for both metrics (`bug` and `medium`).
  Report it next to my numbers, or my numbers mean nothing.
- 16 labels means **one ticket is worth 6.25 points**. Do not read small
  differences as improvements.
- `T-030` has an empty body and `T-004` is gibberish — short-circuit these
  before any LLM call.
- `T-008` is a prompt injection. `T-015` is phishing. `T-014` is a real
  vulnerability disclosure. `T-023` is already resolved by the customer.

---

## Evaluation

`eval/results.json` must match the requested shape exactly:

```json
{
  "metrics": { "category_accuracy": 0.0, "priority_agreement": 0.0 },
  "predictions": [
    { "id": "T-001", "category": "billing", "priority": "high",
      "summary": "...", "suggested_reply": "...",
      "confidence": 0.9, "escalate": false }
  ]
}
```

Extra metric fields are welcome and expected: baselines, within-one-level
priority agreement, over- vs under-estimation counts, confusion matrix,
escalation and fallback counts, labeled vs total counts.

Report numbers honestly. Never tune the prompt against the labeled subset and
then report the result as generalization — that is the exact thing the brief
warns about. Never adjust a metric because a label is debatable; explain the
disagreement in prose instead and leave the number alone.

---

## Things I write myself — do not draft these

- `README.md`
- The system prompt and category definitions
- Error analysis and limitations
- The escalation threshold value and its justification

Suggest and critique these if I ask. Do not produce them unprompted.
