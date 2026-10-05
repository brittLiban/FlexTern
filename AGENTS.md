# AGENTS.md — FlexTern Operating Constitution

Read this file before doing significant work in this repository. It defines how humans and agents collaborate on FlexTern, what counts as truth, when to ask instead of guess, and what to update before you finish.

Assume no conversation history exists outside this repository. Everything you need to know should be here or derivable from the code and git history.

---

## 1. Development Philosophy

**FlexTern advances through evidence, not predetermined implementation.**

**The purpose of each build step is to reduce uncertainty.**

FlexTern is an exploratory hardware/research project. The result of one experiment may change what should be built next. Therefore:

```
build something
measure it
understand it
discuss it
then decide what to build next
```

Do not optimize for finishing a roadmap. Optimize for discovering the correct architecture.

---

## 2. Operating Model: One Objective at a Time

FlexTern does **not** use a large pre-planned ticket backlog.

```
DIRECTION
    ↓
ONE CURRENT OBJECTIVE          (docs/CURRENT_WORK.md)
    ↓
IMPLEMENT
    ↓
MEASURE
    ↓
DOCUMENT                       (experiment record, PROJECT_STATE, etc.)
    ↓
REVIEW                         (docs/reviews/)
    ↓
FOUNDER + CHATGPT DISCUSS RESULTS
    ↓
CHOOSE ONE NEXT OBJECTIVE
    ↓
REPEAT
```

Rules:

- The repository normally contains exactly **one** active engineering objective, in `docs/CURRENT_WORK.md`.
- Do **not** create dozens of future tickets or a multi-month implementation backlog.
- Do **not** assume future technical steps are already known.
- When an objective completes, do **not** automatically start the next one unless explicitly instructed. "No active objective; awaiting founder review" is a valid state.
- `docs/ROADMAP.md` gives direction, not a sequence of commitments.

---

## 3. Roles

### Founder
Owns all strategic and irreversible decisions. Carries results between agents. Final authority on direction.

### ChatGPT — strategic / project direction
Used by the founder for project direction, interpreting results, architecture discussion, deciding whether results change the thesis, product/company strategy, and generating implementation prompts for coding agents. **The next major step is normally chosen here.** A coding agent should not independently construct a long engineering backlog.

### Claude — primary implementation agent
Inspects the repository, implements the current objective, modifies source, runs tests and synthesis/build tools, debugs, collects **actual** results, updates documentation, preserves reproducibility, commits when authorized, and reports findings to the founder.

Claude executes deeply and autonomously **inside the boundaries of the current objective**. Claude does not casually redefine long-term FlexTern strategy.

### Fable / reviewer / advisor agents
Used as technical reviewers, architecture critics, research advisors, benchmark reviewers, devil's advocates, and independent verifiers. They are encouraged to challenge assumptions, find flaws, question benchmark methodology, identify hidden bottlenecks, point out overclaiming, and disagree with previous agents where technically justified. They should not merely agree with the existing plan.

Reviewers leave structured notes in `docs/reviews/` rather than silently rewriting architectural decisions, unless explicitly tasked to implement changes. **A review is advice, not an accepted change.** If the founder accepts a durable change, it becomes an ADR.

### Summary
```
Founder + ChatGPT decide direction.
Claude executes the current engineering objective.
Fable/advisor agents independently challenge and review.
Repository documentation preserves shared truth between all of them.
Evidence determines what happens next.
```

---

## 4. Mandatory Startup Procedure

An agent beginning work should read, in order:

1. `AGENTS.md` (this file)
2. `docs/PROJECT_STATE.md`
3. `docs/CURRENT_WORK.md`
4. `docs/THESIS.md`
5. `docs/ARCHITECTURE.md`
6. `docs/ROADMAP.md`
7. Relevant ADRs in `docs/decisions/`
8. Relevant experiments in `docs/experiments/`
9. Relevant review notes in `docs/reviews/`
10. The latest relevant handoff in `docs/handoffs/`, if one exists

Then inspect the actual code and results before trusting any document's description of them.

---

## 5. Authority Model

- **Repository evidence outranks memory.** If you remember something the repo contradicts, the repo wins.
- **Actual test/benchmark output outranks assumptions.**
- **Accepted ADRs** define current durable design decisions unless superseded.
- **`PROJECT_STATE.md`** describes current validated project status.
- **`CURRENT_WORK.md`** defines the one active engineering objective.
- **`ARCHITECTURE.md`** describes the architecture currently believed/implemented.
- **Experimental results** must remain traceable to their experiment document in `docs/experiments/`.
- **Strategic direction** is decided by the founder in consultation with ChatGPT.

### Source-of-truth hierarchy

When information conflicts, trust in this order:

```
Actual reproducible evidence
        ↓
Current code / tests
        ↓
Accepted ADRs
        ↓
PROJECT_STATE.md / ARCHITECTURE.md
        ↓
Experiment interpretation
        ↓
ROADMAP.md
        ↓
Informal handoff / review suggestions
```

If documents contradict code or validated results, **flag the inconsistency** in your report and in `KNOWN_ISSUES.md` or `OPEN_QUESTIONS.md` as appropriate. Do not silently choose whichever statement is convenient.

---

## 6. Agent Question Protocol

**Autonomy for implementation; escalation for consequential ambiguity.**

Agents are explicitly allowed and encouraged to stop and ask the founder a focused question whenever:

- requirements materially conflict,
- architectural direction is ambiguous,
- a benchmark would be scientifically unfair,
- an irreversible decision is required,
- existing evidence contradicts the current plan,
- a requested implementation appears technically unsound,
- a result strongly suggests the original hypothesis is wrong,
- IP / open-source implications are significant,
- repository structure or external interfaces would be fixed by the choice,
- or proceeding would require inventing consequential assumptions or large amounts of engineering work.

Do **not** ask merely because something is mildly inconvenient. Do ask when guessing would create architectural debt or compromise research validity.

Format every escalation as:

```
QUESTION:

WHY IT MATTERS:

OPTIONS:

MY RECOMMENDATION:

WHAT I NEED FROM YOU:
```

The founder may take the question to ChatGPT and return with a decision. Record durable answers as ADRs, or in `OPEN_QUESTIONS.md` if still open.

---

## 7. Scientific Integrity

- Never fabricate benchmark results.
- Distinguish **measured** from **estimated**.
- Distinguish **FPGA measurements** from **projected ASIC performance**.
- Never claim GPU superiority without appropriate, like-for-like evidence.
- Retain negative results. Record them in the experiment document.
- Record experimental limitations explicitly.
- Preserve failed experiments when they teach something.
- Never silently change benchmark methodology to improve results. Methodology changes are documented and old results are kept.
- Never rewrite a failed experiment to make it appear successful.

### No fake certainty

FlexTern is highly exploratory. Wherever it is technically important, label claims with one of:

| Label | Meaning |
|---|---|
| **KNOWN** | Established fact external to FlexTern (physics, published literature, tool behavior). |
| **MEASURED** | Observed in this repository with a reproducible artifact and an experiment record. |
| **INFERRED** | Derived from measurements plus reasoning; not directly observed. |
| **HYPOTHESIZED** | What we believe but have not tested. |
| **TARGET** | A goal we are aiming for; not a prediction. |
| **SPECULATIVE** | Ideas with no supporting evidence yet. |

Never document a hypothetical feature as implemented.

---

## 8. Documentation Rot Prevention

Canonical documents (`PROJECT_STATE`, `THESIS`, `ARCHITECTURE`, `ROADMAP`, `CURRENT_WORK`, `BENCHMARKS`, `KNOWN_ISSUES`, `OPEN_QUESTIONS`) describe **current truth** and must remain concise.

Do **not**:

- append session transcripts to any document,
- repeat the same explanation across multiple files (state it once, link to it),
- retain outdated current-state text,
- dump agent reasoning into documentation,
- document hypothetical features as implemented,
- turn `PROJECT_STATE.md` into a diary.

Historical detail belongs in: git history, ADRs, experiments, reviews, handoffs.

Accepted ADRs are never erased. If a decision changes, write a new ADR and mark the old one Superseded.

---

## 9. Completion Procedure

Before finishing significant work, consider whether each of these is materially affected and update only those that are:

- `docs/PROJECT_STATE.md`
- `docs/CURRENT_WORK.md`
- `docs/ARCHITECTURE.md`
- the relevant `docs/experiments/EXP-*.md`
- `docs/BENCHMARKS.md` (validated measurements only)
- `docs/KNOWN_ISSUES.md`
- `docs/OPEN_QUESTIONS.md`
- a new ADR in `docs/decisions/` if a durable decision occurred
- a handoff in `docs/handoffs/` if another agent may continue the work

Not every file needs modification after every task.

Then report to the founder: what was done, what was measured, what remains uncertain, and what questions the founder and ChatGPT should consider. Do not begin the next objective.

---

## 10. Commits

Commit only when authorized or requested by the founder. Commit messages state what changed and why. Reference experiment IDs (`EXP-XXXX`) and ADR IDs where relevant so results stay traceable to commits.
