# ADR-0001 — One-objective operating model and agent roles

Status: Accepted
Date: 2026-10-05

## Context

FlexTern is an exploratory hardware research project expected to evolve over years through multiple humans and AI agents, none of whom share conversation history. Early results may invalidate later plans. A conventional ticket backlog would encode assumptions about work that has not been justified by evidence, and agents entering cold would have no reliable way to know what is true.

## Decision

1. The repository carries **one active engineering objective** at a time, in `docs/CURRENT_WORK.md`. No speculative backlog is maintained.
2. Roles are fixed: founder decides direction (with ChatGPT); Claude implements the current objective; Fable and other reviewer agents critique independently and leave notes rather than making changes.
3. The source-of-truth hierarchy in `AGENTS.md` §5 governs conflicts: evidence > code/tests > accepted ADRs > PROJECT_STATE/ARCHITECTURE > experiment interpretation > roadmap > informal notes.
4. Agents escalate consequential ambiguity to the founder using the question protocol in `AGENTS.md` §6 rather than guessing.
5. Documentation is part of the engineering system: experiment records, ADRs, reviews, and handoffs are first-class artifacts with fixed templates.

## Alternatives Considered

- **Full backlog / sprint planning.** Rejected: it presumes knowledge of experiment #37 before experiment #1 has run.
- **No process; rely on agent judgment.** Rejected: agents lose context between sessions and would reconstruct, and likely diverge on, project state.
- **Single agent does everything including strategy.** Rejected: the founder wants strategic decisions to pass through human judgment plus an independent reviewer.

## Why

Short feedback loops with one objective at a time match how exploratory architecture research actually progresses. Separating direction, implementation, and critique into distinct roles reduces the chance that one agent's assumptions silently become project truth.

## Consequences

- Agents must read the startup documents before working; the repository, not memory, is authoritative.
- Finishing an objective does not start the next one. There will be deliberate pauses for founder review.
- Documentation discipline costs time on every task. This is accepted.
- Reviewers may disagree with implementers in writing. This is desired.

## Evidence

Founder direction, 2026-10-05. No experimental evidence applies; this is a process decision.

## Supersedes

None.

## Superseded By

None.
