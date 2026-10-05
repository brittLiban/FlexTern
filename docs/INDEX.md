# FlexTern Documentation Index

Start with `../AGENTS.md`, then follow the Mandatory Startup Procedure it defines.

## Canonical documents (current truth, kept concise)

| Document | Answers |
|---|---|
| [PROJECT_STATE.md](PROJECT_STATE.md) | Where are we right now? What is proven, what is not, what is blocking? Two-minute read. |
| [CURRENT_WORK.md](CURRENT_WORK.md) | What is the **one** active engineering objective, its scope, and definition of done? |
| [THESIS.md](THESIS.md) | Why does FlexTern exist? What do we hypothesize, what is established, what are the risks? |
| [ARCHITECTURE.md](ARCHITECTURE.md) | What is the current design, labeled implemented / planned / exploratory? |
| [ROADMAP.md](ROADMAP.md) | What broad phases might the project move through? Direction, not a backlog. |
| [BENCHMARKS.md](BENCHMARKS.md) | Validated measurements only, each traceable to an experiment. |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Unresolved technical defects and limitations. |
| [OPEN_QUESTIONS.md](OPEN_QUESTIONS.md) | Research questions that matter but are not yet decisions or bugs. |

## Historical / append-only records

| Directory | Contains | Naming |
|---|---|---|
| [decisions/](decisions/) | Architecture Decision Records. Why we made lasting choices. Never erased; superseded instead. | `ADR-XXXX-short-description.md` |
| [experiments/](experiments/) | Experiment records with method, results, limitations. Negative results included. | `EXP-XXXX-short-description.md` |
| [reviews/](reviews/) | Structured critiques from independent reviewer/advisor agents. Advice, not decisions. | `REV-YYYY-MM-DD-topic.md` |
| [handoffs/](handoffs/) | Notes from an agent finishing substantial work for whoever continues. | `YYYY-MM-DD-short-topic.md` |

Each directory contains a `TEMPLATE.md` and a short `README.md`.

## Relationship between documents

- **ARCHITECTURE.md** says *what* the design is. **ADRs** say *why*.
- **CURRENT_WORK.md** says what we are doing. **EXP** records say what happened when we did it.
- **BENCHMARKS.md** is the scoreboard. It cites EXP records; it never holds numbers that are not in one.
- **PROJECT_STATE.md** summarizes. It never holds history. History lives in git, ADRs, experiments, reviews, handoffs.
- **OPEN_QUESTIONS.md** holds unknowns. When one is resolved by evidence, it is closed and the answer lands in an ADR, ARCHITECTURE.md, or THESIS.md as appropriate.

## Source-of-truth hierarchy

Defined in `../AGENTS.md` §5. In short: evidence > code/tests > accepted ADRs > PROJECT_STATE/ARCHITECTURE > experiment interpretation > roadmap > informal notes.
