# MEMO — EXP-0011 parallel track: commercial-validation package (research and outreach preparation only)

Date: 2026-11-05 · Author: Claude · Status: **preparation material for founder-led outreach. No customer has been contacted, no interview has happened, no demand, LOI, pilot or willingness-to-pay exists.** This memo is advice, not an accepted change. It is not permission to build a product.

Labels as in AGENTS §7, plus **REPORTED** (found in press/aggregator/vendor material, not verified against a primary source) and **JUDGMENT** (my opinion). Competitor performance numbers below are vendor claims unless stated.

---

## 1. What we are trying to learn

One question, stated so it can fail: **is there a buyer for whom a ~2× change in inference cost, memory footprint or power materially changes the economics of a deployed product, and who would evaluate a non-incumbent compute implementation?** Everything below is a way to find such a buyer, or to learn that none exists in reach. It deliberately does not assume an "offline technician chatbot".

Prior evidence relevant here (see the Homerun memo and EXP-0006–0009A): our own analysis says a resident-SRAM edge form loses at low duty and hot; a ternary advantage, if any, sits in memory bytes per weight (about 1.3–1.8× vs INT4 on decode, EXP-0003, INFERRED). So the buyer must care about **memory capacity, memory bandwidth/BOM or continuous high-duty power**, not about occasional queries.

## 2. Candidate segments (hypotheses to test, ranked by how directly they match the INT4-vs-ternary question)

| # | Segment | Buyer / budget owner (HYPOTHESIZED roles) | Current solution (REPORTED, vendor material) | Why a ~2× could matter | What would kill it |
|---|---|---|---|---|---|
| A | Industrial / embedded AI module OEMs (SMARC, COM-HPC, gateways, vision boxes) | Product manager / hardware architect of a module or box maker | Qualcomm Dragonwing IQ-8275 (20–40 TOPS) / IQ-9075 (100 TOPS; vendor claim 12 tok/s Llama-2 13B); NVIDIA Jetson Orin Nano Super ($249 at launch, 67 TOPS, 8 GB, 25 W; price may have risen); SiMa.ai Modalix (<10 W, 50 TOPS; vendor claim Llama-2 7B >10 tok/s) | LLM-class models in 8 GB-class memory; DRAM prices are reported to have risen sharply in 2026 (TrendForce via press), so memory per model is a BOM lever | OEMs buy proven SoCs; no one will qualify a new compute block; their customers do not ask for LLMs |
| B | Robotics makers / integrators (humanoid, AMR, arms) running vision-language-action models | Embedded compute lead or robot platform architect | NVIDIA Jetson/Thor class; GR00T N1.7 latency reported in earlier memo; Qualcomm robotics parts | Battery life and thermals; action-head latency dominates, so gain is limited to the backbone (earlier memo, INFERRED) | Gain confined to a minority of the latency budget; budgets thin (REPORTED sub-$10M revenue at most startups) |
| C | Industrial plants needing local/private inference | Plant IT/OT lead, reliability or maintenance manager | Siemens Industrial Copilot (edge variant, IPC 1047E + NVIDIA software, announced "planned" Nov 2024), Microsoft Foundry Local (offline INT4/INT8 Phi models), small on-prem vendors (marketing only) | Hardware cost per site only if the buyer sizes boxes by memory; otherwise software dominates | Cloud/VPC options satisfy most plants; hardware is not the constraint |
| D | Other segments where memory/power/BOM limits deployment: consumer-grade edge devices, kiosks, automotive cabin, retail/POS, security cameras with local summarization, medical devices | Hardware product owner | NPU SoCs from Qualcomm/MediaTek/Apple; Foundry Local; llama.cpp | DRAM share of BOM (one blog estimate 15–20% on mid-range embedded devices, unverified) and LLM-class memory needs | Existing NPUs already hit the device's cost target; model quality is the actual limit |

These are research hypotheses. I could not verify any buyer's budget, spend or purchasing process from primary sources.

## 3. The questions to answer for every segment (the template for notes after each call)

1. Specific buyer and budget owner (title, not a name), and who else must approve.
2. Current hardware/software solution, with part numbers and model sizes in production.
3. The unsolved technical or operational problem (what fails today: memory, latency, power, thermals, cost, certification?).
4. The actual cost of that problem to them (a number they state, with units; record how they derived it).
5. Duty cycle: continuous, bursty, or rare queries; hours per day; ambient temperature.
6. Would a 2× improvement in memory bytes, energy or cost change a decision? Which one, and what would they do with it?
7. What would they need to see to evaluate a new compute implementation (benchmarks, RTL, FPGA, silicon samples, SDK)?
8. Development and acquisition burden implied: time to design-in, qualification, certifications.
9. How FlexTern could differentiate, stated by the customer not by us.
10. Platform path: what adjacent products would they buy if the first worked?

## 4. Target-customer profile (for founder-led outreach)

- **Include:** companies that ship hardware containing an application processor plus DRAM and already run, or are evaluating, models above ~1B parameters on the device; module and gateway makers; robotics makers shipping to customers (not research labs); makers of vision/AI boxes for industrial customers.
- **Titles to find:** product manager (compute), embedded/hardware architect, director of engineering, platform lead for robot compute; for plants: OT/IT lead and maintenance/reliability lead.
- **Exclude for now:** pure cloud AI companies; companies with no hardware; research labs.
- **First sample:** 30 conversations across segments A–D, at least 10 in A and at least 8 in B, so the result can distinguish segments.

## 5. Outreach messages (founder-led; honest; no claim of a product)

**Cold message to a hardware/compute lead**
> Subject: Question about running 1–3B models on [product]
> Hi [name], I'm researching what limits on-device language models in industrial hardware: memory capacity, memory cost, power, or latency. We're benchmarking 1.58-bit (ternary) against 4-bit models on commodity CPUs and I'd like to learn what actually blocks LLM-class features in products like yours. Could I ask you 20 minutes of questions? I'm not selling anything and won't share what you tell me without permission.

**Follow-up if they agree:** send the interview guide questions in advance (Section 3).

**Message to a plant lead**
> Hi [name], I'm trying to understand whether maintenance teams are allowed to run AI tools on plant documents and what hardware they would be willing to put on the plant network. Could I ask you a few questions (20 min)? I'm not selling anything.

Rules: never claim a product exists, never imply customers, never quote performance numbers that Stage 1 has not measured, record permission before using any statement externally.

## 6. Interview guide (20–30 min)

1. Walk me through the product and where it runs. What compute and memory does it have, and who chose them?
2. Do you run, or plan to run, any language or vision-language model on the device? Which, at what size, how fast?
3. What limited it (memory, cost, power, thermals, latency, software support, model quality)? Which of these is the first to bind?
4. What does that limit cost you per unit or per year? How did you estimate it?
5. How often does the model run (continuous, bursts, rare)?
6. If memory per model dropped by half, or energy per answer dropped by half, what would you change? Would it change a design, a price, or a customer promise?
7. Whom would you need to convince to try a different compute option? What evidence would they need? How long does a design-in take?
8. What have you evaluated in the last two years (NVIDIA, Qualcomm, SiMa.ai, others) and why did you choose or reject them?
9. Who owns the budget for this decision? Would you pay for an evaluation, and in what range?
10. Is there anyone else I should talk to (a customer, a competitor, a supplier)?

## 7. Evidence-recording template (one row per conversation; keep verbatim quotes)

| Field | Content |
|---|---|
| ID, date, interviewer | |
| Segment (A–D), company type (no names in shared copies without permission), role | |
| Product, compute/memory used | |
| Model sizes in use / planned | |
| Binding constraint and stated cost (number + units + how derived) | |
| Duty cycle | |
| Would 2× change a decision? (verbatim) | |
| Evaluation requirements and design-in time | |
| Budget owner and stated willingness to pay (verbatim; mark "none stated" explicitly) | |
| Commitments actually made (LOI, pilot, none) | |
| Confidence (observed / reported by interviewee / my inference) | |
| Follow-up | |

Rules for the log: separate what the interviewee said from what we inferred; count a "commitment" only if it is written; a polite "interesting" is recorded as no commitment.

## 8. Gates before recommending a custom accelerator (status as of this memo)

| # | Gate | Status |
|---|---|---|
| 1 | Meaningful advantage vs strong conventional baselines | Not established. EXP-0011 Stage 1 measures speed/memory on a CPU; Stage 2 energy not approved |
| 2 | Comparable application-level quality | Pending Stage 1 quality results |
| 3 | A customer workload where the improvement matters | Not established; this package is how we look for one |
| 4 | At least two OEM/design partners willing to evaluate an implementation | None; zero conversations have happened |
| 5 | A defensible FlexTern-specific contribution | Not established; generic-cell synthesis counts are not performance; earlier experiments constrain the SRAM/duty/temperature case |
| 6 | Feasible economics, resources and capital | Not established |

## 9. Success and stop criteria for this package

- **Proceed to a deeper OEM program** if, after about 30 conversations, at least 6 buyers state a memory/power/BOM constraint with a quantified cost, at least 3 say a ~2× change would change a design or price, and at least 2 agree in writing to evaluate an implementation against defined requirements.
- **Middle band:** 3–5 quantified constraints and 1 written evaluation interest: narrow to the single segment with the strongest signals and run 15 more conversations.
- **Stop or redirect:** fewer than 3 quantified constraints, or nobody names an evaluation path: the ternary-hardware commercial thesis has no supported buyer in these segments.
These thresholds are provisional and must be agreed by the founder before outreach begins; they must not be changed after seeing responses.

## 10. Sources and gaps
- Jetson Orin Nano Super specifications and launch price: Phoronix, Tom's Hardware, CNX Software (secondary); a later page suggests a higher price (unverified).
- Qualcomm Dragonwing IQ-8275/IQ-9075 specifications and LLM throughput claims: Qualcomm product pages and developer documentation (vendor claims).
- SiMa.ai Modalix claims: company announcements and press (vendor claims).
- Siemens Industrial Copilot status: Siemens press release of Nov 2024 and trade coverage (plans, no 2026 confirmation found).
- Microsoft Foundry Local: Microsoft devblog (May 2025) and community tutorials.
- DRAM price movements: TrendForce figures via press and edge-AI trade coverage (projections, secondary).
- **Not found:** any independent head-to-head of these platforms on ternary vs INT4 LLMs; any primary pricing or volume data; any buyer budget data.
