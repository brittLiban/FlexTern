# EXP-0011 Stage 2 — physical energy measurement plan (NOT APPROVED, NOT STARTED)

Status: **plan only.** The founder's directive requires separate approval before any equipment purchase or hardware-dependent measurement. Nothing here has been run. Nothing in Stage 1 (EXP-0011) may be cited as an energy result.

Why this exists: the Homerun and workload memos identify energy per answered query as the decisive unmeasured quantity (ternary vs INT4). Latency, CPU utilization and analytical models cannot substitute for it. A faster run can burn more energy (higher power for less time) or less, and idle platform power dominates at low utilization.

## Research question
At matched quality (Stage 1 definition), does BitNet b1.58 2B4T on bitnet.cpp use materially less **system energy per completed inference** than the quality-matched optimized INT4 comparator, across utilization levels and thermal conditions?

## Quantities to measure (all with units and uncertainty)
1. Gross system energy per completed inference (J), wall-plug, including everything the machine draws.
2. Idle and standby power (W): OS idle with the model unloaded, model loaded but idle, and sleep/standby if the device supports it.
3. Active energy above idle per inference (J) = gross energy over the inference window minus idle power x duration (idle measured immediately before and after, same thermal state).
4. Energy per generated token (J/token) and per prompt token, from runs that separate prefill from decode (same -pg P,1 / P,G design as Stage 1).
5. Energy per answered query at several duty cycles: continuous generation, and fixed query rates (for example one query every 10 s, 60 s, 600 s), including the idle time between queries. This is the quantity the product economics depend on (see EXP-0009A: low duty makes idle power dominate).
6. Thermal sensitivity: same measurement at two enclosure/airflow conditions (normal; restricted airflow or raised ambient), recording temperature and any frequency throttling.
7. Quality check in the same runs: outputs are the same prompts and decoding settings as Stage 1; Stage 1 quality results are reused for matching.

## Equipment (requirements, not products; no prices claimed)
- **Wall-power meter** between the wall and the test machine's supply with logging at >= 1 Hz (>= 10 Hz preferred), stated accuracy of about 1% or better, and an exportable log. This measures the whole system.
- **A battery-less x86 mini-PC (or desktop)** with a fixed, documented power plan and no background work. The laptop used in Stage 1 is not suitable: battery charging, WSL2 virtualization and shared GPU confound the wall measurement.
- **Temperature sensing:** ambient thermometer and CPU/package temperature and frequency readable from the OS (native Linux, no VM).
- **Optional second tier:** an ARM or Jetson-class board with an on-board power monitor (for example INA3221 rails on Jetson modules) to test whether conclusions hold off x86, with the caveat that rail sums omit regulator and peripheral losses.
- A power-conditioning note: use the same outlet, same cable, same meter, for every run.

## Procedure
1. Fix the OS, kernel, governor, runtime commits and model hashes (copy from protocol_v1.json); boot to a minimal service set; disable sleep.
2. Warm-up to thermal steady state (fixed duration), then record idle power for a fixed window.
3. For each (model, runtime) in the Stage 1 comparator set and each (prompt, output) cell: alternate A/B/A/B runs, N repetitions, with fixed cooldowns, randomized but seeded order; log the meter continuously with timestamps synchronized to run start/stop markers (a GPIO-free approach: print wall-clock markers and align logs; verify clock offset before the series).
4. Repeat at the second thermal condition for the primary pair only.
5. Duty-cycle experiments: a scheduler that issues one fixed query every T seconds for 30+ minutes per model, measuring total energy per answered query including idle gaps.
6. Compute energies by integrating power over aligned windows; report per-run values, medians, ranges.

## Statistical design and decision thresholds (to be formalized in protocol_v2 before purchase approval)
- Primary metric: ratio of INT4 to ternary **energy per answered query at a stated duty cycle**, geometric mean over cells, with ranges across repetitions.
- Founder's provisional thresholds (apply to this measured, system-level energy ratio only): >= 2.0x investigate a defensible architectural advantage; 1.3-2.0x software/format optimization, insufficient alone for an ASIC; < 1.3x reconsider ternary as the primary computing thesis; if quality is significantly degraded, make no efficiency claim.
- Required controls: identical prompts and decoding; same thermal state; meter zero check; an INT4 model run under both runtimes to bound runtime effects; a repeated-measure series to quantify run-to-run variance before any claim; report meter resolution as the floor of detectable differences.
- Stop rule for the stage: if the idle-dominated regime makes the ratio indistinguishable from 1 within measured noise at the buyer's duty cycle, record that as the result.

## What a positive Stage 2 would and would not mean
It would show a measured software/format energy advantage on one CPU platform. It would not show that FlexTern's Y-Core, packed datapaths or memory hierarchy beat existing implementations; that requires the gates in the commercial-validation memo (design partners, defensible FlexTern-specific contribution, SRAM leakage/duty/temperature/KV-cache accounting from EXP-0006-0009A).

## Approval requested
(1) Authorization to purchase the meter and a battery-less test machine (founder to choose budget; I have no reliable primary prices). (2) Approval of this plan, including the duty-cycle set and thermal conditions. (3) Confirmation that the Stage 1 comparator set is acceptable as the basis for Stage 2.
