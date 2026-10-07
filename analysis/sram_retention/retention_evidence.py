"""
EXP-0009 evidence: SRAM power modes, leakage, retention, wake latency/energy and voltage-domain granularity.

Every external constant used by the model is defined here with a source tag.  Tiers (prompt): T1 measured silicon or
vendor datasheet; T2 primary simulated / PDK-based; T3 credible secondary; T4 historical or context only.  Missing
fields stay empty.  Power modes are kept distinct (ACTIVE / IDLE-STANDBY / LIGHT SLEEP / DATA RETENTION / DEEP SLEEP /
SHUTDOWN); a mode that loses data is never used as retention by the model.

Units: mW per DECIMAL MB (8e6 bits).  "Mbit" in SRAM papers is 2^20 bits; conversions state the factor.
"""

from __future__ import annotations

import csv
import os

FIELDS = ["source_id", "publication", "year", "institution_or_vendor", "process_node", "bitcell_type", "macro_capacity", "number_of_banks",
          "supply_voltage", "retention_voltage", "temperature", "active_leakage", "standby_leakage", "retention_leakage", "deep_sleep_leakage",
          "leakage_per_bit", "leakage_per_MB", "wake_latency", "wake_energy", "data_retention_guarantee", "frequency_before_sleep",
          "frequency_after_wake", "state_transition_mechanism", "measured_or_simulated", "silicon_or_model", "commercial_or_research",
          "energy_boundary", "tier", "modes_reported", "notes", "citation"]


def _s(**kw) -> dict:
    d = {k: "" for k in FIELDS}
    d.update(kw)
    return d


MBIT = 2 ** 20                       # bits per Mbit as used in SRAM papers
MB_BITS = 8e6                        # bits per decimal MB

SOURCES = [
    _s(source_id="R01", publication="IEEE JSSC (early access 2025, print 2026, pp. 1466-1476), DOI 10.1109/JSSC.2025.3639747: 'A 3-nm FinFET 563-kbit 35.5-Mbit/mm2 Dual-Rail SRAM With 3.89-pJ/Access High Energy Efficient and 27.5-uW/Mbit One-Cycle Latency Low-Leakage Mode'",
       year="2025", institution_or_vendor="TSMC (Aoyagi, Nii, Yabuuchi, ... Liao, Chang)", process_node="3 nm FinFET", bitcell_type="HD 6T", macro_capacity="563 kbit", number_of_banks="",
       supply_voltage="dual rail (array Vdd, periphery at lower Vchip); values not in abstract", retention_voltage="", temperature="25 C (leakage); 85 C (total energy)",
       standby_leakage="27.5 uW/Mbit in the one-cycle-latency low-leakage mode (1-CLM: bitline prechargers off during NOP), 10% below normal standby -> normal standby ~30.6 uW/Mbit",
       leakage_per_bit="26 pW/bit (1-CLM, 25 C)", leakage_per_MB=f"{27.5e-6 * MB_BITS / MBIT * 1e3:.3f} mW/MB (1-CLM, 25 C); ~{27.5e-6 / 0.9 * MB_BITS / MBIT * 1e3:.3f} mW/MB normal standby",
       wake_latency="one cycle from 1-CLM ('rapid wake-up with minimal latency and peak current')", wake_energy="", data_retention_guarantee="data retained (array supply unchanged)",
       state_transition_mechanism="turn off BL prechargers during NOP cycles (light-sleep class)", measured_or_simulated="measured", silicon_or_model="silicon", commercial_or_research="foundry (commercial)",
       energy_boundary="macro incl. periphery", tier="T1", modes_reported="ACTIVE; IDLE-STANDBY; LIGHT SLEEP (1-CLM)",
       notes="Only modern-node (3 nm) measured standby leakage per Mbit found. 6.94 pJ/access/Mbit total energy at 85 C does not separate leakage. No deep retention mode reported.",
       citation="abstract via OpenAlex (doi:10.1109/jssc.2025.3639747) and sscs.ieee.org tag page"),
    _s(source_id="R02", publication="2020 IEEE Symposium on VLSI Circuits, DOI 10.1109/VLSICircuits18222.2020.9162985: 'A 29.2 Mb/mm2 Ultra High Density SRAM Macro using 7nm FinFET Technology with Dual-Edge Driven Wordline/Bitline and Write/Read-Assist Circuit' (EXP-0008 E02)",
       year="2020", institution_or_vendor="Socionext (Yokoyama, Tanaka, ... Tanaka)", process_node="7 nm FinFET", bitcell_type="HD 6T (single 512x512 array)", macro_capacity="256 Kb", number_of_banks="1 array",
       supply_voltage="nominal (not in abstract)", retention_voltage="", temperature="25 C", standby_leakage="16.9 uW standby power at 25 C for 256 Kb (as quoted by the paper's summary page; not restated in the OpenAlex abstract)",
       leakage_per_bit="64 pW/bit", leakage_per_MB=f"{16.9e-6 * MB_BITS / 262144 * 1e3:.3f} mW/MB (25 C)", wake_latency="", wake_energy="", data_retention_guarantee="n/a (no sleep mode reported)",
       measured_or_simulated="measured", silicon_or_model="silicon", commercial_or_research="commercial design house", energy_boundary="macro incl. periphery", tier="T1 (figure quoted from a summary page; abstract not independently confirmed in this session)",
       modes_reported="ACTIVE; IDLE-STANDBY", notes="Ultra-high-density single-array macro; standby is the nominal-voltage idle state.", citation="researchgate summary of the VLSI 2020 paper (page itself returned 403)"),
    _s(source_id="R03", publication="arXiv 2508.02536 (2025): 'ReGate: Enabling Power Gating in Neural Processing Units'", year="2025", institution_or_vendor="academic (NPU power-gating study)", process_node="7 nm FinFET PDK (simulated)",
       bitcell_type="SRAM buffers (compiler-class)", macro_capacity="128 MB on-chip SRAM in 4 KB segments", number_of_banks="32,768 segments", supply_voltage="", retention_voltage="", temperature="",
       retention_leakage="SLEEP mode (data retained): 25% of active-mode static power", deep_sleep_leakage="OFF (gated-Vdd, data NOT retained): 0.2% of active static power",
       wake_latency="SLEEP 4 cycles; OFF 10 cycles", wake_energy="break-even time 41 cycles (SLEEP) / 82 cycles (OFF)", data_retention_guarantee="SLEEP retains; OFF loses",
       state_transition_mechanism="per-4 KB segment power gating controlled by compiler-known buffer lifetimes", measured_or_simulated="simulated (McPAT/NeuroMeter-class power models)", silicon_or_model="model",
       commercial_or_research="research", energy_boundary="SRAM segment static power", tier="T2", modes_reported="ACTIVE; SLEEP (retention); OFF (shutdown)",
       notes="Only modern-node retention RATIO found (0.25, modeled). Cycle counts at an unstated clock.", citation="arxiv.org/html/2508.02536v1"),
    _s(source_id="R04", publication="Synopsys Memory Compilers IP product page and related Synopsys material (light sleep / deep sleep / shut down)", year="2025", institution_or_vendor="Synopsys", process_node="compilers offered for N3P/N4P/N5/N7 and others (figures not node-specific)",
       bitcell_type="compiler SRAM", macro_capacity="", retention_leakage="Light Sleep: array source-biased (diode), up to 50% standby power reduction, fast wake; Deep Sleep: periphery supply off, array source-biased and retained, up to 70% standby saving",
       deep_sleep_leakage="Shut Down: array and periphery off, data NOT retained", data_retention_guarantee="retained in light/deep sleep; lost in shut down", wake_latency="'fast wake-up' (unquantified); in-rush vs wake-time trade-off noted",
       state_transition_mechanism="header/footer switches with source bias", measured_or_simulated="vendor claim", silicon_or_model="product", commercial_or_research="commercial", energy_boundary="macro standby power", tier="T3",
       modes_reported="ACTIVE; LIGHT SLEEP; DEEP SLEEP (retention); SHUT DOWN", notes="Mode definitions used by the model; percentages are marketing ceilings, not measurements.", citation="synopsys.com designware memory compilers page; Synopsys low-power SRAM descriptions"),
    _s(source_id="R05", publication="arXiv 2310.09094 (2023): 'A RISC-V MCU with adaptive reverse body bias and ultra-low-power retention mode in 22 nm FD-SOI'", year="2023", institution_or_vendor="academic/industrial (GlobalFoundries 22FDX)", process_node="22 nm FD-SOI (planar, NOT FinFET)",
       bitcell_type="ULV dual-well 6T", macro_capacity="128 KiB (4 x 32 KiB banks; 1-8 KiB macros)", number_of_banks="4", supply_voltage="0.50-0.55 V logic/SRAM", retention_voltage="retention with adaptive reverse body bias (voltage not stated)", temperature="-40 to 125 C",
       active_leakage="SRAM leakage 4.2 uW (active, bus-gated design)", retention_leakage="3.2 uW at 25 C for 128 KiB; 142 uW at 125 C; 75x lower than active (periphery off, bitcell state kept)",
       deep_sleep_leakage="power-down erases state, ~99% leakage reduction", leakage_per_MB=f"retention {3.2e-6 / (128 * 1024) * 1e6 * 1e3:.4f} mW/MB at 25 C; {142e-6 / (128 * 1024) * 1e6 * 1e3:.3f} mW/MB at 125 C",
       wake_latency="200 ns (wake-up procedure with in-rush current limiting)", wake_energy="", data_retention_guarantee="retained in retention mode; lost in power-down", state_transition_mechanism="SRAM power-state control: periphery power switch + body-bias regulation; 2.1% area overhead per 4 KiB macro",
       measured_or_simulated="measured", silicon_or_model="silicon", commercial_or_research="research", energy_boundary="SRAM macros incl. periphery", tier="T1 (22 nm FD-SOI; supporting technology, not FinFET)", modes_reported="ACTIVE; SLEEP (clock-gated); RETENTION; POWER-DOWN",
       notes="Best measured retention ratio and wake latency found; depends on body biasing that FinFET does not offer. Temperature factor 44x from 25 to 125 C (~1.46x per 10 C).", citation="arxiv.org/pdf/2310.09094 (table and section III)"),
    _s(source_id="R06", publication="ISQED 2014, DOI 10.1109/ISQED.2014.6783302: '40nm Ultra-low leakage SRAM at 170 deg.C operation for embedded flash MCU'", year="2014", institution_or_vendor="Renesas", process_node="40 nm embedded-flash CMOS (planar)", bitcell_type="optimized 6T",
       macro_capacity="160 kb", temperature="25 C and 170 C", retention_leakage="deep standby (sleep) 1.86 uW/Mb at 25 C, 643 uW/Mb at 170 C (data retained; source raise + floated bitlines + periphery cut)", leakage_per_MB=f"{1.86e-6 * MB_BITS / MBIT * 1e3:.4f} mW/MB at 25 C; {643e-6 * MB_BITS / MBIT * 1e3:.3f} mW/MB at 170 C",
       wake_latency="rush-current suppression circuit on resume (time not in abstract)", data_retention_guarantee="retained", state_transition_mechanism="raise pull-down source line, float bitlines, separate cell and periphery rails",
       measured_or_simulated="measured", silicon_or_model="silicon", commercial_or_research="commercial", energy_boundary="macro incl. periphery", tier="T1 for mechanism and temperature scaling; T4 for node", modes_reported="ACTIVE; DEEP STANDBY (retention)",
       notes="Temperature factor 346x over 145 C (~1.50x per 10 C) in a subthreshold-dominated retention mode.", citation="OpenAlex abstract doi:10.1109/isqed.2014.6783302"),
    _s(source_id="R07", publication="Design & Reuse article 25907: 'A 55-nm Ultra Low Leakage SRAM Compiler with Optimized Power Gating Design'", year="2011", institution_or_vendor="academic/UMC 55 nm", process_node="55 nm planar", retention_leakage="retention mode 4x lower than baseline (data retained; diode-clamped cell rail)",
       deep_sleep_leakage="sleep mode 50x lower (data lost)", data_retention_guarantee="retained in retention; lost in sleep", state_transition_mechanism="PMOS diode clamp with 3-bit programmable retention level", measured_or_simulated="simulated (test chip in fabrication at publication)", silicon_or_model="model",
       commercial_or_research="research", tier="T2 (old node; mechanism only)", modes_reported="STANDBY; RETENTION; SLEEP (shutdown)", citation="design-reuse.com/articles/25907"),
    _s(source_id="R08", publication="ISSCC 2014 (Toshiba): 'A 7 ns-access-time 25 uW/MHz 128 kb SRAM for low-power fast wake-up MCU in 65 nm CMOS with 27 fA/b retention current'", year="2014", institution_or_vendor="Toshiba", process_node="65 nm planar",
       macro_capacity="128 kb", retention_leakage="27 fA per bit retention current (1.2 V supply) = ~1000x below conventional", leakage_per_MB=f"{27e-15 * 1.2 * MB_BITS * 1e3:.5f} mW/MB (array only, 1.2 V)", wake_latency="7 ns access from the low-leakage state",
       data_retention_guarantee="retained (10-year battery retention claim)", measured_or_simulated="measured", silicon_or_model="silicon", commercial_or_research="commercial", tier="T4 (node) / T1 (mechanism)", modes_reported="ACTIVE; DEEP SLEEP (retention)",
       notes="Special low-leakage cell; shows that sub-100 fA/bit retention is physically reachable at old nodes with fast wake.", citation="phys.org 2014-02 Toshiba XLL SRAM; ISSCC 2014 digest"),
    _s(source_id="R09", publication="VLSI Symposium 2017 (Renesas): 65 nm SOTB embedded SRAM, 13.7 nW/Mbit standby", year="2017", institution_or_vendor="Renesas", process_node="65 nm SOTB (silicon on thin BOX)", retention_leakage="13.7 nW/Mbit standby with reverse body bias (data retained), ~1000x reduction",
       leakage_per_MB=f"{13.7e-9 * MB_BITS / MBIT * 1e3:.6f} mW/MB", data_retention_guarantee="retained", state_transition_mechanism="reverse body bias in standby", measured_or_simulated="measured", silicon_or_model="silicon", commercial_or_research="commercial", tier="T4 (node) / T1 (mechanism)",
       modes_reported="ACTIVE; STANDBY (retention, RBB)", notes="Body-bias technologies (SOTB, FD-SOI) reach 1000x; FinFET has no body bias.", citation="electronicspecifier.com/?p=64135"),
    _s(source_id="R10", publication="Symposium on VLSI Circuits 2013 (Intel): 'A 22nm 2.5MB slice on-die L3 cache for the next generation Xeon processor'; ISSCC 2006 Intel Tulsa 65 nm 16 MB L3", year="2013 / 2006", institution_or_vendor="Intel", process_node="22 nm tri-gate (FinFET) / 65 nm",
       macro_capacity="2.5 MB slice", deep_sleep_leakage="22 nm: per-slice shutoff reduces SRAM leakage (data lost); 65 nm: sleep transistors in array and periphery cut cache leakage > 2x, with wake-up counters balancing switching vs leakage",
       data_retention_guarantee="shutoff loses data; 65 nm sleep retains", state_transition_mechanism="sleep transistors / shutoff per slice", measured_or_simulated="measured", silicon_or_model="silicon", commercial_or_research="commercial", tier="T1 (mechanism, 22 nm) / T4 (65 nm)",
       modes_reported="ACTIVE; SLEEP; SHUTOFF", notes="Production caches use sleep + shutoff; retention-mode leakage per MB is not published.", citation="OpenAlex abstract (VLSI 2013); realworldtech ISSCC 2006 Tulsa"),
    _s(source_id="R11", publication="Jeong, Kahng, Kang, Rosing, Strong, 'MAPG: Memory Access Power Gating', DATE 2012", year="2012", institution_or_vendor="UC San Diego", process_node="32 nm (modeled)", wake_latency="core wake 5.0-14.2 ns by two-stage switch mode; break-even ~9.5 ns",
       wake_energy="core wake energy 1.6-3.3 nJ (rail recharge, Q = C x dV)", state_transition_mechanism="header power gating with programmable two-stage wake to limit in-rush", measured_or_simulated="simulated (PDN model)", silicon_or_model="model", commercial_or_research="research", tier="T2 (old node)",
       notes="Method for wake energy (Q_domain = C x dV) and in-rush-limited wake time adopted here as assumption B6/B7.", citation="vlsicad.ucsd.edu/Publications/Conferences/280/c280.pdf"),
    _s(source_id="R12", publication="Qin, Cao, Markovic, Vladimirescu, Rabaey, 'SRAM leakage suppression by minimizing standby supply voltage', ISQED 2004", year="2004", institution_or_vendor="UC Berkeley", process_node="130 nm", retention_voltage="data retention voltage (DRV) < 300 mV; > 90% leakage saving at DRV; DRV margin ~10 mV over 25-125 C",
       measured_or_simulated="measured", silicon_or_model="silicon", tier="T4", modes_reported="STANDBY at DRV", notes="Defines DRV and its temperature sensitivity; node too old for a modern verdict.", citation="eecs.berkeley.edu/bears/2004/STARS/final/qin.pdf"),
    _s(source_id="R13", publication="ASAP7 predictive 7 nm FinFET PDK (Clark et al., Microelectronics Journal 2016), models r1p7 '7nm_TT/SS/FF_160803.pm' (BSIM-CMG), simulated here with ngspice 45.2 + OpenVAF-compiled BSIM-CMG 110", year="2016 (models); 2026 (this run)",
       institution_or_vendor="Arizona State University (predictive)", process_node="7 nm FinFET (predictive, not a foundry process)", bitcell_type="6T, all single-fin 'sram' Vt devices", macro_capacity="single cell (array only, no periphery)",
       supply_voltage="0.25-0.80 V swept", retention_voltage="hold state converged down to 0.25 V at TT; hold SNM in hold_snm table", temperature="25 / 50 / 75 / 85 / 105 C swept",
       leakage_per_bit="see asap7_cell_leakage.csv (e.g. TT 25 C: 25.9 pA at 0.70 V, 5.6 pA at 0.35 V; 85 C: 127 pA at 0.70 V, 84 pA at 0.35 V)", leakage_per_MB="array only: 0.145 mW/MB at 0.70 V 25 C; 0.709 mW/MB at 0.70 V 85 C (8e6 cells x I x V)",
       measured_or_simulated="simulated", silicon_or_model="model (predictive)", commercial_or_research="research", energy_boundary="bitcell array only (periphery excluded)", tier="T2",
       modes_reported="hold state vs supply voltage and temperature", notes="Used for RATIOS only (voltage and temperature scaling); absolute level anchored on R01/R02. PDK SRAM-device Ioff spec 0.1 nA/fin at 25 C.", citation="github.com/The-OpenROAD-Project/asap7_pdk_r1p7 (BSD-3); analysis/sram_retention/spice/"),
    _s(source_id="R14", publication="EXP-0008 E05: ISSCC 2014-class 28 nm HKMG embedded SRAM, 0.41 uA standby leakage 32 kb, low-voltage resume-standby", year="2014", process_node="28 nm planar", retention_voltage="0.46 V data-retention sleep", standby_leakage="296 nW/kB at 0.5 V idle", retention_leakage="0.23 nW/kB in DRV sleep (~1000x)",
       leakage_per_MB="0.296 mW/MB idle; 0.00023 mW/MB sleep", measured_or_simulated="measured", silicon_or_model="silicon", tier="T1 (28 nm; not modern)", modes_reported="IDLE; RESUME-STANDBY (retention)", citation="EXP-0008 evidence.py E05"),
    _s(source_id="R15", publication="EXP-0008 E11: secondary statements on FinFET SRAM cell standby current (~100 pA/cell class)", process_node="FinFET (unspecified)", leakage_per_bit="sub-100 pA per cell", tier="T3", citation="EXP-0008 evidence.py E11"),
    _s(source_id="R16", publication="Song et al., 'A 3-nm Gate-All-Around SRAM Featuring an Adaptive Dual-Bitline and an Adaptive Cell-Power Assist Circuit', IEEE JSSC 57(1), 2022, DOI 10.1109/JSSC.2021.3123077", year="2022", institution_or_vendor="Samsung (3GAE)", process_node="3 nm GAA",
       macro_capacity="256 Mb", measured_or_simulated="measured", silicon_or_model="silicon", tier="T1 (no leakage or retention figures in the abstract)", modes_reported="", notes="Confirms 3 nm GAA SRAM silicon exists; leakage not reported in the accessible abstract.", citation="OpenAlex abstract"),
    _s(source_id="R17", publication="Chang et al., 'A 7nm 256Mb SRAM in high-k metal-gate FinFET technology with write-assist circuitry for low-VMIN applications', ISSCC 2017 12.1", year="2017", institution_or_vendor="TSMC", process_node="7 nm FinFET", macro_capacity="256 Mb", bitcell_type="HD 6T 0.027 um2",
       measured_or_simulated="measured", silicon_or_model="silicon", tier="T1 (no leakage figure in the accessible abstract)", citation="OpenAlex abstract"),
    _s(source_id="R18", publication="Unverified secondary claim: '279 mV minimum data-retention voltage' attributed to the TSMC 5 nm 135 Mb SRAM (JSSC 2020, EXP-0008 E01)", process_node="5 nm FinFET", retention_voltage="279 mV (UNVERIFIED: not in the paper's abstract; appeared only in a search-engine summary)", tier="T3 (unverified; NOT used by the model)",
       notes="Kept so the gap is visible: no verified modern-node DRV figure was found.", citation="search summary only"),
]

ASSUMPTIONS = [
    dict(source_id="B1", quantity="Active-standby leakage at 25 C, nominal voltage, macro incl. periphery (mW per decimal MB)",
         value="LOW 0.21 (R01, 3 nm, 1-CLM); PLANNING 0.33 (geometric mean of R01 and R02); HIGH 0.52 (R02, 7 nm UHD). EXP-0008 used 0.60 flat for its 'hot nominal' planning; it is reproduced as a separate input, not as this model's planning value.",
         label="T1-anchored (two macro papers)"),
    dict(source_id="B2", quantity="Temperature scaling of leakage", value="ASAP7 6T cell SPICE (R13): at 0.70 V the hold leakage rises 1.6x (50 C), 3.4x (75 C), 4.9x (85 C), 10.2x (105 C) relative to 25 C; at 0.35 V it rises 3.1x / 9.6x / 14.9x / 34x. Old-node measured retention modes (R05 1.46x, R06 1.50x per 10 C) agree with the low-voltage curve. Applied as interpolated multipliers; LABEL: MODELED (T2) for FinFET nominal-voltage standby, corroborated by T1 old-node data for retention modes.",
         label="T2 modeled + T1 old-node corroboration"),
    dict(source_id="B3", quantity="Periphery share of macro standby leakage", value="0.30 (R01: bitline prechargers alone are 10%; decoders, sense amplifiers, drivers and clocking assumed to bring the periphery to ~30%). Range 0.15-0.45.",
         label="ASSUMPTION"),
    dict(source_id="B4", quantity="Retention modes and their leakage ratio to active standby at the same temperature",
         value="LIGHT_SLEEP (BL precharge off, array nominal, periphery on): 0.90 (R01 T1). DEEP_SLEEP_NOMINAL (periphery off, array at nominal V): 1 - B3 = 0.70 (ASSUMPTION; upper end of R04). RETENTION_VRET (periphery off, array at V_ret): (1 - B3) x array power ratio from R13, which is temperature dependent: at 0.35 V 0.108 (25 C), 0.33 (85 C) -> 0.076 / 0.23 total; at 0.45 V 0.22 / 0.47 -> 0.15 / 0.33 total. R03 (7 nm PDK, 0.25) sits between. OPTIMISTIC_BODY_BIAS: 0.013 (R05 T1, FD-SOI with reverse body bias; not available in FinFET) kept as a bound only. SHUTDOWN (0.002, R03) loses data and is never used as retention.",
         label="T2 modeled ratios anchored by T1 (R01, R05) and T2 (R03)"),
    dict(source_id="B5", quantity="Retention voltage", value="PLANNING 0.35 V (ASAP7 hold state converges and keeps hold SNM > 0 at 0.35 V across 25-105 C per hold_snm table; no verified modern-node DRV figure found, R18 excluded); sweep 0.25-0.50 V. Retention at 0.25 V is marginal at 105 C (q-low node rises above 0.2 V in SPICE).",
         label="T2 modeled; no T1"),
    dict(source_id="B6", quantity="Wake latency", value="per-macro wake t_macro: LOW 10 ns (R03 10 cycles at ~1 GHz), PLANNING 200 ns (R05, 128 KiB-class macro with in-rush limiting), HIGH 1 us (ASSUMPTION for a 256 KB macro at a tighter in-rush limit); macros per domain wake in groups of LOW 64 / PLANNING 8 / HIGH 1 (in-rush limit, ASSUMPTION); floor 2 ns (R01 one cycle). Light sleep exits in one cycle (R01).",
         label="ASSUMPTION ranges anchored on R05 (T1, 22 nm) and R03 (T2)"),
    dict(source_id="B7", quantity="Wake energy per MB of domain", value="array rail capacitance ~4 nF per MB (8e6 cells x ~0.5 fF per cell to the cell rail: single-fin PU source/junction and PU gate at EOT 1 nm; KNOWN physics, order-of-magnitude); full array-rail swing 0.35 -> 0.75 V draws E = C x Vdd x dV = 1.2 uJ/MB (HIGH, applies to RETENTION_VRET exit); periphery-only restore (DEEP_SLEEP_NOMINAL exit) ~12% of that = 0.15 uJ/MB (PLANNING); LOW 0.02 uJ/MB (R03-class, 4 KB segments at 41-cycle break-even). Sequencer 10 pJ per macro. Sleep entry: 0 (stored rail energy dissipates). No measured modern-node macro wake energy exists.",
         label="ASSUMPTION (physics-based estimate, R11 method)"),
    dict(source_id="B8", quantity="Voltage-domain granularity", value="one power switch set per macro (256 KB, EXP-0008 A9) is supported by every compiler source (R04 per instance; R03 per 4 KB segment; R05 per 4 KiB macro with 2.1% area); layer domains are groups of macros switched together; the model sweeps domain = 1 / 2 / 4 / 8 layers / whole model. Independent array-voltage regulation per macro is NOT assumed: V_ret is a shared retention rail per domain.",
         label="T1/T2/T3 supported at macro granularity; domain grouping ASSUMPTION"),
    dict(source_id="B9", quantity="Engine timing for wake slack", value="layer execution time in burst mode = layer bytes / engine bandwidth with engine bandwidth 8 TB/s (EXP-0006 tile aggregate, TARGET) and 1 TB/s sensitivity; in paced mode the token period is spread evenly over the L layers (layer time = 1 / (R x L)). No physical clock is claimed; EXP-0005 RTL is cycle-exact but unclocked.",
         label="TARGET / ASSUMPTION"),
    dict(source_id="B10", quantity="LPDDR5X baseline", value="unchanged from EXP-0008 (A6/A7/A13): 4.0 pJ/bit planning path; channel-busy fraction at IDD2N, rest IDD2P (5.4 mW); between sessions self-refresh (0.77 mW at 25 C, 9.8 mW at 85 C, log-interpolated) when the idle gap is >= 1 s, power-down otherwise; in the duty-cycle sweep the idle fraction is charged the cheaper of power-down and self-refresh at the idle temperature.",
         label="carried (T1 device + T3 PHY)"),
    dict(source_id="B11", quantity="Dynamic energies", value="resident 0.4405 pJ/bit and LPDDR 4.0 pJ/bit planning (EXP-0008); retention does not change bank composition or delivery, so they are reused unchanged.", label="carried"),
    dict(source_id="B13", quantity="Idle temperature", value="an idle edge device cools; duty-cycle and session rows at 85 C active are also given with the idle periods at 40 C (ASSUMPTION; junction temperature of an idle low-power SoC). Default rows keep one temperature for both.", label="ASSUMPTION"),
    dict(source_id="B14", quantity="Shutdown-and-reload idle policy (PX)", value="NOT a retention mode (data lost). Between sessions the array is powered off (residual switch leakage 0.002 x active, R03) and the weights are reloaded from external storage at the next session: E_reload = EXP-0008 cold-start energy (weights x 8 x (4.0 pJ/bit LPDDR + 0.09 pJ/bit x 1.65 SRAM write)) = 13.2 mJ for Falcon-E 1B; first-token latency = reload time at 1-4 GB/s (0.1-0.4 s). Priced as a comparison row so retention is judged against the obvious alternative.", label="EXP-0008 A10 carried"),
    dict(source_id="B15", quantity="LPDDR background temperature", value="JEDEC/Micron IDD values are specified at the maximum operating case temperature, so the IDD2N (32.1 mW) and IDD2P (5.4 mW) charged during generation are hot-corner maxima at every temperature while the SRAM side uses 25 C measurements at 25 C. This favours residency at 25 C by up to ~27 mW of the 345 mW LPDDR total (Falcon-E 1B at 25 tok/s). Planning is left as EXP-0008 defined it; a sensitivity row halves IDD2N/IDD2P at 25 C.", label="disclosure"),
    dict(source_id="B12", quantity="Session / usage model", value="sessions of N tokens at R tok/s separated by idle gaps G; duty cycle d = active time fraction at 25 tok/s; these are usage sweeps, not forecasts.", label="sweep definition"),
]

# ----------------------------------------------------------------------------- constants (every number tagged)
ACTIVE_LEAK_25C_MW_PER_MB = {"low": 0.21, "planning": 0.33, "high": 0.52}   # B1 (R01, R02)
EXP0008_LEAK_MW_PER_MB = 0.60                                                # EXP-0008 A4 planning (reproduction input only)
PERIPHERY_SHARE = {"low": 0.45, "planning": 0.30, "high": 0.15}             # B3 (R01) — 'low' = most periphery = most saved when it is switched off
TEMPS_C = (25, 50, 75, 85)                                                   # EXP-0009 prompt sweep points
# ASAP7 TT 6T hold leakage current per cell, A (R13); keys: (V, T)
ASAP7_CELL_A = {                                                             # R13 (ASAP7 TT SPICE, analysis/sram_retention/spice)
    (0.25, 25): 3.81e-12, (0.30, 25): 4.55e-12, (0.35, 25): 5.63e-12, (0.40, 25): 6.65e-12, (0.45, 25): 8.81e-12, (0.50, 25): 1.154e-11, (0.60, 25): 1.819e-11, (0.70, 25): 2.592e-11, (0.75, 25): 3.058e-11, (0.80, 25): 3.531e-11,
    (0.25, 50): 1.435e-11, (0.30, 50): 1.569e-11, (0.35, 50): 1.762e-11, (0.40, 50): 1.944e-11, (0.45, 50): 2.200e-11, (0.50, 50): 2.461e-11, (0.60, 50): 3.172e-11, (0.70, 50): 4.150e-11, (0.75, 50): 4.661e-11, (0.80, 50): 5.177e-11,
    (0.25, 75): 4.809e-11, (0.30, 75): 5.070e-11, (0.35, 75): 5.377e-11, (0.40, 75): 5.690e-11, (0.45, 75): 6.082e-11, (0.50, 75): 6.520e-11, (0.60, 75): 7.606e-11, (0.70, 75): 8.868e-11, (0.75, 75): 9.550e-11, (0.80, 75): 1.034e-10,
    (0.25, 85): 7.580e-11, (0.30, 85): 7.964e-11, (0.35, 85): 8.362e-11, (0.40, 85): 8.765e-11, (0.45, 85): 9.294e-11, (0.50, 85): 9.834e-11, (0.60, 85): 1.120e-10, (0.70, 85): 1.265e-10, (0.75, 85): 1.351e-10, (0.80, 85): 1.439e-10,
    (0.25, 105): 1.773e-10, (0.30, 105): 1.849e-10, (0.35, 105): 1.924e-10, (0.40, 105): 2.003e-10, (0.45, 105): 2.093e-10, (0.50, 105): 2.189e-10, (0.60, 105): 2.398e-10, (0.70, 105): 2.636e-10, (0.75, 105): 2.766e-10, (0.80, 105): 2.901e-10,
}
ASAP7_VNOM = 0.70                                                            # R13 (ASAP7 nominal supply)
RETENTION_VOLTAGES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)                    # EXP-0009 prompt sweep points
V_RET_PLANNING = 0.35                                                        # B5
LIGHT_SLEEP_RATIO = 0.90                                                     # B4 (R01)
REGATE_SLEEP_RATIO = 0.25                                                    # R03 (T2, 7 nm PDK) — cross-check only
BODY_BIAS_RATIO = 0.013                                                      # R05 (T1, FD-SOI) — optimistic bound only
SHUTDOWN_RATIO = 0.002                                                       # R03 — data lost; never used as retention
MACRO_BYTES = 256e3                                                          # EXP-0008 A9 macro size
WAKE_T_MACRO_S = {"low": 10e-9, "planning": 200e-9, "high": 1e-6}            # B6 (R03, R05)
WAKE_PARALLEL_MACROS = {"low": 64, "planning": 8, "high": 1}                 # B6
WAKE_T_FLOOR_S = 2e-9                                                        # B6 (R01)
WAKE_UJ_PER_MB = {"low": 0.02, "planning": 0.15, "high": 1.2}               # B7
WAKE_UJ_PER_MB_ARRAY_SWING = 1.2                                             # B7 (RETENTION_VRET exit)
WAKE_SEQUENCER_J = 10e-12                                                    # B7
WAKE_SWEEP_S = (0.1e-6, 0.5e-6, 1e-6, 5e-6, 10e-6, 50e-6, 100e-6)            # EXP-0009 prompt sweep points
ENGINE_BW_BPS = {"target": 8e12, "slow": 1e12}                               # B9 (EXP-0006 TARGET)
TOKEN_RATES = (1, 5, 10, 25, 50, 100, 250)                                   # EXP-0009 prompt sweep points
DUTY_CYCLES = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)                            # EXP-0009 prompt sweep points
SESSION_TOKENS = (16, 64, 256, 1024)                                         # EXP-0009 prompt sweep points
IDLE_GAPS_S = (0.0, 1.0, 10.0, 60.0, 300.0)                                  # EXP-0009 prompt sweep points
TIMEOUTS_S = (0.0, 1e-3, 10e-3, 100e-3, 1.0)                                 # EXP-0009 PT policy sweep points
WINDOWS = (1, 2, 4, 8)                                                       # EXP-0009 P1/P2/P4/P8
SELF_REFRESH_GAP_S = 1.0                                                     # B10
IDLE_TEMP_C = 40                                                             # B13
RELOAD_GBPS = (1.0, 4.0)                                                     # B14 (EXP-0008 A10)
PESSIMISTIC_RETENTION_RATIO_25C = 0.23                                       # B4 sensitivity: if real HD cells are subthreshold-dominated at 25 C, the hot ratio applies cold


def write(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sources.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(SOURCES)
    with open(os.path.join(out_dir, "assumptions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "quantity", "value", "label"]); w.writeheader(); w.writerows(ASSUMPTIONS)
    with open(os.path.join(out_dir, "retention_evidence.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "process_node", "tier", "modes_reported", "standby_leakage", "retention_leakage", "deep_sleep_leakage", "leakage_per_MB", "retention_voltage", "temperature", "wake_latency", "wake_energy", "data_retention_guarantee"])
        w.writeheader()
        for s in SOURCES:
            w.writerow({k: s[k] for k in w.fieldnames})
