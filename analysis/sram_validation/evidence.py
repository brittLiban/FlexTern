"""
EXP-0008 sources: modern SRAM read energy, latency, leakage; LPDDR5X device/IO energy, background power,
latency; on-chip wire energy.  Every constant used by model.py is a field here or an explicit ASSUMPTION.

Quality tiers: T1 measured silicon in a primary publication/datasheet; T2 primary simulated/estimated or
projected; T3 credible secondary summary; T4 historical/contextual only.
Energy boundaries are stated per row.  Decimal units; 1 MB = 1e6 B; cells per MB = 8e6.
"""

from __future__ import annotations

import csv
import os

FIELDS = ["source_id", "publication", "year", "vendor_or_institution", "process", "macro_capacity", "macro_dimensions", "read_width",
          "voltage", "frequency", "read_energy_per_access", "derived_energy_per_bit", "leakage_power", "standby_power", "temperature",
          "measured_or_simulated", "silicon_or_model", "commercial_or_research", "included_energy_boundary", "tier", "notes", "citation"]

SOURCES: list[dict] = [
    dict(source_id="E01", publication="IEEE JSSC Dec 2020 (ISSCC 2020 paper 15.1): 'A 5-nm 135-Mb SRAM in EUV and High-Mobility-Channel FinFET Technology with Metal Coupling and Charge-Sharing Write-Assist'", year=2020,
         vendor_or_institution="TSMC (Chang et al.)", process="TSMC N5 FinFET (EUV, high-mobility channel)", macro_capacity="135 Mb test chip of HD macros (per-macro size not stated in the abstract)", macro_dimensions="",
         read_width="not stated in abstract", voltage="0.4 V (energy figure)", frequency="not stated in abstract", read_energy_per_access="",
         derived_energy_per_bit="32.8 fJ/bit read, 54.1 fJ/bit write (at 0.4 V; abstract)", leakage_power="not stated in abstract", standby_power="", temperature="",
         measured_or_simulated="measured", silicon_or_model="silicon", commercial_or_research="commercial process, research publication",
         included_energy_boundary="SRAM macro (cell array + macro periphery) at low voltage", tier="T1",
         notes="Lowest-voltage operating point; at nominal ~0.75 V the V^2 scaling gives ~115 fJ/bit (INFERRED, A2).", citation="https://ieeexplore.ieee.org/document/9290031/ ; https://ieeexplore.ieee.org/document/9062967/"),
    dict(source_id="E02", publication="2020 IEEE Symposium on VLSI Circuits: 'A 29.2 Mb/mm2 Ultra High Density SRAM Macro using 7nm FinFET Technology with Dual-Edge Driven Wordline/Bitline and Write/Read-Assist Circuit'", year=2020,
         vendor_or_institution="Yokoyama, Tanaka, Yabuuchi, Ishii et al. (Renesas-affiliated authors)", process="7 nm FinFET", macro_capacity="256 Kb single array (512 rows x 512 columns)", macro_dimensions="29.2 Mb/mm2",
         read_width="not stated in abstract (64-256 bits assumed, A3)", voltage="not stated in abstract (nominal assumed)", frequency="access time 0.82 ns", read_energy_per_access="5.8 uW/MHz read dynamic power = 5.8 pJ per access",
         derived_energy_per_bit="23-91 fJ/bit for 256-64-bit widths (INFERRED, A3)", leakage_power="not stated", standby_power="", temperature="",
         measured_or_simulated="measured (test chip)", silicon_or_model="silicon", commercial_or_research="research publication",
         included_energy_boundary="SRAM macro (array + periphery) per access", tier="T1 (energy per access); width INFERRED", notes="Also the modern access-time anchor: 0.82 ns.", citation="https://www.researchgate.net/publication/343565282 ; secondary: web search summary"),
    dict(source_id="E03", publication="ISSCC 2025 paper 29.2 / IEEE JSSC 2025: '0.021-um2 High-Density SRAM in Intel 18A RibbonFET Technology with PowerVia Backside Power Delivery'", year=2025,
         vendor_or_institution="Intel", process="Intel 18A RibbonFET (GAA)", macro_capacity="HDC and HCC macros (sizes not in abstract)", macro_dimensions="38.1 Mb/mm2 array density (HDC)",
         read_width="", voltage="1.05 V (5.6 GHz point)", frequency="up to 5.6 GHz (HCC macro)", read_energy_per_access="not stated", derived_energy_per_bit="",
         leakage_power="not stated", standby_power="", temperature="", measured_or_simulated="measured (volume silicon validated)", silicon_or_model="silicon", commercial_or_research="commercial process",
         included_energy_boundary="n/a (frequency only)", tier="T1 (frequency/density)", notes="Cycle time 0.18 ns at 5.6 GHz: modern SRAM cycle anchor.", citation="https://www.researchgate.net/publication/396965964 ; https://community.intel.com/t5/Blogs/Tech-Innovation/Edge-5G/Intel-at-ISSCC-2025-Navid-Shahriari-Invited-Talk-Eight-Papers/post/1667592"),
    dict(source_id="E04", publication="Keckler, Dally, Khailany, Garland, Glasco, 'GPUs and the Future of Parallel Computing', IEEE Micro 2011, Table 1/2", year=2011,
         vendor_or_institution="NVIDIA", process="40 nm (measured-class) and 10 nm (projected)", macro_capacity="on-chip SRAM (64-bit read)", macro_dimensions="",
         read_width="64 bits", voltage="", frequency="", read_energy_per_access="SRAM 14 pJ per 64-bit read at 40 nm; 2.4 / 1.8 pJ projected at 10 nm (high/low)",
         derived_energy_per_bit="SRAM 219 fJ/bit (40 nm); 37 / 28 fJ/bit (10 nm projection). Wire 150 fJ/bit/mm (40 nm), 115 fJ/bit/mm (10 nm). DRAM interface 20-30 pJ/bit (45 nm) -> 2 pJ/bit (16 nm); DRAM access 8-15 pJ/bit (45 nm) -> 2.5 pJ/bit (16 nm)",
         leakage_power="", standby_power="", temperature="", measured_or_simulated="estimated/projected", silicon_or_model="model", commercial_or_research="research",
         included_energy_boundary="SRAM array read; global wire per bit per mm; DRAM interface and access separately", tier="T2 (projection) / T4 (40 nm)", notes="Wire energy per bit per mm is the only sourced local-delivery figure; node scaling of wires is weak, which the paper itself notes.", citation="https://www.cs.toronto.edu/~pekhimenko/courses/csc2231-f17/Papers/GPUs.pdf"),
    dict(source_id="E05", publication="ISSCC 2014-class 28 nm HKMG embedded SRAM paper: 'A 0.41 uA standby leakage 32Kb embedded SRAM with low-voltage resume-standby' (as summarized)", year=2014,
         vendor_or_institution="28 nm HKMG CMOS", process="28 nm HKMG", macro_capacity="32 Kb", macro_dimensions="", read_width="", voltage="0.5 V idle; 0.46 V data-retention", frequency="",
         read_energy_per_access="", derived_energy_per_bit="", leakage_power="296 nW/kB at 0.5 V idle = 0.296 mW/MB; 0.23 nW/kB at 0.46 V DRV sleep = 0.00023 mW/MB", standby_power="", temperature="not stated (room assumed)",
         measured_or_simulated="measured", silicon_or_model="silicon", commercial_or_research="research", included_energy_boundary="macro leakage incl. periphery", tier="T1 (28 nm; NOT modern-node)",
         notes="Only quantitative measured leakage per capacity found; 28 nm planar HKMG, low voltage. Used as the planning anchor with a modern-node range around it (A4).", citation="https://www.researchgate.net/publication/260627237"),
    dict(source_id="E06", publication="Micron LPDDR5X SDRAM datasheet (Y52Q 16 Gb die, 315b/441b/561b parts), IDD specification tables", year=2023,
         vendor_or_institution="Micron", process="1-beta class DRAM", macro_capacity="16 Gb die, dual x16 channels", macro_dimensions="", read_width="x16 per channel",
         voltage="VDD1 1.80 V, VDD2H 1.05 V, VDD2L 0.90 V, VDDQ 0.50 V (TYP)", frequency="8533 Mb/s per pin (-023 speed grade)",
         read_energy_per_access="IDD4R at 8533 Mb/s, x16 mode, single channel of 2-channel die: IDD4R1 16 mA (VDD1), IDD4R2H 382 mA (VDD2H), IDD4R2L 0.2 mA, IDD4RQ 121 mA (VDDQ, typical, 5 pF load, RON 40 ohm, 25 C)",
         derived_energy_per_bit="(16x1.80 + 382x1.05 + 0.2x0.90 + 121x0.50) mW = 490 mW per channel at 17.07 GB/s = 3.59 pJ/bit device-side (INFERRED arithmetic; IDD4R2 are worst-case maxima)",
         leakage_power="IDD6 self-refresh per channel: 25 C 0.17/0.42/0.01/0.03 mA (VDD1/VDD2H/VDD2L/VDDQ) = 0.77 mW; 45 C 1.2 mW; 65 C 2.9 mW; 85 C 9.8 mW. IDD2N idle standby (clocks on) 1.3/28/0.2/0.6 mA = 32.1 mW; IDD2P power-down 1.4/2.3/0.2/0.6 mA = 5.4 mW",
         standby_power="see leakage_power", temperature="25-85 C", measured_or_simulated="datasheet specification (max except where noted typical)", silicon_or_model="silicon", commercial_or_research="commercial",
         included_energy_boundary="DRAM device: array + peripheral + device I/O drivers; EXCLUDES SoC PHY, controller and board loss", tier="T1 (vendor datasheet)",
         notes="IDD4R2H is a worst-case maximum under a stringent pattern; typical device energy is lower. IDD6 values at 25/45/65 C are typical, 85 C is max.", citation="https://mm.digikey.com/Volume0/opasdata/d220001/medias/docus/8885/315b-441b-561b-y52q-sdp-ddp-lpddr5x.pdf"),
    dict(source_id="E07", publication="Industrial Monitor Direct knowledge-base article on on-package memory power (secondary)", year=2024,
         vendor_or_institution="secondary", process="", macro_capacity="", macro_dimensions="", read_width="", voltage="", frequency="",
         read_energy_per_access="", derived_energy_per_bit="LPDDR5X soldered on-board ~3-5 pJ/bit at the PHY level; on-package 1-2 pJ/bit; LPCAMM2 ~4; SODIMM DDR5 ~12 (estimates)",
         leakage_power="", standby_power="", temperature="", measured_or_simulated="estimate", silicon_or_model="model", commercial_or_research="secondary",
         included_energy_boundary="stated as 'PHY level' (unclear whether device included)", tier="T3", notes="Used only to bracket the SoC PHY + controller share above the T1 device figure.", citation="https://industrialmonitordirect.com/blogs/knowledgebase/on-package-memory-vs-off-chip-dimm-power-consumption-analysis"),
    dict(source_id="E08", publication="Micron LPDDR5X marketing figure (EXP-0006 S13): ~110 mW per GB active at 8533 MT/s", year=2024,
         vendor_or_institution="Micron", process="", macro_capacity="", macro_dimensions="", read_width="", voltage="", frequency="",
         read_energy_per_access="", derived_energy_per_bit="unit basis ambiguous (per GB of capacity or per GB/s); not used numerically", leakage_power="", standby_power="", temperature="",
         measured_or_simulated="marketing", silicon_or_model="", commercial_or_research="commercial", included_energy_boundary="unclear", tier="T3",
         notes="Superseded by E06 IDD arithmetic.", citation="https://s25.q4cdn.com/621799436/files/doc_downloads/white_papers/lpddr-gen-purp-ai-wkloads-in-lg-scale-dc-deploy-white-paper-4.pdf"),
    dict(source_id="E09", publication="JEDEC JESD209-6 LPDDR6 (July 2025) and vendor claims (Samsung 21% efficiency gain over LPDDR5X; SK hynix 20% lower power)", year=2025,
         vendor_or_institution="JEDEC / Samsung / SK hynix", process="10 nm-class DRAM", macro_capacity="16 Gb", macro_dimensions="", read_width="", voltage="lower than LPDDR5X (DVFSL)", frequency="up to 14.4 Gb/s",
         read_energy_per_access="", derived_energy_per_bit="~0.8x LPDDR5X (vendor claims)", leakage_power="", standby_power="", temperature="", measured_or_simulated="vendor claim", silicon_or_model="",
         commercial_or_research="commercial (standard published; parts sampling)", included_energy_boundary="relative claim", tier="T3", notes="Sensitivity only.", citation="https://www.jedec.org/node/9555 ; https://www.tomshardware.com/pc-components/dram/sk-hynix-introduces-turbocharged-lpddr6-33-percent-faster-and-20-percent-more-power-efficient-than-lpddr5x-16gb-chips-deliver-10-7-gbps-uses-10nm-node"),
    dict(source_id="E10", publication="Horowitz, ISSCC 2014 1.1 (EXP-0006 S15)", year=2014, vendor_or_institution="Stanford", process="45 nm", macro_capacity="8 KB-1 MB caches", macro_dimensions="",
         read_width="64 bits", voltage="0.9 V", frequency="", read_energy_per_access="10-20 pJ per 64-bit cache access; 100 pJ (1 MB, reproduced figure)", derived_energy_per_bit="0.16-1.6 pJ/bit",
         leakage_power="", standby_power="", temperature="", measured_or_simulated="rough estimates", silicon_or_model="model", commercial_or_research="research", included_energy_boundary="cache access", tier="T4 (historical)",
         notes="The old anchor EXP-0006 relied on; context only.", citation="https://pages.cs.wisc.edu/~markhill/restricted/isscc2014_horowitz_power_scaling.pdf"),
    dict(source_id="E11", publication="Secondary statements on FinFET SRAM cell standby current ('sub-100 pA per cell', '80 pA/cell 4T FinFET research cell')", year=2020,
         vendor_or_institution="secondary / research", process="FinFET (unspecified node)", macro_capacity="", macro_dimensions="", read_width="", voltage="", frequency="", read_energy_per_access="",
         derived_energy_per_bit="", leakage_power="100 pA/cell x 8e6 cells/MB x 0.75 V = 0.6 mW/MB (INFERRED upper-class figure)", standby_power="", temperature="", measured_or_simulated="mixed",
         silicon_or_model="", commercial_or_research="", included_energy_boundary="cell only (no periphery)", tier="T3", notes="Brackets the high end of the modern leakage range (A4).", citation="web search summary (PMC9416021 and related)"),
    dict(source_id="E12", publication="JEDEC LPDDR5X typical timings (secondary engineering references): tRCD ~18 ns, tRPab ~21 ns, read latency ~15-20 ns", year=2023,
         vendor_or_institution="secondary", process="", macro_capacity="", macro_dimensions="", read_width="", voltage="", frequency="", read_energy_per_access="", derived_energy_per_bit="",
         leakage_power="", standby_power="", temperature="", measured_or_simulated="standard values", silicon_or_model="", commercial_or_research="", included_energy_boundary="",
         tier="T3", notes="Device-level first access ~36-40 ns; SoC loaded latency ~60-120 ns (A6).", citation="https://www.synopsys.com/zh-cn/blogs/chip-design/lpddr5x-specification-memory-design.html"),
]

ASSUMPTIONS: list[dict] = [
    dict(source_id="A1", quantity="Modern SRAM macro read energy (fJ/bit, macro boundary: array + macro periphery)",
         value="LOW 33 (E01 at 0.4 V); PLANNING 90 (E02 at an assumed 64-bit width and E01 scaled to ~0.75 V; E04 10 nm projection 28-37 is below this); HIGH 150 (E02 at 64 bits with margin for larger macros / wide-column activation)",
         label="T1-anchored range; width and voltage INFERRED"),
    dict(source_id="A2", quantity="Voltage scaling of SRAM dynamic energy", value="E ~ V^2: 0.4 V -> 0.75 V multiplies by 3.5", label="KNOWN physics (CV^2), applied to E01"),
    dict(source_id="A3", quantity="Read width of the E02 macro", value="64-256 bits (5.8 pJ per access -> 91-23 fJ/bit)", label="ASSUMPTION"),
    dict(source_id="A4", quantity="Modern SRAM leakage per provisioned MB (mW/MB, macro incl. periphery, always on at the operating voltage)",
         value="LOW 0.05 (low-leakage HD cells at reduced retention-capable voltage, 25 C); PLANNING 0.60 (E11 ~100 pA/cell FinFET class at ~0.75 V nominal: 100 pA x 8e6 cells/MB x 0.75 V); HIGH 1.0 (hot corner, 85 C). The 28 nm measured 0.296 mW/MB at 0.5 V (E05) is a reduced-voltage planar point and is NOT used as the modern planning figure. Retention mode (data kept, no access) factors 1x (none), 0.2x, 0.1x are HYPOTHESIZED: E05 shows 1000x at 28 nm, FinFET retention gains are smaller and unmeasured here",
         label="ASSUMPTION: NO modern-node measured leakage per Mb was found; retention factors HYPOTHESIZED"),
    dict(source_id="A5", quantity="Local delivery energy (SRAM macro output -> consumer registers), pJ/bit", value="wire 0.115-0.15 pJ/bit/mm (E04); distances swept: short bank 0.5 mm, medium tile 2 mm, long global 8 mm -> 0.06-0.08 / 0.23-0.30 / 0.92-1.2 pJ/bit; plus 0.02 pJ/bit for registers/mux (ASSUMPTION)",
         label="T2 wire figure (projection) x ASSUMED distances; no floorplan"),
    dict(source_id="A6", quantity="LPDDR5X energy per bit by boundary (pJ/bit)", value="device: E06 IDD4R2H is a datasheet MAXIMUM under a stringent pattern (datasheet note 6) = 3.6 pJ/bit ceiling; typical device assumed 0.7x = 2.5 (ASSUMPTION); SoC PHY + controller: 1.0-2.5 (E07 bracket, E04 2 pJ/bit 16 nm projection). Levels built the same way as the SRAM side (typical for LOW/PLANNING, maximum for HIGH): LOW 3.5 (typical device + 1.0 PHY), PLANNING 4.0 (typical device + 1.5 PHY), HIGH 6.1 (max device + 2.5 PHY)",
         label="T1 device + T2/T3 PHY; boundary 'full external path to the SoC consumer'"),
    dict(source_id="A7", quantity="LPDDR5X background power (per x16 channel, 2 GB)", value="self-refresh 0.77 mW (25 C) to 9.8 mW (85 C) (E06 IDD6); power-down 5.4 mW (IDD2P); idle standby 32 mW (IDD2N); PHY idle not modeled. Channels: one for Falcon-E 1B/BitNet (<= 2 GB), two for 3B head+weights case if BF16 head; the model uses 1 channel of 2 GB for all three (weights + head fit)",
         label="T1 device figures; SoC-side idle ASSUMED 0"),
    dict(source_id="A8", quantity="Latencies", value="SRAM macro access 0.82 ns (E02), cycle 0.18-0.5 ns (E03); LPDDR5X device first access ~36-40 ns (E12), SoC loaded ~60-120 ns (ASSUMPTION); engine consumes sequentially at B bytes/cycle with a prefetch depth of 4 KB (ASSUMPTION)",
         label="T1/T3 + ASSUMPTION"),
    dict(source_id="A9", quantity="Logical bank decomposition", value="6.2 MB logical bank (64 banks, EXP-0005/0006) composed of 256 KB-2 MB macros; one macro activated per access; wide reads 128-2048 bits per access; decoder/address overhead amortized over the width (per-access fixed cost 0.5-1.5 pJ, ASSUMPTION)",
         label="ASSUMPTION"),
    dict(source_id="A10", quantity="Cold start", value="weights loaded once from external storage at 1-4 GB/s (ASSUMPTION), energy at LPDDR planning pJ/bit + SRAM write ~1.6x read (E01 write/read ratio)", label="ASSUMPTION"),
    dict(source_id="A11", quantity="Capacity margin", value="10% physical margin on resident capacity (EXP-0006 A2)", label="ASSUMPTION (carried)"),
    dict(source_id="A12", quantity="Bank-composition overhead (pJ/bit): bank-level address decode, macro-select tree and output multiplexing when a 3-25 MB bank is built from 256 KB-2 MB macros (E02 is a 256 Kb macro; E01 is a per-bit figure)", value="LOW 0.05, PLANNING 0.10, HIGH 0.15", label="ASSUMPTION; the price of composing multi-MB banks from sub-MB macros"),
    dict(source_id="A13", quantity="LPDDR idle during the active token period", value="the channel is busy for W x R / (17.07 GB/s x 70%) of each second; the busy fraction is charged IDD2N (idle standby, clocks on) and the remainder IDD2P (power-down), as a memory controller would", label="ASSUMPTION (E06 figures)"),
]


def write(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sources.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(SOURCES)
    with open(os.path.join(out_dir, "assumptions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "quantity", "value", "label"]); w.writeheader(); w.writerows(ASSUMPTIONS)


# ----------------------------------------------------------------------------- constants (tagged)
SRAM_MACRO_FJ_PER_BIT = {"low": 33.0, "planning": 90.0, "high": 150.0}        # A1 (E01, E02, E04)
WIRE_PJ_PER_BIT_PER_MM = (0.115, 0.15)                                         # E04 (10 nm projection, 40 nm)
DISTANCES_MM = {"short_bank": 0.5, "medium_tile": 2.0, "long_global": 8.0}      # A5
REGISTER_PJ_PER_BIT = 0.02                                                     # A5
PER_ACCESS_FIXED_PJ = (0.5, 1.5)                                               # A9
READ_WIDTHS = (128, 256, 512, 1024, 2048)                                      # A9
MACRO_SIZES_BYTES = (256e3, 512e3, 1e6, 2e6)                                   # A9
LEAK_MW_PER_MB = {"low": 0.05, "planning": 0.60, "high": 1.0}                   # A4 (E11; E05 context)
BANK_OVERHEAD_PJ_PER_BIT = {"low": 0.05, "planning": 0.10, "high": 0.15}      # A12
RETENTION_FACTORS = (1.0, 0.2, 0.1)                                            # A4 (0.2 and 0.1 are HYPOTHESIZED)
LPDDR_PJ_PER_BIT = {"device_max": 3.59, "device_typ": 2.5, "low": 3.5, "planning": 4.0, "high": 6.1}   # A6 (E06, E07, E04)
LPDDR_BACKGROUND_MW = {"self_refresh_25C": 0.77, "self_refresh_85C": 9.8, "power_down": 5.4, "idle_standby": 32.1}   # A7 (E06)
LPDDR_CHANNEL_GBPS = 17.07                                                     # E06: 8533 Mb/s x 16 bits
LPDDR_LATENCY_NS = {"device_first_access": 38.0, "soc_loaded": 90.0}           # A8 (E12)
SRAM_LATENCY_NS = {"macro_access": 0.82, "cycle": 0.25}                        # E02, E03
PREFETCH_BYTES = 4096                                                          # A8
COLD_START_GBPS = (1.0, 4.0)                                                   # A10
SRAM_WRITE_OVER_READ = 54.1 / 32.8                                             # E01
MARGIN = 0.10                                                                  # A11
OLD_SRAM_PLANNING_PJ = 1.6                                                     # E10 (EXP-0006 planning figure, for reproduction only)
OLD_LPDDR_RANGE_PJ = (5.0, 15.0)                                               # EXP-0006 A3 (for reproduction only)
DUTY_CYCLES = (0.01, 0.05, 0.10, 0.25, 0.50, 1.0)                           # EXP-0008 prompt sweep points
TOKEN_RATES = (1, 5, 10, 25, 50, 100, 250)                                     # EXP-0008 prompt sweep points (TARGET rates)
HEAD_CANDIDATE_BITS, KV_CANDIDATE_BITS = 4.156, 4.5                            # EXP-0004
