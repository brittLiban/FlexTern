"""
EXP-0006 source table: every physical constant used by the memory-residency models, with provenance.

Rules (AGENTS.md §7, EXP-0006 prompt):
  * No numeric physical constant may appear in the models unless it is a field of a SOURCES row
    (traceable to a publication / vendor document) or an ASSUMPTIONS row (explicit, labeled).
  * Tiers are never mixed: raw bit-cell density != macro density != cache-only die density
    != whole SRAM-heavy die density.  Each row says which it is.
  * Vendor peak bandwidth is recorded as peak; utilization factors are applied in bandwidth.py.
  * Units: decimal (1 MB = 1e6 B, 1 GB = 1e9 B, 1 Mb = 1e6 bit).  Capacities quoted by vendors in
    binary units are converted and the original quoted.

Labels: KNOWN (published physics/standard/spec), MEASURED-PUBLISHED (a shipping part's published
figure), MODELED-PUBLISHED (a paper's modeled figure), VENDOR-CLAIM, INFERRED (derived here),
ASSUMPTION (set here, no external evidence).
"""

from __future__ import annotations

import csv
import os

MB = 1e6
GB = 1e9

# source_id, technology, publication, year, process_node, capacity, bank_count, reported_area, reported_density,
# reported_bandwidth, reported_energy, voltage, measured_or_simulated, commercial_or_research, notes, citation
SOURCES: list[dict] = [
    dict(source_id="S01", technology="SRAM high-density bit cell (6T), TSMC N5", publication="TSMC N5 technology (IEDM 2019 / ISSCC 2020 disclosures, summarized by WikiChip)", year=2020,
         process_node="N5", capacity="", bank_count="", reported_area="0.021 um2 per bit (cell)", reported_density="47.6 Mb/mm2 raw bit-cell (INFERRED = 1/0.021 um2)",
         reported_bandwidth="", reported_energy="", voltage="", measured_or_simulated="published cell size", commercial_or_research="commercial process",
         notes="RAW BIT-CELL density only: no periphery, no routing, no redundancy. Must not be used as chip density.", citation="https://fuse.wikichip.org/?p=7048 ; https://fuse.wikichip.org/?p=7375"),
    dict(source_id="S02", technology="SRAM high-density bit cell and macro, TSMC N3/N3E", publication="TSMC IEDM 2022 (SemiWiki report); WikiChip N3E", year=2022,
         process_node="N3 / N3E", capacity="", bank_count="", reported_area="0.0199 um2 (N3) / 0.021 um2 (N3E) per bit (cell)", reported_density="33.55 Mb/mm2 (N3), 31.8 Mb/mm2 (N3E) reported SRAM density (macro-level, ~67% of raw cell density)",
         reported_bandwidth="", reported_energy="", voltage="", measured_or_simulated="published", commercial_or_research="commercial process",
         notes="MACRO density (array + local periphery). Still excludes TSVs, repair, routing, controllers, and any compute.", citation="https://semiwiki.com/semiconductor-manufacturers/tsmc/322688-iedm-2022-tsmc-3nm/ ; https://fuse.wikichip.org/?p=7048"),
    dict(source_id="S03", technology="3D-stacked SRAM cache die (AMD 3D V-Cache gen 1, L3D)", publication="ISSCC 2022 paper 26.4, AMD", year=2022,
         process_node="N7", capacity="64 MiB = 67.1 MB", bank_count="8 slices x 8 MiB", reported_area="41 mm2 die", reported_density="1.64 MB/mm2 (INFERRED 67.1 MB / 41 mm2; cache-only die incl. TSVs)",
         reported_bandwidth=">2 TB/s per slice-pair aggregate (vendor: 2 TB/s between dies)", reported_energy="<1/3 energy per bit of micro-bump 3D (vendor claim, no absolute pJ)", voltage="", measured_or_simulated="measured product", commercial_or_research="commercial",
         notes="CACHE-ONLY DIE density: SRAM + TSVs + periphery, no compute. Hybrid bonding, ~9 um pitch.", citation="https://www.techpowerup.com/292256/amd-details-its-3d-v-cache-design-at-isscc ; https://www.slideshare.net/AMD/3d-vcache"),
    dict(source_id="S04", technology="3D-stacked SRAM cache die (AMD 3D V-Cache gen 2, Zen 4)", publication="AMD disclosures 2023 (ascii.jp roadmap series)", year=2023,
         process_node="N7", capacity="64 MiB = 67.1 MB", bank_count="", reported_area="~36 mm2 die", reported_density="1.86 MB/mm2 (INFERRED 67.1 / 36)",
         reported_bandwidth="2.5 TB/s die-to-die (vendor)", reported_energy="", voltage="", measured_or_simulated="measured product", commercial_or_research="commercial",
         notes="CACHE-ONLY DIE density. One stacked die per CCD in products; deeper SRAM stacks are not a shipping product.", citation="https://ascii.jp/elem/000/004/109/4109130/1/"),
    dict(source_id="S05", technology="Monolithic SRAM-heavy AI die (Graphcore Colossus MK2 GC200)", publication="Hot Chips 32/33 and launch materials", year=2020,
         process_node="N7", capacity="900 MB SRAM", bank_count="1472 tiles", reported_area="823 mm2 die", reported_density="1.09 MB/mm2 (INFERRED 900 / 823; WHOLE DIE incl. 1,472 cores)",
         reported_bandwidth="47.5 TB/s aggregate on-chip SRAM", reported_energy="", voltage="", measured_or_simulated="published product", commercial_or_research="commercial",
         notes="WHOLE SRAM-HEAVY DIE density: the only tier that includes compute. Demonstrates 900 MB on one reticle-class N7 die.", citation="https://www.servethehome.com/?p=55464 ; https://wccftech.com/graphcores-colossus-mk2-gc200-7nm-ai-chip-rivals-nvidia-a100-gpu/amp/"),
    dict(source_id="S06", technology="Wafer-scale SRAM-heavy die (Cerebras WSE-3)", publication="Cerebras launch 2024 / spec sheet", year=2024,
         process_node="N5", capacity="44 GB SRAM", bank_count="900,000 cores", reported_area="46,225 mm2", reported_density="0.95 MB/mm2 (INFERRED 44e9 / 46,225; WHOLE WAFER incl. cores)",
         reported_bandwidth="21 PB/s aggregate", reported_energy="", voltage="", measured_or_simulated="published product", commercial_or_research="commercial",
         notes="WHOLE-DIE density with compute; wafer-scale, not reticle-limited (field stitching).", citation="https://www.hpcwire.com/off-the-wire/cerebras-systems-unveils-worlds-fastest-ai-chip-with-whopping-4-trillion-transistors/"),
    dict(source_id="S07", technology="Monolithic SRAM-heavy AI die (Tesla Dojo D1)", publication="Hot Chips 34 (2022) / Tesla AI Day", year=2022,
         process_node="N7", capacity="440 MB SRAM", bank_count="354 cores x 1.25 MB", reported_area="645 mm2", reported_density="0.68 MB/mm2 (INFERRED 440 / 645; WHOLE DIE incl. cores)",
         reported_bandwidth="400 GB/s load per core (vendor)", reported_energy="", voltage="", measured_or_simulated="published product", commercial_or_research="commercial (internal)",
         notes="WHOLE-DIE density with compute.", citation="https://www.nextplatform.com/2022/08/23/inside-teslas-innovative-and-homegrown-dojo-ai-supercomputer/ ; https://en.wikipedia.org/wiki/Tesla_Dojo"),
    dict(source_id="S08", technology="Monolithic SRAM-heavy AI die (Groq TSP / GroqChip)", publication="ISCA 2020; GroqChip product brief v1.7", year=2020,
         process_node="GF 14 nm", capacity="230 MB SRAM (220 MB in ISCA paper)", bank_count="", reported_area="725 mm2", reported_density="0.32 MB/mm2 (INFERRED 230 / 725; WHOLE DIE; older node)",
         reported_bandwidth="80 TB/s on-die", reported_energy="", voltage="", measured_or_simulated="published product", commercial_or_research="commercial",
         notes="Older node; listed to show the density span across nodes, excluded from the N7-class range.", citation="https://groq.com/wp-content/uploads/2024/08/GroqChip™-Processor-Product-Brief-v1.7.pdf ; https://www.theregister.com/2020/09/29/groq_ai_dev_kit/"),
    dict(source_id="S09", technology="SRAM-centric digital in-memory-compute chiplets (d-Matrix Corsair)", publication="Hot Chips 2025 presentation (Bhoja)", year=2025,
         process_node="N6", capacity="2 GB SRAM per card (2 packages x 4 chiplets = 256 MB per chiplet, INFERRED)", bank_count="8 chiplets", reported_area="not published", reported_density="not published",
         reported_bandwidth="150 TB/s aggregate SRAM; plus 256 GB LPDDR5 at 400 GB/s", reported_energy="", voltage="", measured_or_simulated="published product", commercial_or_research="commercial (2025)",
         notes="Demonstrates the SRAM-chiplet + fabric organization with weights resident across chiplets and an LPDDR tier behind it.", citation="https://hc2025.hotchips.org/assets/program/conference/day2/24_dMatrix_Bhoja_final.pdf ; https://www.servethehome.com/d-matrix-presents-corsair-an-in-memory-computing-architecture-for-inference-at-hot-chips-2025/"),
    dict(source_id="S10", technology="eDRAM die (Intel Crystalwell, 22 nm tri-gate)", publication="ISSCC 2014 'A 1Gb 2GHz embedded DRAM in 22nm tri-gate CMOS'", year=2014,
         process_node="Intel 22 nm", capacity="128 MiB = 134 MB", bank_count="", reported_area="77 mm2 die", reported_density="1.74 MB/mm2 (INFERRED 134 / 77; whole eDRAM die)",
         reported_bandwidth="~50 GB/s to CPU (product)", reported_energy="", voltage="", measured_or_simulated="measured product", commercial_or_research="commercial (discontinued)",
         notes="Density comparable to N7 cache dies at a 2014 node; eDRAM at advanced nodes is not offered by leading foundries today (ASSUMPTION A6).", citation="https://pure.psu.edu/en/publications/a-1gb-2ghz-embedded-dram-in-22nm-tri-gate-cmos-technology/ ; https://chipsandcheese.com/p/broadwells-edram-vcache-before-vcache"),
    dict(source_id="S11", technology="eDRAM L3 (IBM POWER9, 14 nm SOI)", publication="IBM POWER9 disclosures (WikiChip)", year=2017,
         process_node="GF 14 nm SOI", capacity="120 MB L3", bank_count="", reported_area="not isolated", reported_density="cell 0.0174 um2 (bit cell only)",
         reported_bandwidth="", reported_energy="", voltage="", measured_or_simulated="published product", commercial_or_research="commercial",
         notes="Bit-cell figure only; no die-level density usable.", citation="https://fuse.wikichip.org/?p=3383"),
    dict(source_id="S12", technology="HBM3E stacked DRAM", publication="JEDEC JESD238 (HBM3); Micron HBM3E product brief; Samsung HBM3E 12H announcement", year=2024,
         process_node="DRAM 1b-class", capacity="24 GB (8-high) to 36 GB (12-high) per stack", bank_count="16 channels / 32 pseudo-channels", reported_area="stack footprint not used", reported_density="",
         reported_bandwidth=">1.2 TB/s per stack (9.2 Gb/s/pin x 1024); up to 1.28 TB/s (Samsung 12H)", reported_energy="Micron: 2.5x performance/W vs HBM2E (no pJ/bit)", voltage="", measured_or_simulated="published product", commercial_or_research="commercial",
         notes="PEAK interface bandwidth; sustained is lower (utilization factors applied).", citation="https://micron.net/content/dam/micron/global/public/documents/products/product-flyer/hbm3e-product-brief.pdf ; https://news.samsung.com/global/samsung-develops-industry-first-36gb-hbm3e-12h-dram"),
    dict(source_id="S13", technology="LPDDR5X", publication="JEDEC JESD209-5B (2021); Micron LPDDR5X white paper (power)", year=2021,
         process_node="DRAM", capacity="per package up to 16 GB (not used)", bank_count="", reported_area="", reported_density="",
         reported_bandwidth="8533 Mb/s per pin -> 68.3 GB/s per 64-bit, 136.5 GB/s per 128-bit (KNOWN arithmetic)", reported_energy="Micron: ~110 mW per GB active at 8533 MT/s (vendor marketing figure; unit basis ambiguous)", voltage="VDDQ 0.5 V", measured_or_simulated="standard + vendor", commercial_or_research="commercial",
         notes="PEAK bandwidth. Energy per bit is NOT derivable cleanly from the vendor figure; see ASSUMPTION A3 range.", citation="https://www.businesswire.com/news/home/20210728005691/en ; https://s25.q4cdn.com/621799436/files/doc_downloads/white_papers/lpddr-gen-purp-ai-wkloads-in-lg-scale-dc-deploy-white-paper-4.pdf"),
    dict(source_id="S14", technology="DRAM interface energy (HBM2, GDDR5)", publication="O'Connor et al., 'Fine-Grained DRAM', MICRO-50", year=2017,
         process_node="DRAM (modeled)", capacity="", bank_count="", reported_area="", reported_density="",
         reported_bandwidth="", reported_energy="HBM2 3.9 pJ/bit; GDDR5 14.0 pJ/bit (modeled energy per bit accessed)", voltage="", measured_or_simulated="modeled (paper's energy model)", commercial_or_research="research",
         notes="Modeled, not measured on silicon; the standard reference for stacked-DRAM pJ/bit.", citation="https://research.nvidia.com/sites/default/files/pubs/2017-10_Fine-Grained-DRAM%3A-Energy-Efficient/oconnor_and_chatterjee.micro2017.pdf"),
    dict(source_id="S15", technology="Energy per operation and per memory access, 45 nm 0.9 V", publication="Horowitz, 'Computing's energy problem (and what we can do about it)', ISSCC 2014 1.1", year=2014,
         process_node="45 nm", capacity="", bank_count="", reported_area="", reported_density="",
         reported_bandwidth="", reported_energy="quoted text: DRAM access '1 to 2nJ'; DRAM I/O 'over 20pJ/bit'; with improved I/O 'still large (10pJ/bit, 0.6nJ/8B)'; 'internal cache access or functional operation (10pJ)'; 'a cache fetch is 20pJ'", voltage="0.9 V", measured_or_simulated="rough published estimates (Fig. 1.1.9)", commercial_or_research="research/tutorial",
         notes="Old node; used only as an order-of-magnitude anchor for SRAM read energy (10-20 pJ per 64-bit word = 0.16-0.31 pJ/bit small arrays). The widely reproduced 1 MB cache = 100 pJ/64b (1.6 pJ/bit) figure from Fig. 1.1.9 could not be re-read from the PDF here and is marked UNVERIFIED-HERE.", citation="https://pages.cs.wisc.edu/~markhill/restricted/isscc2014_horowitz_power_scaling.pdf"),
    dict(source_id="S16", technology="Hybrid-bonded die-to-die I/O energy", publication="Synopsys 3DIO PHY article; AMD Hot Chips 2021 V-Cache claims", year=2024,
         process_node="advanced", capacity="", bank_count="", reported_area="", reported_density="",
         reported_bandwidth="4-6 Gb/s per lane (Synopsys)", reported_energy="<0.05 pJ/bit (Synopsys claim); AMD: <1/3 of micro-bump energy per bit (relative only)", voltage="", measured_or_simulated="vendor claims", commercial_or_research="commercial IP / product",
         notes="VENDOR-CLAIM; used as the lower end of a 0.05-0.3 pJ/bit range (ASSUMPTION A4).", citation="https://www.synopsys.com/articles/addressing-ai-advanced-packaging-3dio-phy.html ; https://tomshardware.com/news/amd-unveils-more-ryzen-3d-packaging-and-v-cache-details-at-hot-chips"),
    dict(source_id="S17", technology="Hybrid bonding density / bandwidth density; TSMC SoIC roadmap", publication="Wevolver summary of published demonstrations; AnandTech on TSMC SoIC pitch", year=2024,
         process_node="advanced", capacity="", bank_count="", reported_area="", reported_density="sub-10 um pitch; SoIC 9 um -> 3 um by 2027 (roadmap)",
         reported_bandwidth="3.4 TB/s/mm2 bandwidth density demonstrated (research)", reported_energy="", voltage="", measured_or_simulated="published demonstrations / roadmap", commercial_or_research="commercial (SoIC) / research (record density)",
         notes="Shows a stacked-die crossing is not the bandwidth limiter; roadmap figures are TARGETS.", citation="https://www.wevolver.com/article/hybrid-bonding-enabling-high-density-3d-integration-for-next-generation-electronics ; https://at-web1.www.anandtech.com/show/21414/tsmcs-3d-stacked-soic-packaging-making-quick-progress-3um-pitch-in-2027"),
    dict(source_id="S18", technology="HBM-PIM (Samsung Aquabolt-XL)", publication="ISSCC 2021; Hot Chips 33", year=2021,
         process_node="HBM2 DRAM (20 nm class)", capacity="HBM2 stack with PIM dies (4 of 8 dies)", bank_count="32 PCUs per 4 Gb die (one per bank pair)", reported_area="", reported_density="",
         reported_bandwidth="", reported_energy="vendor: >2x system performance, >70% energy reduction on tested kernels (no pJ/bit)", voltage="", measured_or_simulated="measured prototype (vendor-reported)", commercial_or_research="research prototype (sampled, not volume)",
         notes="Only public example of compute inside an HBM stack; FP16 300 MHz PCUs, not a ternary engine.", citation="https://www.businesswire.com/news/home/20210216006156/en ; https://www.eenewseurope.com/news/samsung-HBM-PIM-ai-chip"),
    dict(source_id="S19", technology="GDDR6-AiM (SK hynix accelerator-in-memory)", publication="ISSCC 2022", year=2022,
         process_node="GDDR6 DRAM", capacity="", bank_count="", reported_area="", reported_density="",
         reported_bandwidth="", reported_energy="vendor: up to 80% power reduction, 16x speedup on selected kernels (relative)", voltage="", measured_or_simulated="measured prototype (vendor-reported)", commercial_or_research="research prototype",
         notes="Relative claims only.", citation="https://news.skhynix.co.kr/developed-processing-in-memory ; https://blocksandfiles.com/2022/02/16/sk-hynix-builds-dram-processor-in-memory-chip/"),
    dict(source_id="S20", technology="DRAM PIM DIMM (UPMEM)", publication="Hot Chips 31 (2019); AnandTech analysis", year=2019,
         process_node="2x nm DRAM", capacity="8 GB + 128 DPUs per DIMM; 64 MB DRAM bank per DPU", bank_count="128 per DIMM", reported_area="", reported_density="",
         reported_bandwidth="~1 GB/s per DPU to its 64 MB bank (2 TB/s aggregate over 2,048 DPUs)", reported_energy="", voltage="", measured_or_simulated="published product", commercial_or_research="commercial",
         notes="Shows the per-bank bandwidth a DRAM-process compute unit actually gets (~1 GB/s per 64 MB).", citation="https://at-web1.www.anandtech.com/show/14750/hot-chips-31-analysis-inmemory-processing-by-upmem/2 ; https://www.nextplatform.com/2019/10/03/accelerating-compute-by-cramming-it-into-dram/"),
    dict(source_id="S21", technology="EUV single-exposure field (reticle) limit", publication="ASML / SPIE Advanced Lithography; industry standard field 26 x 33 mm", year=2015,
         process_node="any EUV node", capacity="", bank_count="", reported_area="858 mm2 (26 x 33 mm)", reported_density="",
         reported_bandwidth="", reported_energy="", voltage="", measured_or_simulated="standard", commercial_or_research="commercial",
         notes="Maximum monolithic die without field stitching; high-NA halves it to 26 x 16.5 mm.", citation="https://bits-chips.com/article/asml-throws-weight-behind-bigger-euv-masks/ ; https://euvlitho.com/Blogs/2015%20SPIE%20Advanced%20Lithography%20EUVL%20Conference%20Summary%20and%20Analysis%20030915.pdf"),
    dict(source_id="S22", technology="eDRAM L4 (IBM z15 system controller)", publication="Tom's Hardware report on z15", year=2019,
         process_node="GF 14 nm", capacity="960 MiB L4 eDRAM", bank_count="", reported_area="not used (SC chip ~696 mm2 reported elsewhere, unverified here)", reported_density="",
         reported_bandwidth="", reported_energy="", voltage="", measured_or_simulated="published product", commercial_or_research="commercial",
         notes="Shows ~1 GB-class eDRAM on one large die existed at 14 nm; density not used because the die area was not verified here.", citation="https://www.tomshardware.com/uk/news/ibm-z15-almost-doubles-cache-density-features-960mib-l4-edram"),
]

# Explicit assumptions used by the models (no external evidence, or a range chosen from several sources).
ASSUMPTIONS: list[dict] = [
    dict(source_id="A1", quantity="Chip-level SRAM density tiers (MB per mm2 of die, decimal MB)",
         value="raw_bitcell_n5_n3e=5.95 (S01, S02; NOT usable for chips); macro_n3e=3.8-4.2 (S02); cache_only_die_n7=1.64-1.86 (S03, S04); sram_heavy_die_n7=0.68-1.09 (S07, S05); sram_heavy_die_n5_waferscale=0.95 (S06); edram_die_22nm=1.74 (S10)",
         label="INFERRED from the cited parts; a FlexTern die is assumed to land in the 'sram_heavy_die' tier because it carries Y-Core banks (EXP-0005) beside the SRAM"),
    dict(source_id="A2", quantity="Usable-capacity fraction of physical memory (ECC, repair, alignment, tail waste, metadata)",
         value="margins 5%, 10%, 20%, 30% swept; 10% nominal", label="ASSUMPTION"),
    dict(source_id="A3", quantity="Energy per bit ranges (pJ/bit) for byte-movement estimates",
         value="sram_local_read=0.16-1.6 (S15, 45 nm, small-to-large arrays; modern-node SRAM UNMEASURED here); hbm_class=3.9-6 (S14 HBM2 modeled 3.9; HBM3E assumed no worse than 1.5x); lpddr5x=5-15 (S13 vendor figure of ambiguous unit basis bracketed by S15's 10 pJ/bit improved-I/O DRAM); ddr_class_io=20 (S15); hybrid_bond_crossing=0.05-0.3 (S16 vendor claim as lower end; upper end ASSUMPTION); on_die_fabric_per_mm=not modeled",
         label="INFERRED ranges; never summed into a whole-chip J/token"),
    dict(source_id="A4", quantity="Bandwidth utilization factors applied to peak interface bandwidth",
         value="50%, 70%, 80% swept; 70% nominal", label="ASSUMPTION"),
    dict(source_id="A5", quantity="Hypothetical larger ternary models (memory-capacity scaling only, no quality claim)",
         value="ternary fraction of parameters 95% (Falcon-E 3B measured 95.6%, 1B 92.2%, BitNet 86.4%); hidden sizes 4096/5120/6656/8192 for 7B/13B/30B/70B; vocab 32768; head BF16 or EXP-0004 candidate 4.156 bits; norms/scales <0.1% ignored beyond the margin",
         label="HYPOTHESIZED scaling; shapes are Llama-class conventions, not measured checkpoints"),
    dict(source_id="A6", quantity="Technology availability",
         value="eDRAM at leading-edge logic nodes is not currently offered by the leading foundries (no public N5/N3 eDRAM product found); multi-die SRAM stacks deeper than one V-Cache die are not a shipping product; compute inside an HBM stack exists only as a research prototype (S18)",
         label="ASSUMPTION from absence of public evidence; may be wrong"),
    dict(source_id="A7", quantity="Bank-local compute logic area",
         value="not estimated: EXP-0005 cell counts are generic (technology-independent) and cannot be converted to mm2 without a cell library",
         label="UNMEASURED"),
    dict(source_id="A9", quantity="Practical ceiling of multi-die SRAM organizations (chiplets or stacks) before DRAM-class capacity is required",
         value="2-4 GB (d-Matrix Corsair ships 2 GB of SRAM per card across 8 chiplets, S09; 4 x Graphcore-class dies as the upper bound)", label="ASSUMPTION anchored on S09/S05"),
    dict(source_id="A10", quantity="Preload amortization for the full-residency policy row",
         value="1,000 tokens served per preload (EXP-0005 used steady state, i.e. infinite); the steady-state figure is also reported", label="ASSUMPTION"),
    dict(source_id="A8", quantity="Collector and broadcast constraints",
         value="one result group (4 outputs) per cycle per collector; bank-local activation buffer K x 8 bits; delivered activation bytes = NB x injected (EXP-0005 traffic_model.per_token); cycles per tensor = max(rows_per_bank x K / b, rowq) (EXP-0005 workload_mapping.nearmem_cycles)",
         label="MEASURED in EXP-0005 RTL (bound), ANALYTICAL here"),
]

FIELDS = ["source_id", "technology", "publication", "year", "process_node", "capacity", "bank_count", "reported_area", "reported_density",
          "reported_bandwidth", "reported_energy", "voltage", "measured_or_simulated", "commercial_or_research", "notes", "citation"]


def write(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sources.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(SOURCES)
    with open(os.path.join(out_dir, "assumptions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "quantity", "value", "label"]); w.writeheader(); w.writerows(ASSUMPTIONS)
    L = ["# EXP-0006 source notes", "", "Source quality: six rows are primary documents (S03 AMD ISSCC 2022 slides; S09 d-Matrix Hot Chips 2025 PDF; S12 Micron HBM3E product brief and Samsung newsroom; S13 JEDEC release and Micron white paper; S14 the MICRO-50 paper; S15 the ISSCC 2014 paper). The remaining rows are technical-press or WikiChip/SemiWiki summaries of primary disclosures (S01, S02, S04-S08, S10, S11, S16-S22) and are adequate for order-of-magnitude use only.", "", "Every physical constant in `analysis/memory_residency/` traces to a row of `sources.csv` (external) or `assumptions.csv` (explicit). Tiers are kept apart: raw bit-cell density (S01/S02) is never used as a chip density; macro density (S02) excludes TSVs, routing and compute; cache-only die density (S03/S04) includes TSVs and periphery but no compute; whole SRAM-heavy die density (S05/S06/S07) is the only tier that includes compute and is the tier a FlexTern die is assumed to land in (A1). Vendor bandwidths are peak; utilization factors (A4) are applied. Energy figures are ranges (A3), the SRAM side rests on a 45 nm tutorial (S15) and is labeled accordingly; nothing is summed into a chip J/token.", ""]
    for s in SOURCES:
        L.append(f"- **{s['source_id']}** {s['technology']} — {s['publication']} ({s['year']}, {s['process_node']}; {s['commercial_or_research']}; {s['measured_or_simulated']}). "
                 + "; ".join(f"{k}: {s[k]}" for k in ("capacity", "reported_area", "reported_density", "reported_bandwidth", "reported_energy") if s[k]) + f". {s['notes']} <{s['citation']}>")
    L.append("")
    for a in ASSUMPTIONS:
        L.append(f"- **{a['source_id']}** ({a['label']}) {a['quantity']}: {a['value']}")
    with open(os.path.join(out_dir, "source_notes.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ----------------------------------------------------------------------------- constants the models import
RETICLE_MM2 = 858.0                                   # S21
DENSITY_TIERS_MB_PER_MM2 = {                           # A1 (decimal MB)
    "raw_bitcell_n5_n3e (NOT a chip density)": (5.95, 5.95),
    "macro_n3e (array + local periphery)": (3.8, 4.2),
    "cache_only_die_n7 (V-Cache, incl. TSVs, no compute)": (1.64, 1.86),
    "sram_heavy_die_n7 (Dojo..Graphcore, incl. compute)": (0.68, 1.09),
    "sram_heavy_waferscale_n5 (WSE-3, incl. compute)": (0.95, 0.95),
    "edram_die_22nm_2014 (Crystalwell)": (1.74, 1.74),
}
CHIP_TIER = "sram_heavy_die_n7 (Dojo..Graphcore, incl. compute)"
CACHE_DIE_TIER = "cache_only_die_n7 (V-Cache, incl. TSVs, no compute)"
VCACHE_DIE_MB, VCACHE_DIE_MM2, VCACHE_DIE_BW_TBPS = 67.1, 41.0, 2.0     # S03
GRAPHCORE_DIE_MB, GRAPHCORE_DIE_MM2 = 900.0, 823.0                       # S05
MULTI_DIE_SRAM_CEILING_BYTES = 3.6e9                                     # A9 (4 x Graphcore-class dies; d-Matrix ships 2 GB)
PRELOAD_AMORTIZATION_TOKENS = 1000                                       # A10
HBM3E_STACK_GB, HBM3E_STACK_PEAK_GBPS = 36.0, 1200.0                     # S12
LPDDR5X_GBPS_PER_64BIT = 8533e6 * 64 / 8 / 1e9                           # S13 = 68.26 GB/s
UPMEM_GBPS_PER_DPU, UPMEM_MB_PER_DPU = 1.0, 67.1                          # S20 (64 MiB)
ENERGY_PJ_PER_BIT = {                                                    # A3
    "sram_local_read_45nm": (0.16, 1.6),
    "hbm_class": (3.9, 6.0),
    "lpddr5x_class": (5.0, 15.0),
    "ddr_class_io": (20.0, 20.0),
    "hybrid_bond_crossing": (0.05, 0.3),
}
MARGINS = (0.05, 0.10, 0.20, 0.30)                                       # A2
NOMINAL_MARGIN = 0.10                                                    # A2
UTILIZATIONS = (0.5, 0.7, 0.8)                                           # A4
NOMINAL_UTIL = 0.7                                                       # A4
TARGET_TOKS = (10, 25, 50, 100, 250)                                     # TARGET rates from the EXP-0006 prompt (A4 applies)
HYPOTHETICAL_MODELS = [                                                  # A5
    dict(name="hyp-7B", params=7.0e9, hidden=4096, layers=32, ternary_fraction=0.95),
    dict(name="hyp-13B", params=13.0e9, hidden=5120, layers=40, ternary_fraction=0.95),
    dict(name="hyp-30B", params=30.0e9, hidden=6656, layers=60, ternary_fraction=0.95),
    dict(name="hyp-70B", params=70.0e9, hidden=8192, layers=80, ternary_fraction=0.95),
]
HEAD_CANDIDATE_BITS = 4.156   # EXP-0004 measured (results/model_precision/head_int4.json)
KV_CANDIDATE_BITS = 4.5       # EXP-0004 measured
