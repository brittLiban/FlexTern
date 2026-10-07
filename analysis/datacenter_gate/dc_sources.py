"""
EXP-0007 sources and assumptions: every external constant in the datacenter model, with provenance.
Peak product figures are PEAK (ceilings/parameters), never achieved performance.  Decimal units.
Labels: MEASURED-PUBLISHED (shipping part's published spec), MEASURED-BENCHMARK (MLPerf result with
its context), MODELED-PUBLISHED, VENDOR-CLAIM, SECONDARY (tech press summary), ASSUMPTION, TARGET.
"""

from __future__ import annotations

import csv
import os

FIELDS = ["source_id", "system_or_quantity", "publication", "year", "precision", "capacity", "bandwidth", "interconnect",
          "power", "measured_or_peak", "software_or_context", "label", "notes", "citation"]

SOURCES: list[dict] = [
    dict(source_id="D01", system_or_quantity="NVIDIA B200 (Blackwell) GPU", publication="NVIDIA DGX B200 datasheet; B200 SXM spec sheet", year=2024,
         precision="FP4 (NVFP4) 9 PFLOPS dense / 18 PFLOPS sparse tensor; FP8 4.5 PFLOPS dense", capacity="180 GB HBM3e (B200 SXM; 192 GB in GB200 Superchip configurations)",
         bandwidth="8 TB/s HBM peak", interconnect="NVLink 5: 1.8 TB/s bidirectional per GPU (900 GB/s per direction)", power="1,000 W TDP",
         measured_or_peak="peak", software_or_context="datasheet", label="MEASURED-PUBLISHED spec (peak)",
         notes="Used as System A ceilings: HBM 8 TB/s, FP4 9 PFLOPS dense, NVLink 0.9 TB/s per direction, 180 GB.", citation="https://openzeka.com/wp-content/uploads/2024/11/dgx-scale-ai-infrastructure-dgx-b200-datasheet-nvidia-web-1-1.pdf ; https://flopper.io/gpu/nvidia-b200-sxm-180gb/spec-sheet.pdf"),
    dict(source_id="D02", system_or_quantity="AMD Instinct MI355X", publication="AMD Advancing AI 2025 launch; Tom's Hardware spec report", year=2025,
         precision="FP4/FP6 ~10 PFLOPS dense (20.1 PFLOPS with sparsity)", capacity="288 GB HBM3E", bandwidth="8 TB/s HBM peak",
         interconnect="Infinity Fabric 538 GB/s per chip, 8-chip full mesh", power="1,400 W TBP", measured_or_peak="peak", software_or_context="launch materials",
         label="MEASURED-PUBLISHED spec (peak); partly SECONDARY", notes="Cross-check that System A parameters are current-generation, not vendor-specific.", citation="https://www.tomshardware.com/tech-industry/artificial-intelligence/amd-reveals-core-specs-for-instinct-mi355x-cdna4-ai-accelerator-slated-for-shipping-in-the-second-half-of-2025"),
    dict(source_id="D03", system_or_quantity="TensorRT-LLM KV-cache quantization support", publication="NVIDIA TensorRT-LLM documentation, Quantization", year=2025,
         precision="FP8 KV cache; NVFP4 KV cache (with FP8 weights/activations); W4A16/W4A8 weights", capacity="", bandwidth="", interconnect="", power="",
         measured_or_peak="feature", software_or_context="production inference framework on Hopper/Blackwell", label="MEASURED-PUBLISHED (documented feature)",
         notes="The conventional baseline is granted native FP8 KV and, as a sensitivity, NVFP4 KV. No ternary monopoly on compressed KV.", citation="https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html"),
    dict(source_id="D04", system_or_quantity="MLPerf Inference v5.0/v5.1 Llama 2 70B on 8x B200", publication="NVIDIA developer blog (v5.0); Lambda blog (v5.1)", year=2025,
         precision="FP4 weights (NVIDIA submission)", capacity="", bandwidth="", interconnect="NVLink 8-GPU", power="not reported here",
         measured_or_peak="MEASURED benchmark", software_or_context="Llama 2 70B, OpenOrca prompts (~1k input, ~300 output tokens average), offline 102,725 tok/s (v5.1) and server 98,443 tok/s (v5.0), 8 GPUs, 99.9% accuracy target",
         label="MEASURED-BENCHMARK", notes="Used ONLY as a sanity anchor (within 2x) for the System A estimate at 70B, large batch, short context. Not used to derive batch, KV precision or utilization.",
         citation="https://developer.nvidia.com/blog/nvidia-blackwell-delivers-massive-performance-leaps-in-mlperf-inference-v5-0 ; https://lambda.ai/blog/lambda-mlperf-inference-v5.1"),
    dict(source_id="D05", system_or_quantity="Cerebras Inference, Llama 3.1 70B", publication="Cerebras press release Oct 2024; secondary blog for power", year=2024,
         precision="16-bit weights (Meta original)", capacity="44 GB SRAM per WSE-3 (EXP-0006 S06); 140 GB of weights implies >= 4 wafers (INFERRED)", bandwidth="21 PB/s on-wafer",
         interconnect="wafer-to-wafer not disclosed", power="~23 kW per system (SECONDARY blog figure)", measured_or_peak="vendor-measured",
         software_or_context="2,100 tokens/s per user at batch 1", label="VENDOR-CLAIM; power SECONDARY", notes="Architectural precedent: SRAM-resident 70B at 16 bits costs multiple wafers; ternary would cut resident bytes 8x.",
         citation="https://www.cerebras.ai/press-release/cerebras-triples-its-industry-leading-inference-performance-setting-new-all-time-record ; https://www.spheron.network/blog/cerebras-vs-nvidia-h100-inference-2026/"),
    dict(source_id="D06", system_or_quantity="Groq LPU, Llama 2 70B deployment", publication="The Next Platform (Nov 2023)", year=2023,
         precision="INT8", capacity="230 MB SRAM per LPU (EXP-0006 S08); 576 LPUs across 8 racks for 70B", bandwidth="80 TB/s per chip on-die",
         interconnect="switched pods across 8 racks", power="vendor claim 1-3 J/token (not verified)", measured_or_peak="peak on-die bandwidth; throughput vendor-reported", software_or_context="512-in / 1,024-out prompts, >300 tokens/s per user",
         label="SECONDARY + VENDOR-CLAIM", notes="Architectural precedent: SRAM-resident 70B at INT8 = 576 chips / 8 racks. Energy claim not used numerically.",
         citation="https://www.nextplatform.com/2023/11/27/groq-says-it-can-deploy-1-million-ai-inference-chips-in-two-years/"),
    dict(source_id="D07", system_or_quantity="NVLink-C2C energy per bit", publication="NVIDIA developer blog / Hot Chips 34 Grace presentation", year=2022,
         precision="", capacity="", bandwidth="900 GB/s bidirectional on package", interconnect="die-to-die on package", power="1.3 pJ/bit",
         measured_or_peak="vendor-stated", software_or_context="", label="VENDOR-CLAIM", notes="Package-local tier energy.", citation="https://developer.nvidia.com/blog/nvidia-grace-cpu-superchip-architecture-in-depth"),
    dict(source_id="D08", system_or_quantity="GPU-to-GPU NVLink SerDes and long-reach SerDes energy", publication="Doubleword blog (NVLink ~4 pJ/bit); ISSCC-class SerDes papers (XSR 1.55 pJ/bit; 112G-LR 4.5-6 pJ/bit)", year=2024,
         precision="", capacity="", bandwidth="", interconnect="board-local (NVLink), rack-local SerDes", power="~4 pJ/bit NVLink (SECONDARY); 1.55-6 pJ/bit SerDes classes",
         measured_or_peak="mixed", software_or_context="", label="SECONDARY / published SerDes figures", notes="Board tier 3-6 pJ/bit range (A4).", citation="https://blog.doubleword.ai/nvlink-scale-up ; https://events.vtools.ieee.org/m/257895"),
    dict(source_id="D09", system_or_quantity="Optical interconnect energy", publication="Synopsys article on direct-drive electro-optics; industry CPO summaries", year=2024,
         precision="", capacity="", bandwidth="800G class", interconnect="rack/scale-out optics", power="pluggable ~8 pJ/bit; onboard ~6; CPO target ~3 pJ/bit",
         measured_or_peak="vendor/industry figures", software_or_context="", label="SECONDARY", notes="Rack tier 3-8 pJ/bit range (A4).", citation="https://www.synopsys.com/articles/direct-drive-electro-optical.html"),
    dict(source_id="D10", system_or_quantity="HBM energy per bit", publication="O'Connor et al. MICRO-50 (HBM2 3.9 pJ/bit, modeled); arXiv RPU paper (HBM3e 3.44 pJ/bit, modeled, 2026); EXP-0006 A3", year=2017,
         precision="", capacity="", bandwidth="", interconnect="", power="3.4-6 pJ/bit DRAM-stack class", measured_or_peak="modeled",
         software_or_context="", label="MODELED-PUBLISHED", notes="HBM class range 3.9-6 kept from EXP-0006; 3.44 cited as modeled lower end only.", citation="https://research.nvidia.com/sites/default/files/pubs/2017-10_Fine-Grained-DRAM%3A-Energy-Efficient/oconnor_and_chatterjee.micro2017.pdf ; https://arxiv.org/pdf/2602.18568"),
    dict(source_id="D11", system_or_quantity="Llama 3.1 70B / 405B and Llama 2 7B architecture shapes", publication="Hugging Face config.json files (Meta Llama 3.1 70B; Llama 3.1 405B); Llama 2 7B", year=2024,
         precision="", capacity="70B: 80 layers, hidden 8192, intermediate 28672, 64 heads / 8 KV heads, head_dim 128, vocab 128256; 405B: 126 layers, 16384, 53248, 128 / 8, vocab 128256; 7B: 32 layers, 4096, 11008, 32 / 32, vocab 32000",
         bandwidth="", interconnect="", power="", measured_or_peak="published config", software_or_context="", label="MEASURED-PUBLISHED (architecture); ternary versions HYPOTHESIZED",
         notes="Shapes only. No native ternary model of these sizes exists; quality is not claimed.", citation="https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8/blob/9f5dfe39a764599bee5c95e965c755e982b704a7/config.json ; https://apxml.com/models/llama-3-1-405b"),
    dict(source_id="D12", system_or_quantity="Continuous batching / PagedAttention (vLLM)", publication="Kwon et al., SOSP 2023", year=2023,
         precision="", capacity="", bandwidth="", interconnect="", power="", measured_or_peak="measured in paper", software_or_context="2-4x throughput over prior serving systems; near-zero KV fragmentation waste",
         label="MEASURED-PUBLISHED (paper)", notes="Justifies granting the baseline continuous batching; serving utilization factors remain assumptions (A2).", citation="https://arxiv.org/abs/2309.06180"),
    dict(source_id="D13", system_or_quantity="NVFP4 effective bits", publication="NVIDIA Blackwell NVFP4 format description (E2M1 elements, E4M3 scale per 16-element block)", year=2025,
         precision="4 + 8/16 = 4.5 bits per weight effective", capacity="", bandwidth="", interconnect="", power="", measured_or_peak="format definition", software_or_context="",
         label="KNOWN (format arithmetic)", notes="System A and D are charged 4.5 bits per weight; ternary charged 2.0 (per-tensor BF16 scale negligible).", citation="https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html"),
    dict(source_id="D14", system_or_quantity="EXP-0006 memory sources reused", publication="results/memory_residency/sources.csv (S03 V-Cache 2 TB/s per 64 MB die; S05 Graphcore 47.5 TB/s per 900 MB; S09 d-Matrix 2 GB SRAM per card; S12 HBM3E 36 GB / 1.2 TB/s per stack; S13 LPDDR5X 68.3 GB/s per 64-bit; S15/S16 SRAM and hybrid-bond energy)", year=2026,
         precision="", capacity="", bandwidth="", interconnect="", power="", measured_or_peak="as labeled there", software_or_context="", label="see EXP-0006", notes="Tile capacity, local bandwidth and SRAM energy anchors.", citation="results/memory_residency/sources.csv"),
]

ASSUMPTIONS: list[dict] = [
    dict(source_id="A1", quantity="System A node", value="8 GPUs per NVLink domain (tensor-parallel), 180 GB HBM each (D01), 8 TB/s peak each, FP4 9 PFLOPS dense each, NVLink 0.9 TB/s per direction each; capacity beyond one node adds nodes with pipeline parallelism (reuse = microbatch)", label="ASSUMPTION anchored on D01"),
    dict(source_id="A2", quantity="Utilization factors", value="HBM 70% of peak; FP4 tensor compute 50% of dense peak; links 70%; serving utilization (effective batch / nominal batch) swept 100% (ideal), 85%, 70%, 50%", label="ASSUMPTION (D12 justifies continuous batching exists; the factors are not measured)"),
    dict(source_id="A3", quantity="Effective bits", value="weights: BF16 16, INT8 8, FP4/NVFP4 4.5 (D13), ternary 2.0, ternary dense TARGET 1.6; KV: BF16 16, FP8 8, candidate 4.5 (EXP-0004; also NVFP4 KV for the baseline, D03); activations crossing tiles/GPUs 1 byte (FP8/INT8)", label="KNOWN arithmetic / EXP-0004 / TARGET"),
    dict(source_id="A4", quantity="Energy per bit tiers (pJ/bit)", value="HBM/DRAM-stack 3.9-6 (D10, EXP-0006 A3); LPDDR5X 5-15 (EXP-0006 A3); on-die SRAM local read 1.6 planning (EXP-0006, 45 nm anchor; range 0.16-1.6); chip-local crossing 0.05-0.3; package-local 1.3 (D07); board-local 3-6 (D08); rack-local 3-8 (D09)", label="INFERRED ranges; compute energy NOT modeled"),
    dict(source_id="A5", quantity="Resident tile", value="SRAM capacity swept 256 MB / 512 MB / 1 GB / 2 GB with 10% margin (EXP-0006 A2); local read bandwidth 8 TB/s per tile (ASSUMPTION; Graphcore 47.5 TB/s per 900 MB die and V-Cache 2 TB/s per 64 MB bracket it); per-tile DRAM for KV: one HBM3E stack class (36 GB, 1.2 TB/s peak) when KV does not fit the SRAM; tile-to-tile link 0.9 TB/s per direction (NVLink-C2C class, D07)", label="ASSUMPTION anchored on EXP-0006 sources"),
    dict(source_id="A6", quantity="Resident compute rate", value="TARGET: 4,096 ternary terms per cycle per tile at 1 GHz = 4.1e12 terms/s per tile (~16 EXP-0005 256-term engines, ~400k generic cells); INT4 resident uses the same count of multiply terms at the same rate (no penalty applied, which favors D)", label="TARGET, never achieved"),
    dict(source_id="A7", quantity="Topology placement", value="layers assigned whole to tiles when a layer fits; otherwise a layer is split across k = ceil(layer_bytes / usable) tiles (tensor-parallel, all-reduce traffic counted); head on its own tiles; pipeline stages = tiles / k; microbatch b = max(1, floor(B / stages)); tiers by stage index: 4 tiles per package, 16 per board, 128 per rack", label="ASSUMPTION"),
    dict(source_id="A8", quantity="Hypothetical ternary models", value="shapes from D11 (7B Llama-2-class, 70B and 405B Llama-3.1-class); all linear projections ternary; head untied BF16 or EXP-0004 candidate; vocab 32k or 128k swept for 70B/400B; no quality claim", label="HYPOTHESIZED"),
    dict(source_id="A9", quantity="Prefill", value="prompt = context length C; weights read once per batch of B prompts; dense compute 2 x params x tokens; attention 4 x L x H x C^2 / 2 per sequence; KV written for C tokens; head for the last token only", label="ANALYTICAL"),
]


def write(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sources.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(SOURCES)
    with open(os.path.join(out_dir, "assumptions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["source_id", "quantity", "value", "label"]); w.writeheader(); w.writerows(ASSUMPTIONS)


# ----------------------------------------------------------------------------- constants (every number tagged)
GPU_HBM_BYTES = 180e9                 # D01
GPU_HBM_PEAK_BPS = 8e12               # D01
GPU_FP4_DENSE_FLOPS = 9e15            # D01
GPU_NVLINK_BPS_PER_DIR = 0.9e12       # D01
GPUS_PER_NODE = 8                     # A1
UTIL_HBM, UTIL_COMPUTE, UTIL_LINK = 0.7, 0.5, 0.7     # A2
SERVING_UTILS = (1.0, 0.85, 0.7, 0.5)                   # A2
WEIGHT_BITS = {"bf16": 16.0, "int8": 8.0, "fp4": 4.5, "ternary": 2.0, "ternary_dense_target": 1.6}   # A3 / D13
KV_BITS = {"bf16": 16.0, "fp8": 8.0, "candidate_4p5": 4.5}                                           # A3 / D03 / EXP-0004
ACT_BYTES = 1                         # A3
PJ_PER_BIT = {"hbm": (3.9, 6.0), "lpddr5x": (5.0, 15.0), "sram_local": (0.16, 1.6), "chip": (0.05, 0.3),   # A4
              "package": (1.3, 1.3), "board": (3.0, 6.0), "rack": (3.0, 8.0)}                                # A4 (D07, D08, D09, D10)
SRAM_PLANNING_PJ = 1.6                # A4 (EXP-0006 planning figure)
TILE_CAPS = (256e6, 512e6, 1e9, 2e9)  # A5
TILE_MARGIN = 0.10                    # A5
TILE_LOCAL_BPS = 8e12                 # A5
TILE_DRAM_BYTES, TILE_DRAM_PEAK_BPS = 36e9, 1.2e12   # A5 (HBM3E stack class, D14/S12)
TILE_LINK_BPS = 0.9e12                # A5 (D07 class)
TILE_TERMS_PER_S = 4096 * 1e9         # A6 TARGET
TILES_PER_PACKAGE, TILES_PER_BOARD, TILES_PER_RACK = 4, 16, 128   # A7
MLPERF_8xB200_OFFLINE_TOK_S = 102725  # D04 (sanity anchor only)
