"""
Shared helpers for EXP-0004 (low-precision head / KV feasibility on Falcon-E 1B).

- Pinned checkpoint from analysis/model_specs.py (ADR-0004); SHA-256 verified before use.
- TORCHDYNAMO_DISABLE=1 is set because transformers' BitLinear methods are decorated with
  @torch.compile and this machine has no C++ compiler for inductor; eager execution is used.
- Environment capture, seeding, cache construction, and byte accounting for caches.

All numbers produced here are software measurements on this machine; none is a hardware result.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
from model_specs import SPECS  # noqa: E402

SPEC = SPECS["falcon-e-1b"]
RESULTS_DIR = os.path.join("results", "model_precision")
SEED = 20261006

# KV cache quantization settings (transformers QuantizedCache, KIVI-style).
# Backend: hqq (pure PyTorch). The quanto backend registers a CUDA extension that is JIT-built with nvcc on
# first use; this WSL environment has the NVIDIA driver but no CUDA toolkit, so quanto cannot be used here
# (see KI-006). Override with FLEXTERN_KV_BACKEND=quanto on a machine with a toolkit.
KV_BACKEND = os.environ.get("FLEXTERN_KV_BACKEND", "hqq")
KV_INT4 = {"backend": KV_BACKEND, "nbits": 4, "axis_key": 0, "axis_value": 0, "q_group_size": 64, "residual_length": 128}
KV_INT8_FALLBACK = {"backend": "hqq", "nbits": 8, "axis_key": 0, "axis_value": 0, "q_group_size": 64, "residual_length": 128}


def disable_quanto_extensions() -> list[str]:
    """optimum-quanto registers JIT-compiled C++/CUDA extensions whenever CUDA is visible and builds them
    on first use with torch.utils.cpp_extension.load, which needs a CUDA toolkit (nvcc, CUDA_HOME). This
    WSL environment has the driver but no toolkit, so we unregister the extensions and let quanto take its
    pure-PyTorch fallback paths. Set FLEXTERN_QUANTO_EXT=1 to keep the extensions. Returns what was removed."""
    if os.environ.get("FLEXTERN_QUANTO_EXT") == "1":
        return []
    try:
        import optimum.quanto  # noqa: F401  (triggers extension registration)
        from optimum.quanto.library.extensions import extension as _ext
    except Exception:  # noqa: BLE001
        return []
    removed = [k for k in list(_ext._extensions) if k in ("quanto_cuda", "quanto_cpp", "quanto_hip", "quanto_mps")]
    for k in removed:
        _ext._extensions.pop(k, None)
    return removed


QUANTO_EXTENSIONS_REMOVED = disable_quanto_extensions()


def seed_everything(seed: int = SEED) -> None:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def checkpoint_path() -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(SPEC.model_id, SPEC.weights_file, revision=SPEC.revision)


def verify_checkpoint() -> dict:
    p = checkpoint_path()
    size = os.path.getsize(p)
    sha = sha256_file(p)
    ok = size == SPEC.expected_file_size and sha == SPEC.expected_sha256
    if not ok:
        raise RuntimeError(f"checkpoint identity mismatch: size={size} sha={sha}")
    return {"path": p, "size": size, "sha256": sha, "matches_adr_0004": ok, "model_id": SPEC.model_id, "revision": SPEC.revision}


def load_model(device: str = "cuda"):
    """Load the pinned Falcon-E-1B-Base packed checkpoint in BF16 on the given device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SPEC.model_id, revision=SPEC.revision)
    model = AutoModelForCausalLM.from_pretrained(SPEC.model_id, revision=SPEC.revision, dtype=torch.bfloat16, device_map=device)
    model.eval()
    return model, tok


KV_INT4_AXIS1 = {**KV_INT4, "axis_key": 1, "axis_value": 1}   # groups of 64 along head_dim within one token


def cache_unpacked_ternary_weights(model) -> dict:
    """Runtime speed-up ONLY, numerically identical: transformers' BitLinear.forward re-unpacks its packed uint8
    ternary weights to BF16 on every call (unpack_weights(w)). With torch.compile disabled that costs ~3.2 GB of
    temporaries per token across 168 modules. This unpacks each module once, keeps the BF16 copy on the GPU
    (8x the packed bytes), and makes forward use it. The activation quantizer and post-quant scaling are
    untouched, so outputs are identical to the stock path. Reported memory for the ternary layers must then be
    taken from the packed size (396 MB), not from resident GPU memory. Returns byte accounting."""
    import types
    import torch
    from transformers.integrations.bitnet import unpack_weights
    packed = unpacked = 0
    n = 0
    for mod in model.modules():
        if type(mod).__name__ != "BitLinear" or hasattr(mod, "_w_unpacked"):
            continue
        with torch.inference_mode():
            mod._w_unpacked = unpack_weights(mod.weight, dtype=mod.dtype)
        packed += mod.weight.numel() * mod.weight.element_size()
        unpacked += mod._w_unpacked.numel() * mod._w_unpacked.element_size()
        n += 1

        def forward(self, input):
            import torch.nn.functional as F
            input_quant, input_scale = self.activation_quant(input)
            y = F.linear(input_quant.to(self.dtype), self._w_unpacked)
            y = self.post_quant_process(y, self.weight_scale, input_scale)
            if self.bias is not None:
                y += self.bias.view(1, -1).expand_as(y)
            return y
        mod.forward = types.MethodType(forward, mod)
    return {"modules": n, "packed_bytes": packed, "unpacked_resident_bytes": unpacked}


def make_cache(model, kind: str):
    """kind: 'dynamic' (reference BF16); 'int4' (4-bit KIVI-style, transformers default axis 0: groups strided
    across the flattened cache); 'int4_ax1' (4-bit, groups of 64 within a token along head_dim);
    'int8' (8-bit fallback)."""
    from transformers import DynamicCache, QuantizedCache
    if kind == "dynamic":
        return DynamicCache(config=model.config)
    if kind == "int4":
        return QuantizedCache(config=model.config, **KV_INT4)
    if kind == "int4_ax1":
        return QuantizedCache(config=model.config, **KV_INT4_AXIS1)
    if kind == "int8":
        return QuantizedCache(config=model.config, **KV_INT8_FALLBACK)
    raise ValueError(kind)


def kv_settings(kind: str) -> dict:
    return {"dynamic": {"dtype": "model dtype (BF16)"}, "int4": KV_INT4, "int4_ax1": KV_INT4_AXIS1, "int8": KV_INT8_FALLBACK}[kind]


def _tensor_bytes_in(obj, seen: set | None = None) -> int:
    """Sum nbytes of every torch tensor reachable from obj's attributes (one level of containers)."""
    import torch
    seen = seen if seen is not None else set()
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return obj.untyped_storage().nbytes() if obj.numel() else 0
    total = 0
    if isinstance(obj, (list, tuple)):
        for x in obj:
            total += _tensor_bytes_in(x, seen)
        return total
    if isinstance(obj, dict):
        for x in obj.values():
            total += _tensor_bytes_in(x, seen)
        return total
    if isinstance(obj, torch.Size):
        return 0
    d = getattr(obj, "__dict__", None)
    if d:
        for v in d.values():
            total += _tensor_bytes_in(v, seen)
    # quanto QTensor subclasses keep their payload in tensor attributes (_data, _scale, _shift)
    for name in ("_data", "_scale", "_shift", "_zeropoint"):
        v = getattr(obj, name, None)
        if isinstance(v, torch.Tensor):
            total += _tensor_bytes_in(v, seen)
    return total


def cache_bytes(cache) -> dict:
    """Byte accounting for a transformers Cache: residual (full precision) vs quantized payload, per layer summed."""
    import torch
    residual = quantized = 0
    layers = getattr(cache, "layers", [])
    for layer in layers:
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if isinstance(t, torch.Tensor) and t.numel():
                residual += t.numel() * t.element_size()
        for name in ("_quantized_keys", "_quantized_values"):
            q = getattr(layer, name, None)
            if q is not None:
                quantized += _tensor_bytes_in(q)
    return {"residual_bytes": residual, "quantized_bytes": quantized, "total_bytes": residual + quantized}


def quantized_layer_internals(cache, layer_idx: int = 0) -> dict:
    """Describe the stored tensors of one quantized layer: dtypes, shapes, effective bits."""
    import torch
    layer = cache.layers[layer_idx]
    out = {"class": type(layer).__name__, "nbits": getattr(layer, "nbits", None), "q_group_size": getattr(layer, "q_group_size", None),
           "residual_length": getattr(layer, "residual_length", None), "axis_key": getattr(layer, "axis_key", None),
           "cumulative_length": getattr(layer, "cumulative_length", None)}
    for name in ("keys", "values"):
        t = getattr(layer, name, None)
        out[f"residual_{name}"] = {"dtype": str(t.dtype), "shape": list(t.shape)} if isinstance(t, torch.Tensor) else None
    for name in ("_quantized_keys", "_quantized_values"):
        q = getattr(layer, name, None)
        if q is None:
            out[name] = None
            continue
        if isinstance(q, tuple):   # hqq backend: (packed uint8 tensor, meta dict with scale/zero/shape/group_size/packing)
            qt, meta = q
            desc = {"type": "hqq (packed tensor, meta)", "shape": list(meta.get("shape", [])),
                    "_data": {"dtype": str(qt.dtype), "shape": list(qt.shape), "bytes": qt.numel() * qt.element_size()},
                    "packing": meta.get("packing"), "nbits": meta.get("nbits"), "group_size": meta.get("group_size"), "axis": meta.get("axis")}
            for attr in ("scale", "zero"):
                v = meta.get(attr)
                if isinstance(v, torch.Tensor):
                    desc[f"_{attr}"] = {"dtype": str(v.dtype), "shape": list(v.shape), "bytes": v.numel() * v.element_size()}
            payload = desc["_data"]["bytes"] + sum(desc[a]["bytes"] for a in ("_scale", "_zero") if a in desc)
        else:                      # quanto backend: QTensor subclass with _data/_scale/_shift
            desc = {"type": type(q).__name__, "shape": list(q.shape) if hasattr(q, "shape") else None}
            for attr in ("_data", "_scale", "_shift", "_zeropoint"):
                v = getattr(q, attr, None)
                if isinstance(v, torch.Tensor):
                    desc[attr] = {"dtype": str(v.dtype), "shape": list(v.shape), "bytes": v.numel() * v.element_size()}
            payload = sum(desc[a]["bytes"] for a in ("_data", "_scale", "_shift", "_zeropoint") if a in desc)
        elems = 1
        for s in (desc["shape"] or [0]):
            elems *= int(s)
        desc["elements"] = elems
        desc["payload_bytes"] = payload
        desc["effective_bits_per_element"] = 8 * payload / elems if elems else None
        out[name] = desc
    return out


def cuda_mem() -> dict:
    import torch
    if not torch.cuda.is_available():
        return {}
    return {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated()}


def reset_peak() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def sync() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def environment() -> dict:
    import torch, transformers, numpy
    env = {"generated": time.strftime("%Y-%m-%d %H:%M:%S %z"), "python": sys.version.split()[0], "platform": platform.platform(),
           "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
           "transformers": transformers.__version__, "numpy": numpy.__version__, "seed": SEED,
           "TORCHDYNAMO_DISABLE": os.environ.get("TORCHDYNAMO_DISABLE"),
           "model_id": SPEC.model_id, "revision": SPEC.revision, "expected_sha256": SPEC.expected_sha256}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env.update({"gpu": p.name, "gpu_total_mem_bytes": p.total_memory, "gpu_capability": f"{p.major}.{p.minor}"})
    for pkg in ("optimum.quanto", "hqq", "lm_eval", "datasets", "accelerate"):
        try:
            m = __import__(pkg)
            env[pkg] = getattr(m, "__version__", "present")
        except Exception as e:  # noqa: BLE001
            env[pkg] = f"unavailable: {type(e).__name__}"
    return env


def write_json(name: str, obj) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    p = os.path.join(RESULTS_DIR, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return p


def read_json(name: str):
    with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as f:
        return json.load(f)
