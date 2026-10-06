"""
Minimal, dependency-light safetensors reader for FlexTern workload analysis.

Why not the `safetensors` package? Its numpy backend cannot load BF16, and we
want explicit control over which bytes are touched: the 657 MB embedding is
never read, only its header entry. Parsing the header ourselves also lets the
tool prove that every byte in the file is accounted for by exactly one tensor.

Safetensors layout (https://github.com/huggingface/safetensors):
    bytes 0..7   : little-endian u64, N = header length
    bytes 8..8+N : UTF-8 JSON header {name: {dtype, shape, data_offsets}}
    bytes 8+N..  : raw tensor data; data_offsets are relative to this point

Also contains the BitNet 2-bit unpacking routine, transcribed from
transformers/integrations/bitnet.py (unpack_weights / pack_weights):
    stored code = q + 1  (q in {-1,0,+1}  ->  code in {0,1,2}; code 3 unused)
    packed shape [R, K] holds unpacked shape [4R, K]
    unpacked[i*R + r, c] = ((packed[r, c] >> (2*i)) & 3) - 1,  i = 0..3
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

import numpy as np

VALUES_PER_ITEM = 4  # ternary values per uint8, as in transformers


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int  # absolute byte offset in file
    end: int    # absolute byte offset (exclusive)

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


def read_header(path: str) -> tuple[int, dict]:
    """Return (header_len, header_dict) without reading tensor data."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n).decode("utf-8"))
    return n, header


def bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    """BF16 is the top 16 bits of an IEEE float32. Shift left and reinterpret."""
    return (raw_u16.astype(np.uint32) << 16).view(np.float32)


def unpack_ternary(packed: np.ndarray) -> np.ndarray:
    """Unpack BitNet 2-bit codes to int8 ternary values, mirroring transformers.

    packed: uint8 array of shape [R, K] (or [R]).
    returns: int8 array of shape [4R, K] (or [4R]) with values in {-1, 0, +1}.
             Code 3 (never produced by pack_weights) maps to +2 and is NOT
             silently folded; callers count it separately.
    """
    if packed.dtype != np.uint8:
        raise TypeError("packed must be uint8")
    r = packed.shape[0]
    out_shape = (VALUES_PER_ITEM * r,) + tuple(packed.shape[1:])
    out = np.empty(out_shape, dtype=np.int8)
    for i in range(VALUES_PER_ITEM):
        codes = (packed >> (2 * i)) & 3
        out[i * r:(i + 1) * r] = codes.astype(np.int8) - 1
    return out


def pack_ternary(q: np.ndarray) -> np.ndarray:
    """Inverse of unpack_ternary, for round-trip tests. q in {-1,0,+1}, dim0 % 4 == 0."""
    if q.shape[0] % VALUES_PER_ITEM != 0:
        raise ValueError("first dim must be a multiple of 4")
    r = q.shape[0] // VALUES_PER_ITEM
    codes = (q.astype(np.int16) + 1).astype(np.uint8)
    packed = np.zeros((r,) + tuple(q.shape[1:]), dtype=np.uint8)
    for i in range(VALUES_PER_ITEM):
        packed |= codes[i * r:(i + 1) * r] << (2 * i)
    return packed


class SafetensorsFile:
    """Memory-mapped safetensors file with per-tensor byte accounting."""

    def __init__(self, path: str):
        self.path = path
        self.header_len, header = read_header(path)
        self.data_start = 8 + self.header_len
        self.metadata = header.get("__metadata__", {})
        self.tensors: dict[str, TensorInfo] = {}
        for name, v in header.items():
            if name == "__metadata__":
                continue
            o0, o1 = v["data_offsets"]
            self.tensors[name] = TensorInfo(
                name=name,
                dtype=v["dtype"],
                shape=tuple(int(s) for s in v["shape"]),
                start=self.data_start + int(o0),
                end=self.data_start + int(o1),
            )
        self._mm = np.memmap(path, dtype=np.uint8, mode="r")
        self.file_size = int(self._mm.shape[0])

    # ---- accounting ---------------------------------------------------------
    def verify_layout(self) -> dict:
        """Prove the header covers the file: contiguous, non-overlapping, exact."""
        infos = sorted(self.tensors.values(), key=lambda t: t.start)
        pos = self.data_start
        for t in infos:
            if t.start != pos:
                raise AssertionError(f"gap/overlap before {t.name}: expected {pos}, got {t.start}")
            expected = t.numel * _dtype_size(t.dtype)
            if t.nbytes != expected:
                raise AssertionError(f"{t.name}: {t.nbytes} bytes but shape*dtype = {expected}")
            pos = t.end
        if pos != self.file_size:
            raise AssertionError(f"tensors end at {pos}, file is {self.file_size} bytes")
        return {
            "file_size": self.file_size,
            "header_len": self.header_len,
            "data_bytes": self.file_size - self.data_start,
            "tensor_count": len(self.tensors),
        }

    # ---- access -------------------------------------------------------------
    def raw(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        return self._mm[t.start:t.end]

    def uint8(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        if t.dtype != "U8":
            raise TypeError(f"{name} is {t.dtype}, not U8")
        return np.asarray(self.raw(name)).reshape(t.shape)

    def bf16_as_f32(self, name: str) -> np.ndarray:
        t = self.tensors[name]
        if t.dtype != "BF16":
            raise TypeError(f"{name} is {t.dtype}, not BF16")
        u16 = np.frombuffer(self.raw(name).tobytes(), dtype="<u2").reshape(t.shape)
        return bf16_to_f32(u16)


def _dtype_size(dtype: str) -> int:
    return {"U8": 1, "I8": 1, "BOOL": 1, "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
            "F32": 4, "I32": 4, "U32": 4, "F64": 8, "I64": 8, "U64": 8}[dtype]
