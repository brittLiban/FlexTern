"""Dataset audit for FT-0010 Stage A: rights, structure, and layout groups.

Fetches only small metadata (HF tree listing, calibration.json, map.png) for every
ground-truth scene of MTMC_Tracking_2026 (train + val), then groups scenes by
physical layout. Names are not trusted: Cosmos-Transfer re-renderings share cameras,
calibration and ground truth with their source scene (README), and map.png byte
identity / identical camera matrices expose any further sharing.

Usage:  python -m analysis.edge_inference.dataset_audit [--out results/edge_inference/dataset_audit.json]
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from .common import HF_API, HF_BASE, HF_REPO, RESULTS, cache_dir, sha256_bytes, sha256_json

# README scene table (MTMC_Tracking_2026), copied verbatim; the audit re-derives sharing from data.
README_SPLIT = {**{i: "train" for i in range(0, 20)}, 20: "val", 21: "val", 22: "val",
                23: "test", 24: "test", 25: "test", 26: "test", 27: "test"}
# README: CT2.5 re-renderings -> source scene
README_CT25 = {1: 0, 3: 2, 5: 4, 7: 6, 21: 20, 24: 23}


def scene_name(i: int) -> str:
    return f"Warehouse_{i:03d}"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "flextern-ft0010"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _cached(split: str, scene: str, rel: str) -> bytes:
    p = cache_dir() / "meta" / split / scene / rel
    if p.exists() and p.stat().st_size > 0:
        return p.read_bytes()
    b = _get(f"{HF_BASE}/{split}/{scene}/{rel}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b)
    return b


def download(split: str, scene: str, rel: str) -> Path:
    """Download one raw file (streamed) to the cache; skip if present. Returns local path."""
    dst = cache_dir() / "raw" / split / scene / rel
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(f"{HF_BASE}/{split}/{scene}/{rel}", headers={"User-Agent": "flextern-ft0010"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 22)
            if not b:
                break
            f.write(b)
    tmp.rename(dst)
    return dst


def tree(split: str, scene: str) -> list[dict]:
    url = f"{HF_API}/tree/main/MTMC_Tracking_2026/{split}/{scene}?recursive=true"
    return [x for x in json.loads(_get(url)) if x["type"] == "file"]


def calib_fingerprint(calib: dict) -> str:
    """Hash of everything that defines camera geometry (not names/ordering metadata)."""
    sens = []
    for s in calib["sensors"]:
        sens.append({k: s.get(k) for k in ("intrinsicMatrix", "extrinsicMatrix", "cameraMatrix", "homography")})
    sens.sort(key=lambda d: json.dumps(d, sort_keys=True))
    return sha256_json(sens)


def group_scenes(scenes: dict) -> dict:
    """Union scenes sharing identical camera geometry OR identical map.png bytes.

    Mutates each scene dict with its `layout_group` and returns {group: [scene names]}.
    """
    names = list(scenes)
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for key in ("geometry_sha256", "map_png_sha256"):
        seen = {}
        for n in names:
            k = scenes[n][key]
            if k in seen:
                parent[find(n)] = find(seen[k])
            else:
                seen[k] = n
    groups = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    layout_groups = {f"L{j:02d}": sorted(v) for j, v in enumerate(sorted(groups.values(), key=lambda v: sorted(v)[0]))}
    for g, v in layout_groups.items():
        for n in v:
            scenes[n]["layout_group"] = g
    return layout_groups


def audit(scene_ids=range(0, 23)) -> dict:
    scenes = {}
    for i in scene_ids:
        split, name = README_SPLIT[i], scene_name(i)
        files = tree(split, name)
        calib_b = _cached(split, name, "calibration.json")
        map_b = _cached(split, name, "map.png")
        calib = json.loads(calib_b)
        vids = [x for x in files if x["path"].endswith(".mp4")]
        gt = [x for x in files if x["path"].endswith("ground_truth.json")]
        cams = []
        for s in calib["sensors"]:
            a = {x["name"]: x["value"] for x in s["attributes"]}
            cams.append({"id": s["id"], "fps": float(a["fps"]), "w": int(a["frameWidth"]), "h": int(a["frameHeight"])})
        scenes[name] = {
            "scene_id": i, "readme_split": split,
            "ct25_of": scene_name(README_CT25[i]) if i in README_CT25 else None,
            "n_cameras": len(cams), "cameras": cams,
            "video_bytes": {x["path"].rsplit("/", 1)[-1]: x["size"] for x in vids},
            "ground_truth_bytes": gt[0]["size"] if gt else None,
            "calibration_sha256": sha256_bytes(calib_b),
            "geometry_sha256": calib_fingerprint(calib),
            "map_png_sha256": sha256_bytes(map_b),
        }
    layout_groups = group_scenes(scenes)
    # README-derived expectation vs data-derived
    readme_pairs = sorted((scene_name(a), scene_name(b)) for a, b in README_CT25.items() if scene_name(a) in scenes)
    mismatches = [(a, b) for a, b in readme_pairs if scenes[a]["layout_group"] != scenes[b]["layout_group"]]
    return {"source": f"https://huggingface.co/datasets/{HF_REPO}", "license": "cc-by-4.0 (HF dataset card)",
            "scenes": scenes, "layout_groups": layout_groups,
            "readme_ct25_pairs_not_confirmed_by_data": mismatches}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS / "dataset_audit.json"))
    a = ap.parse_args()
    res = audit()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True))
    for g, v in res["layout_groups"].items():
        sc = res["scenes"][v[0]]
        print(g, v, f"cams={sc['n_cameras']}")
    print("README CT2.5 pairs not confirmed by data:", res["readme_ct25_pairs_not_confirmed_by_data"])


if __name__ == "__main__":
    main()
