"""Complete repository state for run manifests (EXP-0010C).

EXP-0010B manifests used `common.git_info`, which reports dirtiness only under analysis/edge_inference and therefore recorded
"b056ace, clean" for runs made from an untracked, edited analysis/control_gate. This module records the whole repository:
HEAD commit, branch, every dirty/untracked path, and a content hash over the tracked diff plus the bytes of untracked files.
`require_clean_code()` is the guard used before any fresh-holdout run: analysis/ and tests/ must be clean (results/ and docs/
legitimately change while runs are written). It is a new function; Stage A's `git_info` is unchanged.
Git is always invoked with `-c core.autocrlf=true` so that WSL and Windows report the same state for the CRLF working tree.
"""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE_PATHS = ("analysis", "tests")


def _git(*args: str, text: bool = True):
    # core.autocrlf=true: the repo is checked out with CRLF on Windows; git run from WSL without it reports every file as modified
    r = subprocess.run(["git", "-c", "core.autocrlf=true", "-C", str(REPO), *args], capture_output=True, text=text, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr if text else r.stderr[:200]}")
    return r.stdout


def porcelain(paths: tuple[str, ...] | None = None) -> list[str]:
    args = ["status", "--porcelain", "--untracked-files=all"]
    if paths:
        args += ["--", *paths]
    return [ln for ln in _git(*args).splitlines() if ln.strip()]


def dirty_hash() -> str:
    """SHA-256 over `git diff HEAD` (tracked changes) and the bytes of every untracked file (path-sorted)."""
    h = hashlib.sha256()
    h.update(_git("diff", "HEAD").encode())
    for ln in sorted(porcelain()):
        if ln.startswith("??"):
            p = REPO / ln[3:].strip().strip('"')
            h.update(ln.encode())
            if p.is_file():
                h.update(p.read_bytes())
    return h.hexdigest()


def repo_state() -> dict:
    status = porcelain()
    return {"commit": _git("rev-parse", "HEAD").strip(), "branch": _git("rev-parse", "--abbrev-ref", "HEAD").strip(),
            "dirty": bool(status), "n_dirty_paths": len(status), "dirty_paths": status[:200],
            "dirty_code_paths": porcelain(CODE_PATHS), "tree_state_sha256": dirty_hash() if status else None,
            "scope": "whole repository (git status --porcelain --untracked-files=all)"}


def require_clean_code() -> dict:
    """Raise SystemExit unless analysis/ and tests/ are clean. Returns the repo state for the manifest."""
    dirty = porcelain(CODE_PATHS)
    if dirty:
        raise SystemExit("refusing to run: uncommitted changes under analysis/ or tests/ (commit the locked code first):\n  " + "\n  ".join(dirty[:20]))
    return repo_state()
