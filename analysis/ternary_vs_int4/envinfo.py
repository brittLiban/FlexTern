"""Environment snapshot recorded in every manifest (EXP-0011). Runs in WSL2; Windows power state is read through powershell.exe."""
from __future__ import annotations

import os
import platform
import subprocess
import time

from . import config as C


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001 - environment probing must never abort a run
        return f"ERROR {e}"


def power_state() -> dict:
    """AC/battery and Windows power plan as seen from the Windows host. WSL2 cannot read CPU frequency or temperature."""
    out = _run(["powershell.exe", "-NoProfile", "-Command",
                "$b=Get-CimInstance Win32_Battery; Write-Output ('BAT ' + $b.BatteryStatus + ' ' + $b.EstimatedChargeRemaining); "
                "powercfg /getactivescheme"], 30).replace("\x00", "")
    return {"raw": out.splitlines()}


def power_check() -> dict:
    """Parse Win32_Battery: BatteryStatus 1/4/5 = discharging/low/critical (refuse to time); 2/3/6/7/8/9 = on AC or charging."""
    raw = power_state()["raw"]
    bat = next((ln for ln in raw if ln.startswith("BAT ")), "")
    parts = bat.split()
    status = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else None
    charge = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else None
    on_ac = status is not None and status not in (1, 4, 5)
    return {"battery_status": status, "charge_pct": charge, "on_ac": on_ac, "ok": bool(on_ac and charge is not None and charge >= C.MIN_CHARGE_PCT),
            "plan": next((ln for ln in raw if "Power Scheme" in ln), "")}


def snapshot(light: bool = False) -> dict:
    d = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "loadavg": os.getloadavg(),
         "mem_available_mb": int([ln for ln in open("/proc/meminfo") if ln.startswith("MemAvailable")][0].split()[1]) // 1024}
    d["power_check"] = power_check()                     # recorded for every invocation (also in light mode)
    if not light:
        d.update(power=power_state(), kernel=platform.release(), python=platform.python_version(),
                 lscpu=_run(["lscpu"]).splitlines()[:25], nproc=os.cpu_count())
    return d


def runtime_versions() -> dict:
    out = {}
    for name, b in C.RUNTIME_BIN.items():
        repo = b.parents[1]
        out[name] = {"bin": str(b), "commit": _run(["git", "-C", str(repo), "rev-parse", "HEAD"]),
                     "submodules": _run(["git", "-C", str(repo), "submodule", "status"]) if name == "bitnet" else "",
                     "build_info": _run([str(b / "llama-bench"), "--version"])[:300]}
    return out
