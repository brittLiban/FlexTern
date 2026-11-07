# Read-only snapshot of the machine's power state (does NOT change any power or sleep setting). EXP-0013 Phase 1.1 closeout.
# Usage: powershell -NoProfile -File scripts/capture_power_state.ps1 LABEL >> results/.../power_state.jsonl
param([string]$Label = "")
$b = Get-CimInstance Win32_Battery
$p = Get-CimInstance -Namespace root\wmi -ClassName BatteryStatus -ErrorAction SilentlyContinue
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
[pscustomobject]@{
  label = $Label; time = (Get-Date).ToString('o');
  win32_battery_status = $b.BatteryStatus; charge_pct = $b.EstimatedChargeRemaining;
  power_online = $p.PowerOnline; charging = $p.Charging; discharging = $p.Discharging;
  active_power_scheme = ((powercfg /getactivescheme) -join ' ');
  cpu = $cpu.Name; cpu_current_mhz = $cpu.CurrentClockSpeed; cpu_max_mhz = $cpu.MaxClockSpeed; cpu_load_pct = $cpu.LoadPercentage
} | ConvertTo-Json -Compress
