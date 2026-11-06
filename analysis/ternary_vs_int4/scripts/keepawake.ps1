# Process-scoped keep-awake for EXP-0011 timing: asks Windows not to sleep while this process runs; changes no persistent power setting.
Add-Type -Namespace W -Name P -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$ES_CONTINUOUS = [uint32]2147483648; $ES_SYSTEM_REQUIRED = [uint32]1
while ($true) {
  [void][W.P]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
  Start-Sleep -Seconds 30
  if (Test-Path "C:\Users\liban\AppData\Local\Temp\claude\C--Users-liban-Documents-flextern\d89e3b97-aca6-495f-9278-c5d62bcb1e21\scratchpad\keepawake.stop") { break }
}
