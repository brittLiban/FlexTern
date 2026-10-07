# EXP-0009 Tier-2 SPICE corroboration (ASAP7 6T SRAM cell hold leakage)

Not primary evidence. Used only for voltage and temperature RATIOS; absolute leakage is anchored on the T1 macro papers (R01, R02).

Reproduce (WSL/Linux):

1. `sudo apt-get install ngspice` (45.2 used; the Debian build has no built-in BSIM-CMG, so the model is loaded through OSDI).
2. Download OpenVAF 23.5.0 (openvaf_23_5_0_linux_amd64.tar.gz from openvaf.semimod.de) and the BSIM-CMG 110 Verilog-A sources from `github.com/dwarning/VA-Models` (`code/bsimcmg/vacode110/`); run `openvaf bsimcmg.va` to get `bsimcmg.osdi`.
3. Download the ASAP7 r1p7 model cards `models/hspice/7nm_{TT,SS,FF}_160803.pm` from `github.com/The-OpenROAD-Project/asap7_pdk_r1p7` (BSD-3) and convert them for OSDI:
   `sed -E -e 's/^\.model (nmos_[a-z]+) nmos level = 72/.model \1 bsimcmg devtype=1/' -e 's/^\.model (pmos_[a-z]+) pmos level = 72/.model \1 bsimcmg devtype=0/' -e 's/\+version = 107 /+/' 7nm_TT_160803.pm > asap7_TT_osdi.pm`
4. For each corner / temperature / voltage, substitute MODELFILE / TEMPC / VDDV in `leak.sp` and run `ngspice -b`; collect `itot` (the current drawn from VDD, BL and BLB with WL = 0) into `asap7_cell_leakage.csv`.
5. For the hold static noise margin, substitute into `vtc.sp` (`VTCOUT` = output file), run, and fit the largest square between the hold VTC and its mirror (`hold_snm.csv`, TT corner only, no Monte Carlo).

Cell: all six devices are single-fin ASAP7 `sram` Vt devices (PU pmos_sram, PD and PG nmos_sram), WL = 0, BL = BLB = VDD, q = VDD, qb = 0. Array only; no periphery. The PDK, OpenVAF and the Verilog-A sources are not committed.
