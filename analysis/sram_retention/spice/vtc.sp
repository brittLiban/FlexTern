* ASAP7 6T hold VTC: force q, read qb (inverter 2 with its pass gate attached, WL=0, BL=VDD)
.control
pre_osdi bsimcmg.osdi
.endc
.include MODELFILE
.param vdd=VDDV
Vdd vdd 0 {vdd}
Vwl wl 0 0
Vblb blb 0 {vdd}
Vq q 0 0
Npu2 qb q vdd vdd pmos_sram nfin=1
Npd2 qb q 0 0 nmos_sram nfin=1
Npg2 blb wl qb 0 nmos_sram nfin=1
.temp TEMPC
.control
dc Vq 0 VDDV 0.005
wrdata VTCOUT v(qb)
.endc
.end
