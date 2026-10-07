* ASAP7 6T HD SRAM bitcell hold-state leakage and hold VTC (EXP-0009 T2)
.control
pre_osdi bsimcmg.osdi
.endc
.include MODELFILE
.param vdd=VDDV
Vdd vdd 0 {vdd}
Vwl wl 0 0
Vbl bl 0 {vdd}
Vblb blb 0 {vdd}
Npu1 q qb vdd vdd pmos_sram nfin=1
Npd1 q qb 0 0 nmos_sram nfin=1
Npu2 qb q vdd vdd pmos_sram nfin=1
Npd2 qb q 0 0 nmos_sram nfin=1
Npg1 bl wl q 0 nmos_sram nfin=1
Npg2 blb wl qb 0 nmos_sram nfin=1
.nodeset v(q)={vdd} v(qb)=0
.temp TEMPC
.control
op
let itot = -(i(vdd)+i(vbl)+i(vblb))
print v(q) v(qb) i(vdd) i(vbl) i(vblb) itot
.endc
.end
