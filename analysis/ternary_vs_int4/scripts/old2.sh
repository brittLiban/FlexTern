#!/bin/bash
cd ~/exp0011/BitNet_old
ls preset_kernels 2>/dev/null | head -5
H=$(ls preset_kernels/*/bitnet-lut-kernels-tl2.h 2>/dev/null | head -1)
K=$(ls preset_kernels/*/kernel_config_tl2.ini 2>/dev/null | head -1)
echo "using $H $K"
cp "$H" include/bitnet-lut-kernels.h; [ -n "$K" ] && cp "$K" include/kernel_config.ini
rm -rf build
cmake -B build -DBITNET_X86_TL2=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release > ~/exp0011/build_bitnet_old.log 2>&1 || { tail -8 ~/exp0011/build_bitnet_old.log; exit 1; }
cmake --build build --config Release -j 6 --target llama-perplexity llama-bench >> ~/exp0011/build_bitnet_old.log 2>&1 || { tail -12 ~/exp0011/build_bitnet_old.log; exit 1; }
tail -2 ~/exp0011/build_bitnet_old.log
cd /tmp
./../home/liban/exp0011/BitNet_old/build/bin/llama-perplexity -m ~/exp0011/models/bitnet-b1.58-2B-4T-i2_s.gguf -f ~/exp0011/data/wiki.test.raw -c 512 --chunks 20 -t 8 -ngl 0 2>&1 | grep -E "Final estimate|pre-token|error" | head
echo OLD_DONE
