#!/bin/bash
# build bitnet.cpp as of the 2B4T release era (commit 765741d, 2025-05-21 "update submodule") next to the current checkout
set -e
cd ~/exp0011
if [ ! -d BitNet_old ]; then cp -r BitNet BitNet_old; fi
cd BitNet_old
rm -rf build build_avx2
git checkout -q 765741d
git submodule update --init --recursive 2>&1 | tail -3
echo "BitNet_old $(git rev-parse HEAD) $(git log -1 --format=%cd)"; git submodule status
cmake -B build -DBITNET_X86_TL2=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release > ~/exp0011/build_bitnet_old.log 2>&1
cmake --build build --config Release -j 6 --target llama-perplexity llama-bench >> ~/exp0011/build_bitnet_old.log 2>&1
tail -2 ~/exp0011/build_bitnet_old.log
echo OLD_BUILD_DONE
