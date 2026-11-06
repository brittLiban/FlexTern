#!/bin/bash
# usage: build1.sh bitnet|upstream
set -e
cd ~/exp0011
if [ "$1" = "bitnet" ]; then
  cd BitNet
  cmake -B build -DBITNET_X86_TL2=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release > ~/exp0011/build_bitnet.log 2>&1
  cmake --build build --config Release -j 6 >> ~/exp0011/build_bitnet.log 2>&1
  ls build/bin | head -30
else
  cd llama.cpp
  cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ > ~/exp0011/build_upstream.log 2>&1
  cmake --build build --config Release -j 6 >> ~/exp0011/build_upstream.log 2>&1
  ls build/bin | head -40
fi
echo BUILD_OK
