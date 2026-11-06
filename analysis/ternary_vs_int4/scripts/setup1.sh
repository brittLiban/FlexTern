#!/bin/bash
set -e
mkdir -p ~/exp0011 && cd ~/exp0011
if [ ! -d BitNet ]; then git clone --recursive https://github.com/microsoft/BitNet.git; fi
if [ ! -d llama.cpp ]; then git clone https://github.com/ggml-org/llama.cpp.git; fi
(cd BitNet && echo "BitNet $(git rev-parse HEAD) $(git log -1 --format=%cd)" && git submodule status)
(cd llama.cpp && echo "llama.cpp $(git rev-parse HEAD) $(git log -1 --format=%cd)")
