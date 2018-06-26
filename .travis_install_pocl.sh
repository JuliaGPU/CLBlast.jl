#!/bin/bash

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
mkdir -p ~/.local
export OPENCL_VENDOR_PATH=~/.local/etc/OpenCL/vendors

git clone https://github.com/pocl/pocl.git
cd pocl
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_ICD=1 -DCMAKE_INSTALL_PREFIX=~/.local/ ..
make -j `nprocs`
make install

clinfo # should find pocl
