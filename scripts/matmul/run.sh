#!/bin/bash
BENCHDNN=./third-party/oneDNN/build/tests/benchdnn/benchdnn

export OMP_NUM_THREADS=1
export ONEDNN_VERBOSE=1      # prints which kernel/ISA path is used
# optional pinning

ONEDNN_MAX_CPU_ISA=AVX512_CORE_AMX \
  taskset -c 1 numactl --localalloc \
    ./${BENCHDNN} --mode=p --matmul \
      --dt=bf16:bf16:f32 --stag=ab --wtag=ab --dtag=ab --bia-dt=undef \
      m1024n1024k1024