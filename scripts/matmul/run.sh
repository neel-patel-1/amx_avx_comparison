#!/bin/bash
BENCHDNN=./third-party/oneDNN/build/tests/benchdnn/benchdnn

export OMP_NUM_THREADS=1
export ONEDNN_VERBOSE=1      # prints which kernel/ISA path is used
# optional pinning

# 'any' lets the library decide which physical layout will be used for a certain memory descriptor of the given problem -- hopefully useful for AMX

ONEDNN_MAX_CPU_ISA=AVX512_CORE_BF16 \
  taskset -c 1 numactl --localalloc \
    ./${BENCHDNN} --mode=p --matmul \
      --dt=bf16:bf16:f32 --stag=any --wtag=any --dtag=any --bia-dt=undef \
      10x30:30x20