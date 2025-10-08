#!/bin/bash
BENCHDNN=./third-party/oneDNN/build/tests/benchdnn/benchdnn

export OMP_NUM_THREADS=1
export ONEDNN_VERBOSE=0      # prints which kernel/ISA path is used
# optional pinning

# 'any' lets the library decide which physical layout will be used for a certain memory descriptor of the given problem -- hopefully useful for AMX

ISAS=(
  AVX512_CORE_AMX
  AVX512_CORE_BF16
)

DIMS=(
  1024x1024:1024x1024
)
mkdir -p logs

for ISA in "${ISAS[@]}"; do
  echo "Running with ISA: $ISA"
  for DIM in "${DIMS[@]}"; do
    ONEDNN_MAX_CPU_ISA=$ISA \
      taskset -c 1 numactl --localalloc \
        ./${BENCHDNN} --mode=p --matmul \
          --dt=bf16:bf16:f32 --stag=any --wtag=any --dtag=any --bia-dt=undef \
          $DIM | tee logs/matmul_${ISA}_dim${DIM//:/x}.log
  done
done