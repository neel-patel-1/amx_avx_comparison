#!/usr/bin/env bash
set -euo pipefail

OBJDBG_OPTS="-d -M intel"

# common CMake flags that keep ggml AMX runtime support disabled
COMMON_CMAKE="-DGGML_AMX_TILE=OFF -DGGML_AMX_INT8=OFF -DGGML_AMX_BF16=OFF -DGGML_NATIVE=OFF"

# variants: name -> CFLAGS
declare -A VARIANTS
VARIANTS[baseline]="-O3 -march=x86-64"
VARIANTS[allow_avx]="-O3 -march=haswell"  # AVX2 enabled, no AVX512
VARIANTS[no_avx512]="-O3 -march=native -mno-avx512f -mno-avx512bw -mno-avx512vl -mno-avx512dq"
VARIANTS[no_avx]="-O3 -march=native -mno-avx -mno-avx2 -mno-avx512f -mno-avx512bw -mno-avx512vl"

# patterns to search in objdump output
PATTERNS_AMX="tilezero|tileloadd|tilerelease|tilestored"
PATTERNS_AVX512="kmov|kxor|vpdp|vpmadd|vpermt2|vfmadd132|zmm"
PATTERNS_AVX="ymm|xmm|vaddps|vfmadd|vmovdqu|vmulps|vblend"

# helper: check object for patterns
check_binary_for_patterns() {
  local builddir="$1"
  local bin="$builddir/bin/llama-cli"
  if [[ ! -f "$bin" ]]; then
    echo "no binary at $bin to inspect"
    return
  fi
  echo "Inspecting $bin for ISA patterns..."
  objdump $OBJDBG_OPTS "$bin" | egrep -n --color=always "$PATTERNS_AMX|$PATTERNS_AVX512|$PATTERNS_AVX" || true
}

# main loop
for name in baseline allow_avx no_avx512 no_avx; do
  builddir="build_${name}"
  echo
  echo "=== BUILD VARIANT: $name ==="
  echo "CFLAGS: ${VARIANTS[$name]}"

  rm -rf "$builddir"
  cmake -B "$builddir" $COMMON_CMAKE -DCMAKE_C_FLAGS="${VARIANTS[$name]}" -DCMAKE_CXX_FLAGS="${VARIANTS[$name]}"
  cmake --build "$builddir" -j"$(nproc)"

  # inspect binary
  check_binary_for_patterns "$builddir"

  # quick smoke-run (version) to ensure it runs; replace with a real bench if you want
  if [[ -x "$builddir/bin/llama-cli" ]]; then
    echo "Running quick smoke test (llama-cli --version):"
    "$builddir/bin/llama-cli" --version || true
  fi
done

echo "All variants built. Use the objdump output and run your benchmark to measure perf deltas."