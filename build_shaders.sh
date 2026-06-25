#!/usr/bin/env bash
# Recompile the cooperative-matrix GEMM shaders to SPIR-V.
# Requires a glslangValidator with GL_KHR_cooperative_matrix support.
set -e

GLSLANG="${GLSLANG:-glslangValidator}"
SRC="$(dirname "$0")/shaders/gemm_coopmat.comp"
CSRC="$(dirname "$0")/shaders/gemm_coopmat_compute.comp"
OUT="$(dirname "$0")/app/src/main/assets/shaders"
TENV="--target-env spirv1.3"

mkdir -p "$OUT"
# Memory GEMM (streams A/B from global memory).
"$GLSLANG" $TENV -DA_TYPE=float16_t -DC_TYPE=float32_t -V "$SRC" -o "$OUT/gemm_fp16_fp32.spv"
"$GLSLANG" $TENV -DA_TYPE=float16_t -DC_TYPE=float16_t -V "$SRC" -o "$OUT/gemm_fp16_fp16.spv"
"$GLSLANG" $TENV -DA_TYPE=int8_t    -DC_TYPE=int32_t   -V "$SRC" -o "$OUT/gemm_s8_s32.spv"
"$GLSLANG" $TENV -DA_TYPE=uint8_t   -DC_TYPE=uint32_t  -V "$SRC" -o "$OUT/gemm_u8_u32.spv"
# Compute-bound (one cache-resident tile, MulAdd over K; no DRAM streaming).
"$GLSLANG" $TENV -DA_TYPE=float16_t -DC_TYPE=float32_t -V "$CSRC" -o "$OUT/gemm_compute_fp16_fp32.spv"
"$GLSLANG" $TENV -DA_TYPE=float16_t -DC_TYPE=float16_t -V "$CSRC" -o "$OUT/gemm_compute_fp16_fp16.spv"
"$GLSLANG" $TENV -DA_TYPE=int8_t    -DC_TYPE=int32_t   -V "$CSRC" -o "$OUT/gemm_compute_s8_s32.spv"
"$GLSLANG" $TENV -DA_TYPE=uint8_t   -DC_TYPE=uint32_t  -V "$CSRC" -o "$OUT/gemm_compute_u8_u32.spv"
echo "Shaders written to $OUT"
