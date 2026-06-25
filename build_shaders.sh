#!/usr/bin/env bash
# Recompile the cooperative-matrix GEMM shaders to SPIR-V.
# Requires a glslangValidator with GL_KHR_cooperative_matrix support.
set -e

GLSLANG="${GLSLANG:-glslangValidator}"
SRC="$(dirname "$0")/shaders/gemm_coopmat.comp"
PSRC="$(dirname "$0")/shaders/wmma_peak.comp"
OUT="$(dirname "$0")/app/src/main/assets/shaders"
TENV="--target-env spirv1.3"

mkdir -p "$OUT"
# GEMM dimension-sweep shaders (currently disabled in native-bench.cpp).
"$GLSLANG" $TENV -DA_TYPE=float16_t -DC_TYPE=float32_t -V "$SRC" -o "$OUT/gemm_fp16_fp32.spv"
"$GLSLANG" $TENV -DA_TYPE=float16_t -DC_TYPE=float16_t -V "$SRC" -o "$OUT/gemm_fp16_fp16.spv"
"$GLSLANG" $TENV -DA_TYPE=int8_t    -DC_TYPE=int32_t   -V "$SRC" -o "$OUT/gemm_s8_s32.spv"
"$GLSLANG" $TENV -DA_TYPE=uint8_t   -DC_TYPE=uint32_t  -V "$SRC" -o "$OUT/gemm_u8_u32.spv"
# WMMA peak-throughput shaders (the active test).
"$GLSLANG" $TENV -DA_TYPE=float16_t -DC_TYPE=float32_t -V "$PSRC" -o "$OUT/wmma_peak_fp16_fp32.spv"
"$GLSLANG" $TENV -DA_TYPE=float16_t -DC_TYPE=float16_t -V "$PSRC" -o "$OUT/wmma_peak_fp16_fp16.spv"
"$GLSLANG" $TENV -DA_TYPE=int8_t    -DC_TYPE=int32_t   -V "$PSRC" -o "$OUT/wmma_peak_s8_s32.spv"
"$GLSLANG" $TENV -DA_TYPE=uint8_t   -DC_TYPE=uint32_t  -V "$PSRC" -o "$OUT/wmma_peak_u8_u32.spv"
echo "Shaders written to $OUT"
