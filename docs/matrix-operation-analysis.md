# Matrix Operation & Data-Type Analysis

This document describes **what matrix operation this benchmark measures** and
**which data types it exercises**, based on the actual source
(`shaders/gemm_coopmat.comp` and `app/src/main/cpp/native-bench.cpp`).

## The operation: GEMM (matrix multiply-accumulate)

The benchmark measures a **General Matrix Multiply (GEMM)**:

```
D = alpha * (A x B) + beta * C
```

- `A` is `M x K`, `B` is `K x N`, `C` / `D` are `M x N`, all **row-major**
  (`gemm_coopmat.comp:13`).
- At benchmark time the host sets `alpha = 1.0`, `beta = 0.0`
  (`native-bench.cpp:708`), so the timed work is effectively a pure matrix
  product **`D = A x B`**.
- Problem size is ~`2048 x 2048 x 2048`, rounded up to tile / `lK` multiples
  (`native-bench.cpp:603-606`), executed **10 times per dispatch**
  (`repeats = 10`).

The hardware primitive is the **cooperative-matrix** instruction
`coopMatMulAdd` at **subgroup scope** (`gemm_coopmat.comp:90`). It multiply-
accumulates a small `lM x lN x lK` tile on the GPU's matrix/tensor engine; the
shader stacks these tiles over a `C_ROWS x C_COLS` accumulator grid per subgroup
and marches over `K` to build the full GEMM.

### Performance metric

Throughput is reported in **TFLOPS** (`native-bench.cpp:778-779`):

```
flops  = 2 * M * N * K * repeats     // 1 multiply + 1 add per element = 2 FLOP
tflops = flops / seconds / 1e12
```

## The data types: 4 combinations

The benchmarked combinations are listed in `kCombos`
(`native-bench.cpp:199-204`). Each pairs an **input type (A, B)** with an
**output / accumulator type (C, D)**:

| # | Input (A, B)        | Output / Accumulate (C, D) | Class          | Shader                 |
|---|---------------------|----------------------------|----------------|------------------------|
| 1 | **fp16** (float16)  | **fp32** (float32)         | floating-point | `gemm_fp16_fp32.spv`   |
| 2 | **fp16**            | **fp16**                   | floating-point | `gemm_fp16_fp16.spv`   |
| 3 | **s8** (signed i8)  | **s32** (signed i32)       | integer        | `gemm_s8_s32.spv`      |
| 4 | **u8** (unsigned i8)| **u32** (unsigned i32)     | integer        | `gemm_u8_u32.spv`      |

These are the combinations that matter most for ML / inference acceleration:

- **fp16 -> fp32** — 16-bit half-precision inputs save memory/bandwidth while a
  32-bit accumulator preserves precision. This is the standard deep-learning
  training/inference pattern.
- **fp16 -> fp16** — accumulate in 16-bit too: faster, but risks precision loss.
- **s8 -> s32 / u8 -> u32** — the standard for **INT8 quantized inference**:
  8-bit inputs with a 32-bit accumulator to avoid overflow.

## Type selection is driven by the device, not hard-coded

The 4 combinations above are **candidates**, not an unconditional run list. The
actual measured set is decided at runtime:

1. **Dump what the GPU advertises.**
   `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` enumerates every
   supported `(M, N, K, A/B/C/D type, scope)` configuration
   (`native-bench.cpp:333-351`).

2. **Feature gating** (`native-bench.cpp:381-386`):
   - The **float** path (combos 1, 2) needs `shaderFloat16` +
     `storageBuffer16BitAccess`.
   - The **int** path (combos 3, 4) needs `shaderInt8` +
     `storageBuffer8BitAccess`.
   - Both need `cooperativeMatrix`, `vulkanMemoryModel`, `subgroupSizeControl`,
     and `computeFullSubgroups`.

3. **Shape matching.** For each surviving combo, the code finds a
   **subgroup-scope** advertised shape whose A/B/C/D types match exactly
   (`native-bench.cpp:480-486`), then injects that shape's `lM` / `lN` / `lK`
   into the shader via specialization constants.

4. **Graceful degradation.** If nothing matches, the GEMM is skipped and the app
   emits a **capabilities-only report** instead of failing
   (`native-bench.cpp:388-394`, `:490-497`).

So on Exynos 2600 the actually-measured set depends on which types the driver
exposes on its matrix engine.

## How each type is measured

- **Tile sweep** — for every type, three workgroup/tile configurations are tried
  (`native-bench.cpp:583-587`): `{invocations 128, 2x2}`, `{256, 2x2}`,
  `{256, 4x2}`, varying the per-subgroup accumulator grid (`C_ROWS x C_COLS`)
  and workgroup size to search for a good configuration.
- **Subgroup control** — the subgroup size is pinned via
  `requiredSubgroupSize` and full subgroups are forced
  (`native-bench.cpp:721-727`).
- **Timing** — 3 warmup submits, then one timed submit; each dispatch internally
  repeats 10 times with a memory barrier between iterations
  (`native-bench.cpp:752-775`).
- **Inputs** — filled with small integers in `-2..2`
  (`native-bench.cpp:661-666`) to avoid fp16 overflow / precision issues.

## Summary

A benchmark that measures the **throughput (TFLOPS) of GEMM
(`D = alpha*A*B + beta*C`) on the GPU's subgroup-scope cooperative-matrix engine**,
across **four data-type combinations (fp16/fp16, fp16/fp32, s8/s32, u8/u32)**,
sweeping tile configurations, and restricted to whatever the device actually
supports.
