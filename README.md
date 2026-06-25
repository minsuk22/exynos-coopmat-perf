# Exynos Cooperative-Matrix GEMM Benchmark

An Android port of [jeffbolznv/vk_cooperative_matrix_perf](https://github.com/jeffbolznv/vk_cooperative_matrix_perf),
rewritten to analyze the GPU performance of **Exynos 2600** (and other GPUs that
expose `VK_KHR_cooperative_matrix`).

## What it does

The app runs a headless Vulkan compute benchmark (no rendering) that:

1. Enumerates Vulkan devices and reports which expose `VK_KHR_cooperative_matrix`.
2. Dumps every advertised cooperative-matrix configuration (M×N×K, A/B/C/D
   component types, scope) — useful for analysis on its own.
3. Reports whether the features the GEMM path needs are present
   (`cooperativeMatrix`, `vulkanMemoryModel`, `shaderFloat16`/`shaderInt8`,
   16-/8-bit storage, `subgroupSizeControl`, `computeFullSubgroups`).
4. Runs `D = alpha·A·B + beta·C` over a small sweep of tile/workgroup sizes for
   each supported type combination (fp16→fp32, fp16→fp16, s8→s32, u8→u32) and
   reports **TFLOPS**.

Results are shown on screen, written to logcat (tag `VKCOOP`), and saved to
`Android/data/com.exynos.vkcoopmat/files/coopmat_report.txt`.

For a detailed breakdown of the matrix operation and the data types it
exercises, see [docs/matrix-operation-analysis.md](docs/matrix-operation-analysis.md)
([한글](docs/matrix-operation-analysis.ko.md)).

왜 MATMUL에서는 FP16/INT8 성능 차이가 사라지는지(메모리 바운드)에 대한 설명:
[docs/matmul-vs-wmma-peak-ko.md](docs/matmul-vs-wmma-peak-ko.md).

## Differences from the upstream NVIDIA benchmark

| Upstream | This port |
| --- | --- |
| `VK_NV_cooperative_matrix2` "workgroup" path | dropped (NVIDIA-only) |
| `VK_KHR_cooperative_matrix` subgroup path | kept, the focus here |
| `buffer_reference` + `bufferDeviceAddress` | replaced with plain SSBO bindings (no BDA needed) |
| uvec4-packed shared-memory tiling | removed; `coopMatLoad` reads straight from global memory |
| desktop command-line exe | Android APK with a one-tap UI |

Removing `buffer_reference` and the 128-bit shared-memory packing trades some
peak throughput for much broader mobile-driver compatibility. The shared-memory
tiled version can be added back later for higher numbers once the simple path is
confirmed working on the device.

## Project layout

```
app/src/main/cpp/native-bench.cpp   # JNI + Vulkan benchmark (the port)
app/src/main/cpp/CMakeLists.txt
shaders/gemm_coopmat.comp           # portable subgroup coopmat GEMM (GLSL source)
app/src/main/assets/shaders/*.spv   # precompiled SPIR-V (built from the .comp)
app/src/main/java/.../MainActivity.java
```

## Rebuilding the shaders

Requires a `glslangValidator` with `GL_KHR_cooperative_matrix` support
(e.g. from the Vulkan SDK or the KhronosGroup/glslang CI builds):

```sh
glslangValidator --target-env spirv1.3 -DA_TYPE=float16_t -DC_TYPE=float32_t -V \
    shaders/gemm_coopmat.comp -o app/src/main/assets/shaders/gemm_fp16_fp32.spv
# ...repeat for fp16/fp16, int8_t/int32_t, uint8_t/uint32_t
```

(See `build_shaders.sh`.)

## Building the APK

Prerequisites: JDK 17, Android SDK with platform-34, build-tools 34.0.0,
CMake 3.22.1, and NDK 26.3.11579264. Point `local.properties` at your SDK.

```sh
./gradlew assembleDebug
# output: app/build/outputs/apk/debug/app-debug.apk
```

Install and run:

```sh
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.exynos.vkcoopmat/.MainActivity
adb logcat -s VKCOOP
```
