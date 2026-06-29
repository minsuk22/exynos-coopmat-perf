/*
 * Exynos cooperative-matrix GEMM benchmark (native/JNI).
 *
 * Ported from jeffbolznv/vk_cooperative_matrix_perf for Android/Exynos:
 *   - Targets VK_KHR_cooperative_matrix at subgroup scope (the NVIDIA
 *     VK_NV_cooperative_matrix2 "workgroup" path is intentionally dropped).
 *   - Uses plain SSBO bindings instead of buffer_reference, so the device does
 *     not need bufferDeviceAddress.
 *   - Runs headless compute (no swapchain/surface) and returns a text report
 *     to Java; everything is also mirrored to logcat (tag "VKCOOP").
 *
 * The benchmark first dumps the device's advertised cooperative-matrix shapes
 * (useful on its own for analyzing the GPU), then, if the required features are
 * present, runs D = alpha*A*B + beta*C across a small sweep of tile sizes and
 * reports TFLOPS.
 */

#include <jni.h>
#include <android/log.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

#define LOG_TAG "VKCOOP"
#define ALOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define ALOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// ---------------------------------------------------------------------------
// Fallback definitions for VK_KHR_cooperative_matrix, in case the NDK Vulkan
// headers predate it. Newer NDKs already define these.
// ---------------------------------------------------------------------------
#if !defined(VK_KHR_cooperative_matrix)
#define VK_KHR_cooperative_matrix 1
#define VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME "VK_KHR_cooperative_matrix"

typedef enum VkComponentTypeKHR {
    VK_COMPONENT_TYPE_FLOAT16_KHR = 0,
    VK_COMPONENT_TYPE_FLOAT32_KHR = 1,
    VK_COMPONENT_TYPE_FLOAT64_KHR = 2,
    VK_COMPONENT_TYPE_SINT8_KHR = 3,
    VK_COMPONENT_TYPE_SINT16_KHR = 4,
    VK_COMPONENT_TYPE_SINT32_KHR = 5,
    VK_COMPONENT_TYPE_SINT64_KHR = 6,
    VK_COMPONENT_TYPE_UINT8_KHR = 7,
    VK_COMPONENT_TYPE_UINT16_KHR = 8,
    VK_COMPONENT_TYPE_UINT32_KHR = 9,
    VK_COMPONENT_TYPE_UINT64_KHR = 10,
    VK_COMPONENT_TYPE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkComponentTypeKHR;

typedef enum VkScopeKHR {
    VK_SCOPE_DEVICE_KHR = 1,
    VK_SCOPE_WORKGROUP_KHR = 2,
    VK_SCOPE_SUBGROUP_KHR = 3,
    VK_SCOPE_QUEUE_FAMILY_KHR = 5,
    VK_SCOPE_MAX_ENUM_KHR = 0x7FFFFFFF
} VkScopeKHR;

#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR ((VkStructureType)1000506000)
#define VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR ((VkStructureType)1000506001)
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR ((VkStructureType)1000506002)

typedef struct VkCooperativeMatrixPropertiesKHR {
    VkStructureType sType;
    void* pNext;
    uint32_t MSize;
    uint32_t NSize;
    uint32_t KSize;
    VkComponentTypeKHR AType;
    VkComponentTypeKHR BType;
    VkComponentTypeKHR CType;
    VkComponentTypeKHR ResultType;
    VkBool32 saturatingAccumulation;
    VkScopeKHR scope;
} VkCooperativeMatrixPropertiesKHR;

typedef struct VkPhysicalDeviceCooperativeMatrixFeaturesKHR {
    VkStructureType sType;
    void* pNext;
    VkBool32 cooperativeMatrix;
    VkBool32 cooperativeMatrixRobustBufferAccess;
} VkPhysicalDeviceCooperativeMatrixFeaturesKHR;

typedef VkResult (VKAPI_PTR *PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)(
    VkPhysicalDevice, uint32_t*, VkCooperativeMatrixPropertiesKHR*);
#endif // VK_KHR_cooperative_matrix

// ---------------------------------------------------------------------------
// Fallback definitions for VK_KHR_pipeline_executable_properties, used to dump
// the driver's per-pipeline executables: statistics (register usage, etc.) and
// internal representations (the GPU ISA / disassembly), when the driver exposes
// them. Newer NDKs already define these.
// ---------------------------------------------------------------------------
#if !defined(VK_KHR_pipeline_executable_properties)
#define VK_KHR_pipeline_executable_properties 1
#define VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME "VK_KHR_pipeline_executable_properties"

#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR ((VkStructureType)1000269000)
#define VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR ((VkStructureType)1000269001)
#define VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR ((VkStructureType)1000269002)
#define VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR ((VkStructureType)1000269003)
#define VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR ((VkStructureType)1000269004)
#define VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR ((VkStructureType)1000269005)

#define VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR ((VkPipelineCreateFlagBits)0x00000040)
#define VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR ((VkPipelineCreateFlagBits)0x00000080)

typedef struct VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR {
    VkStructureType sType;
    void* pNext;
    VkBool32 pipelineExecutableInfo;
} VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR;

typedef struct VkPipelineInfoKHR {
    VkStructureType sType;
    const void* pNext;
    VkPipeline pipeline;
} VkPipelineInfoKHR;

typedef struct VkPipelineExecutablePropertiesKHR {
    VkStructureType sType;
    void* pNext;
    VkShaderStageFlags stages;
    char name[VK_MAX_DESCRIPTION_SIZE];
    char description[VK_MAX_DESCRIPTION_SIZE];
    uint32_t subgroupSize;
} VkPipelineExecutablePropertiesKHR;

typedef struct VkPipelineExecutableInfoKHR {
    VkStructureType sType;
    const void* pNext;
    VkPipeline pipeline;
    uint32_t executableIndex;
} VkPipelineExecutableInfoKHR;

typedef enum VkPipelineExecutableStatisticFormatKHR {
    VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR = 0,
    VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR = 1,
    VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR = 2,
    VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR = 3,
    VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_MAX_ENUM_KHR = 0x7FFFFFFF
} VkPipelineExecutableStatisticFormatKHR;

typedef union VkPipelineExecutableStatisticValueKHR {
    VkBool32 b32;
    int64_t  i64;
    uint64_t u64;
    double   f64;
} VkPipelineExecutableStatisticValueKHR;

typedef struct VkPipelineExecutableStatisticKHR {
    VkStructureType sType;
    void* pNext;
    char name[VK_MAX_DESCRIPTION_SIZE];
    char description[VK_MAX_DESCRIPTION_SIZE];
    VkPipelineExecutableStatisticFormatKHR format;
    VkPipelineExecutableStatisticValueKHR value;
} VkPipelineExecutableStatisticKHR;

typedef struct VkPipelineExecutableInternalRepresentationKHR {
    VkStructureType sType;
    void* pNext;
    char name[VK_MAX_DESCRIPTION_SIZE];
    char description[VK_MAX_DESCRIPTION_SIZE];
    VkBool32 isText;
    size_t dataSize;
    void* pData;
} VkPipelineExecutableInternalRepresentationKHR;

typedef VkResult (VKAPI_PTR *PFN_vkGetPipelineExecutablePropertiesKHR)(
    VkDevice, const VkPipelineInfoKHR*, uint32_t*, VkPipelineExecutablePropertiesKHR*);
typedef VkResult (VKAPI_PTR *PFN_vkGetPipelineExecutableStatisticsKHR)(
    VkDevice, const VkPipelineExecutableInfoKHR*, uint32_t*, VkPipelineExecutableStatisticKHR*);
typedef VkResult (VKAPI_PTR *PFN_vkGetPipelineExecutableInternalRepresentationsKHR)(
    VkDevice, const VkPipelineExecutableInfoKHR*, uint32_t*, VkPipelineExecutableInternalRepresentationKHR*);
#endif // VK_KHR_pipeline_executable_properties

// ---------------------------------------------------------------------------
// Fallback for VK_KHR_shader_integer_dot_product (dp4a), used by the optimized
// FMA int8 GEMM. Core in Vulkan 1.3; newer NDKs define it.
// ---------------------------------------------------------------------------
#if !defined(VK_KHR_shader_integer_dot_product)
#define VK_KHR_shader_integer_dot_product 1
#define VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME "VK_KHR_shader_integer_dot_product"
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES ((VkStructureType)1000280000)
typedef struct VkPhysicalDeviceShaderIntegerDotProductFeatures {
    VkStructureType sType;
    void* pNext;
    VkBool32 shaderIntegerDotProduct;
} VkPhysicalDeviceShaderIntegerDotProductFeatures;
#endif // VK_KHR_shader_integer_dot_product

#define ARRAY_LENGTH(x) (sizeof(x) / sizeof((x)[0]))

// ---------------------------------------------------------------------------
// Report accumulator: appends to a string and echoes to logcat.
// ---------------------------------------------------------------------------
struct Report {
    std::string text;
    void line(const char *fmt, ...) {
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        text += buf;
        text += '\n';
        ALOGI("%s", buf);
    }
    // Append a possibly large multi-line blob (e.g. ISA disassembly). The fixed
    // buffer in line() would truncate it, and logcat truncates long lines, so we
    // store it whole and echo it to logcat one line at a time.
    void raw(const std::string &s) {
        text += s;
        size_t start = 0;
        while (start < s.size()) {
            size_t nl = s.find('\n', start);
            size_t len = (nl == std::string::npos) ? s.size() - start : nl - start;
            ALOGI("%.*s", (int)len, s.c_str() + start);
            if (nl == std::string::npos) break;
            start = nl + 1;
        }
    }
};

static const char *componentTypeName(VkComponentTypeKHR t) {
    switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "fp16";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "fp32";
    case VK_COMPONENT_TYPE_FLOAT64_KHR: return "fp64";
    case VK_COMPONENT_TYPE_SINT8_KHR:   return "s8";
    case VK_COMPONENT_TYPE_SINT16_KHR:  return "s16";
    case VK_COMPONENT_TYPE_SINT32_KHR:  return "s32";
    case VK_COMPONENT_TYPE_UINT8_KHR:   return "u8";
    case VK_COMPONENT_TYPE_UINT32_KHR:  return "u32";
    default: return "?";
    }
}

static const char *scopeName(VkScopeKHR s) {
    switch (s) {
    case VK_SCOPE_DEVICE_KHR: return "device";
    case VK_SCOPE_WORKGROUP_KHR: return "workgroup";
    case VK_SCOPE_SUBGROUP_KHR: return "subgroup";
    case VK_SCOPE_QUEUE_FAMILY_KHR: return "queuefamily";
    default: return "?";
    }
}

static uint32_t componentBits(VkComponentTypeKHR t) {
    switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR:
    case VK_COMPONENT_TYPE_SINT16_KHR:
    case VK_COMPONENT_TYPE_UINT16_KHR: return 16;
    case VK_COMPONENT_TYPE_FLOAT32_KHR:
    case VK_COMPONENT_TYPE_SINT32_KHR:
    case VK_COMPONENT_TYPE_UINT32_KHR: return 32;
    case VK_COMPONENT_TYPE_FLOAT64_KHR: return 64;
    case VK_COMPONENT_TYPE_SINT8_KHR:
    case VK_COMPONENT_TYPE_UINT8_KHR: return 8;
    default: return 0;
    }
}

// fp32 -> fp16 (round to nearest even, simplified) for input init / verification.
static uint16_t floatToHalf(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFFu;
    if (exp <= 0) return (uint16_t)sign; // flush small/denorm to zero (inputs are tiny ints/2)
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | (exp << 10) | (man >> 13));
}

static float halfToFloat(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) { out = sign; }
        else {
            exp = 127 - 15 + 1;
            while ((man & 0x400u) == 0) { man <<= 1; exp--; }
            man &= 0x3FFu;
            out = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &out, 4);
    return f;
}

// Type combination we know how to benchmark, paired with a precompiled shader.
struct TypeCombo {
    VkComponentTypeKHR inputType;
    VkComponentTypeKHR outputType;
    const char *spvFile;       // simple GEMM (streams A/B from global)
    const char *peakSpvFile;   // WMMA peak microbenchmark shader
    const char *shmemSpvFile;  // shared-memory tiled GEMM (WMMA)
    const char *fmaSpvFile;    // FMA/MAD GEMM (no WMMA)
    bool isFloat;
};

static const TypeCombo kCombos[] = {
    { VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, "gemm_fp16_fp32.spv", "wmma_peak_fp16_fp32.spv", "gemm_v07_fp16_fp32.spv", "fma_v06_fp16_fp32.spv", true },
    { VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR, "gemm_fp16_fp16.spv", "wmma_peak_fp16_fp16.spv", "gemm_v07_fp16_fp16.spv", "fma_v06_fp16_fp16.spv", true },
    { VK_COMPONENT_TYPE_SINT8_KHR,   VK_COMPONENT_TYPE_SINT32_KHR,  "gemm_s8_s32.spv",   "wmma_peak_s8_s32.spv",   "gemm_v07_s8_s32.spv",   "fma_v06_s8_s32.spv",   false },
    { VK_COMPONENT_TYPE_UINT8_KHR,   VK_COMPONENT_TYPE_UINT32_KHR,  "gemm_u8_u32.spv",   "wmma_peak_u8_u32.spv",   "gemm_v07_u8_u32.spv",   "fma_v06_u8_u32.spv",   false },
};

static int32_t findMemoryType(const VkPhysicalDeviceMemoryProperties &mp,
                              uint32_t bits, VkMemoryPropertyFlags req) {
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (mp.memoryTypes[i].propertyFlags & req) == req) {
            return (int32_t)i;
        }
    }
    return -1;
}

struct Buffer {
    VkBuffer host = VK_NULL_HANDLE, dev = VK_NULL_HANDLE;
    VkDeviceMemory hostMem = VK_NULL_HANDLE, devMem = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    void *ptr = nullptr;
};

// ---------------------------------------------------------------------------
// Dump a pipeline's executables via VK_KHR_pipeline_executable_properties:
// per-executable statistics (register/VGPR/SGPR usage, spills, etc.) and
// internal representations (the GPU ISA / disassembly text), when the driver
// chooses to expose them. Quietly degrades to whatever the driver provides.
// ---------------------------------------------------------------------------
// Print only the per-pipeline statistics (register usage, etc.), one line per
// stat, prefixed with `tag`. Lighter than dumpPipelineExecutables (no ISA text).
// ---------------------------------------------------------------------------
static void dumpPipelineStatsOnly(
    VkDevice device, VkPipeline pipeline, Report &rep, const char *tag,
    PFN_vkGetPipelineExecutablePropertiesKHR pProps,
    PFN_vkGetPipelineExecutableStatisticsKHR pStats) {
    if (!pProps || !pStats) return;
    VkPipelineInfoKHR pi = { VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR };
    pi.pipeline = pipeline;
    uint32_t numExec = 0;
    if (pProps(device, &pi, &numExec, nullptr) != VK_SUCCESS || numExec == 0) return;
    std::vector<VkPipelineExecutablePropertiesKHR> props(numExec);
    for (auto &p : props) { p = {}; p.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR; }
    pProps(device, &pi, &numExec, props.data());
    for (uint32_t e = 0; e < numExec; ++e) {
        VkPipelineExecutableInfoKHR ei = { VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR };
        ei.pipeline = pipeline; ei.executableIndex = e;
        uint32_t n = 0;
        pStats(device, &ei, &n, nullptr);
        if (!n) { rep.line("    %s: (no statistics exposed by driver)", tag); continue; }
        std::vector<VkPipelineExecutableStatisticKHR> st(n);
        for (auto &s : st) { s = {}; s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR; }
        pStats(device, &ei, &n, st.data());
        for (const auto &s : st) {
            switch (s.format) {
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                rep.line("    %s: %-30s = %s", tag, s.name, s.value.b32 ? "true" : "false"); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                rep.line("    %s: %-30s = %lld", tag, s.name, (long long)s.value.i64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                rep.line("    %s: %-30s = %llu", tag, s.name, (unsigned long long)s.value.u64); break;
            case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                rep.line("    %s: %-30s = %f", tag, s.name, s.value.f64); break;
            default: break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
static void dumpPipelineExecutables(
    VkDevice device, VkPipeline pipeline, Report &rep,
    PFN_vkGetPipelineExecutablePropertiesKHR pProps,
    PFN_vkGetPipelineExecutableStatisticsKHR pStats,
    PFN_vkGetPipelineExecutableInternalRepresentationsKHR pIR) {
    if (!pProps) return;

    VkPipelineInfoKHR pi = { VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR };
    pi.pipeline = pipeline;
    uint32_t numExec = 0;
    if (pProps(device, &pi, &numExec, nullptr) != VK_SUCCESS || numExec == 0) {
        rep.line("  [pipeline-exec] driver reported no executables");
        return;
    }
    std::vector<VkPipelineExecutablePropertiesKHR> props(numExec);
    for (auto &p : props) { p = {}; p.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR; }
    pProps(device, &pi, &numExec, props.data());

    for (uint32_t e = 0; e < numExec; ++e) {
        rep.line("  [exec %u] %s  (subgroupSize=%u)", e, props[e].name, props[e].subgroupSize);
        if (props[e].description[0]) rep.line("    desc: %s", props[e].description);

        VkPipelineExecutableInfoKHR ei = { VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR };
        ei.pipeline = pipeline;
        ei.executableIndex = e;

        // --- Statistics (register usage, occupancy, spills, ...) ---
        if (pStats) {
            uint32_t numStat = 0;
            pStats(device, &ei, &numStat, nullptr);
            if (numStat == 0) {
                rep.line("    (no statistics exposed)");
            } else {
                std::vector<VkPipelineExecutableStatisticKHR> stats(numStat);
                for (auto &s : stats) { s = {}; s.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR; }
                pStats(device, &ei, &numStat, stats.data());
                for (const auto &st : stats) {
                    switch (st.format) {
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                        rep.line("    stat: %-32s = %s", st.name, st.value.b32 ? "true" : "false"); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                        rep.line("    stat: %-32s = %lld", st.name, (long long)st.value.i64); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                        rep.line("    stat: %-32s = %llu", st.name, (unsigned long long)st.value.u64); break;
                    case VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR:
                        rep.line("    stat: %-32s = %f", st.name, st.value.f64); break;
                    default: break;
                    }
                }
            }
        }

        // --- Internal representations (the GPU ISA / disassembly) ---
        if (pIR) {
            uint32_t numIR = 0;
            pIR(device, &ei, &numIR, nullptr);
            if (numIR == 0) {
                rep.line("    (no internal representations exposed; driver hides ISA)");
                continue;
            }
            std::vector<VkPipelineExecutableInternalRepresentationKHR> irs(numIR);
            for (auto &r : irs) { r = {}; r.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INTERNAL_REPRESENTATION_KHR; }
            // First pass: driver fills metadata + required dataSize per IR.
            pIR(device, &ei, &numIR, irs.data());
            std::vector<std::vector<char>> bufs(numIR);
            for (uint32_t r = 0; r < numIR; ++r) {
                bufs[r].resize(irs[r].dataSize ? irs[r].dataSize : 1);
                irs[r].pData = bufs[r].data();
            }
            // Second pass: driver fills the actual data.
            pIR(device, &ei, &numIR, irs.data());
            for (uint32_t r = 0; r < numIR; ++r) {
                rep.line("    --- IR: %s (%s, %zu bytes) ---", irs[r].name,
                         irs[r].isText ? "text" : "binary", irs[r].dataSize);
                if (irs[r].description[0]) rep.line("    desc: %s", irs[r].description);
                if (irs[r].isText && irs[r].dataSize > 0) {
                    // Text IRs are null-terminated; print the whole disassembly.
                    rep.raw(std::string(bufs[r].data()));
                    rep.raw("\n");
                } else if (irs[r].dataSize > 0) {
                    rep.line("    (binary IR, %zu bytes, not printed)", irs[r].dataSize);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Print the GLSL shader source (bundled alongside the SPIR-V) into the report,
// so the report shows exactly what shader was compiled and run.
// ---------------------------------------------------------------------------
// GLSL type name for a Vulkan cooperative-matrix component type (for showing the
// shader source specialized to a data type).
static const char *glslTypeName(VkComponentTypeKHR t) {
    switch (t) {
    case VK_COMPONENT_TYPE_FLOAT16_KHR: return "float16_t";
    case VK_COMPONENT_TYPE_FLOAT32_KHR: return "float32_t";
    case VK_COMPONENT_TYPE_FLOAT64_KHR: return "float64_t";
    case VK_COMPONENT_TYPE_SINT8_KHR:   return "int8_t";
    case VK_COMPONENT_TYPE_SINT16_KHR:  return "int16_t";
    case VK_COMPONENT_TYPE_SINT32_KHR:  return "int32_t";
    case VK_COMPONENT_TYPE_UINT8_KHR:   return "uint8_t";
    case VK_COMPONENT_TYPE_UINT32_KHR:  return "uint32_t";
    default: return "?";
    }
}

// Print the bundled GLSL source with the A_TYPE / C_TYPE macros substituted for a
// concrete data type, so the report shows the exact shader code per type.
static void dumpTypedShaderSource(const std::string &shaderDir, const char *name,
                                  const char *aType, const char *cType, Report &rep) {
    std::string path = shaderDir + "/" + name;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    rep.line("");
    if (!f) {
        rep.line("  (shader source not bundled: %s)", path.c_str());
        return;
    }
    std::streamsize sz = f.tellg();
    f.seekg(0);
    std::string src((size_t)(sz > 0 ? sz : 0), '\0');
    if (sz > 0) f.read(&src[0], sz);
    auto replaceAll = [](std::string &s, const std::string &from, const std::string &to) {
        size_t p = 0;
        while ((p = s.find(from, p)) != std::string::npos) { s.replace(p, from.size(), to); p += to.size(); }
    };
    replaceAll(src, "A_TYPE", aType);
    replaceAll(src, "C_TYPE", cType);
    rep.line("  ==== shader %s (A_TYPE=%s, C_TYPE=%s) ====", name, aType, cType);
    rep.raw(src);
    if (src.empty() || src.back() != '\n') rep.raw("\n");
    rep.line("  ==== end shader ====");
}

// ---------------------------------------------------------------------------
// The actual benchmark.
// ---------------------------------------------------------------------------
static std::string runBenchmark(const std::string &shaderDir) {
    Report rep;
    rep.line("=== Exynos Cooperative-Matrix GEMM Benchmark ===");

    VkInstance instance = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    {
        uint32_t loaderVersion = VK_API_VERSION_1_0;
        auto pfnEnumVersion = (PFN_vkEnumerateInstanceVersion)
            vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion");
        if (pfnEnumVersion) {
            pfnEnumVersion(&loaderVersion);
        }
        rep.line("Vulkan loader instance version: %u.%u.%u",
                 VK_API_VERSION_MAJOR(loaderVersion),
                 VK_API_VERSION_MINOR(loaderVersion),
                 VK_API_VERSION_PATCH(loaderVersion));

        VkApplicationInfo app = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
        app.pApplicationName = "ExynosCoopMatPerf";
        app.apiVersion = VK_API_VERSION_1_3;

        VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        ici.pApplicationInfo = &app;
        VkResult r = vkCreateInstance(&ici, nullptr, &instance);
        if (r != VK_SUCCESS) {
            // Retry at 1.1 in case the loader rejects 1.3.
            app.apiVersion = VK_API_VERSION_1_1;
            r = vkCreateInstance(&ici, nullptr, &instance);
        }
        if (r != VK_SUCCESS) {
            rep.line("ERROR: vkCreateInstance failed (VkResult=%d)", (int)r);
            return rep.text;
        }
    }

    {
        uint32_t numPhys = 0;
        vkEnumeratePhysicalDevices(instance, &numPhys, nullptr);
        if (numPhys == 0) {
            rep.line("ERROR: no Vulkan physical devices found");
            vkDestroyInstance(instance, nullptr);
            return rep.text;
        }
        std::vector<VkPhysicalDevice> phys(numPhys);
        vkEnumeratePhysicalDevices(instance, &numPhys, phys.data());

        // Function pointers for extension/2 queries.
        auto pfnGetProps2 = (PFN_vkGetPhysicalDeviceProperties2)
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2");
        auto pfnGetFeats2 = (PFN_vkGetPhysicalDeviceFeatures2)
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2");
        auto pfnGetCoopProps = (PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR)
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR");

        // Pick a device that supports VK_KHR_cooperative_matrix.
        int chosen = -1;
        rep.line("Found %u physical device(s):", numPhys);
        for (uint32_t i = 0; i < numPhys; ++i) {
            VkPhysicalDeviceProperties pp;
            vkGetPhysicalDeviceProperties(phys[i], &pp);

            uint32_t numExt = 0;
            vkEnumerateDeviceExtensionProperties(phys[i], nullptr, &numExt, nullptr);
            std::vector<VkExtensionProperties> exts(numExt);
            vkEnumerateDeviceExtensionProperties(phys[i], nullptr, &numExt, exts.data());
            bool hasCoop = false;
            for (auto &e : exts) {
                if (strcmp(e.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0)
                    hasCoop = true;
            }
            rep.line("  [%u] %s  (API %u.%u.%u)  coopmat=%s",
                     i, pp.deviceName,
                     VK_API_VERSION_MAJOR(pp.apiVersion),
                     VK_API_VERSION_MINOR(pp.apiVersion),
                     VK_API_VERSION_PATCH(pp.apiVersion),
                     hasCoop ? "yes" : "no");
            if (hasCoop && chosen < 0) chosen = (int)i;
        }

        if (chosen < 0) {
            rep.line("");
            rep.line("No device advertises VK_KHR_cooperative_matrix.");
            rep.line("This GPU/driver cannot run the cooperative-matrix path.");
            vkDestroyInstance(instance, nullptr);
            return rep.text;
        }

        VkPhysicalDevice pd = phys[chosen];
        VkPhysicalDeviceProperties pdProps;
        vkGetPhysicalDeviceProperties(pd, &pdProps);
        rep.line("");
        rep.line("Using device [%d]: %s", chosen, pdProps.deviceName);

        // --- Subgroup info ---
        VkPhysicalDeviceSubgroupSizeControlProperties sgsc = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES };
        VkPhysicalDeviceSubgroupProperties sgProps = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES, &sgsc };
        VkPhysicalDeviceProperties2 p2 = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &sgProps };
        if (pfnGetProps2) pfnGetProps2(pd, &p2);
        rep.line("Subgroup size: %u (control range %u..%u)",
                 sgProps.subgroupSize, sgsc.minSubgroupSize, sgsc.maxSubgroupSize);

        // --- Detect VK_KHR_pipeline_executable_properties (for ISA dump) ---
        bool hasPipeExec = false;
        {
            uint32_t numExt = 0;
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &numExt, nullptr);
            std::vector<VkExtensionProperties> exts(numExt);
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &numExt, exts.data());
            for (auto &e : exts)
                if (strcmp(e.extensionName, VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME) == 0)
                    hasPipeExec = true;
        }
        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fPipeExec = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR };
        if (hasPipeExec && pfnGetFeats2) {
            VkPhysicalDeviceFeatures2 q = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &fPipeExec };
            pfnGetFeats2(pd, &q);
        }
        bool wantPipeExec = hasPipeExec && fPipeExec.pipelineExecutableInfo;
        rep.line("pipeline_executable_properties: ext=%s feature=%u -> ISA dump %s",
                 hasPipeExec ? "yes" : "no", fPipeExec.pipelineExecutableInfo,
                 wantPipeExec ? "ENABLED" : "unavailable");

        // --- Dump cooperative matrix shapes ---
        std::vector<VkCooperativeMatrixPropertiesKHR> coop;
        if (pfnGetCoopProps) {
            uint32_t n = 0;
            pfnGetCoopProps(pd, &n, nullptr);
            coop.resize(n);
            for (auto &c : coop) { c.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR; c.pNext = nullptr; }
            pfnGetCoopProps(pd, &n, coop.data());
        }
        rep.line("");
        rep.line("Advertised cooperative-matrix configurations: %zu", coop.size());
        for (size_t i = 0; i < coop.size(); ++i) {
            const auto &c = coop[i];
            rep.line("  MxNxK=%ux%ux%u  A=%s B=%s C=%s D=%s  sat=%u  scope=%s",
                     c.MSize, c.NSize, c.KSize,
                     componentTypeName(c.AType), componentTypeName(c.BType),
                     componentTypeName(c.CType), componentTypeName(c.ResultType),
                     c.saturatingAccumulation, scopeName(c.scope));
        }

        // --- Query feature support we depend on ---
        VkPhysicalDeviceCooperativeMatrixFeaturesKHR fCoop = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
        VkPhysicalDeviceVulkanMemoryModelFeatures fVMM = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES, &fCoop };
        VkPhysicalDeviceShaderFloat16Int8Features fF16 = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES, &fVMM };
        VkPhysicalDevice16BitStorageFeatures f16Store = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES, &fF16 };
        VkPhysicalDevice8BitStorageFeatures f8Store = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES, &f16Store };
        VkPhysicalDeviceSubgroupSizeControlFeatures fSgsc = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES, &f8Store };
        VkPhysicalDeviceShaderIntegerDotProductFeatures fIDot = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES, &fSgsc };
        VkPhysicalDeviceFeatures2 feats2 = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &fIDot };
        if (pfnGetFeats2) pfnGetFeats2(pd, &feats2);
        bool hasIntDot = fIDot.shaderIntegerDotProduct;

        rep.line("");
        rep.line("Required feature support:");
        rep.line("  cooperativeMatrix        = %u", fCoop.cooperativeMatrix);
        rep.line("  vulkanMemoryModel        = %u", fVMM.vulkanMemoryModel);
        rep.line("  shaderFloat16            = %u", fF16.shaderFloat16);
        rep.line("  shaderInt8               = %u", fF16.shaderInt8);
        rep.line("  storageBuffer16BitAccess = %u", f16Store.storageBuffer16BitAccess);
        rep.line("  storageBuffer8BitAccess  = %u", f8Store.storageBuffer8BitAccess);
        rep.line("  subgroupSizeControl      = %u", fSgsc.subgroupSizeControl);
        rep.line("  computeFullSubgroups     = %u", fSgsc.computeFullSubgroups);

        bool canFloat = fCoop.cooperativeMatrix && fVMM.vulkanMemoryModel &&
                        fF16.shaderFloat16 && f16Store.storageBuffer16BitAccess &&
                        fSgsc.subgroupSizeControl && fSgsc.computeFullSubgroups;
        bool canInt   = fCoop.cooperativeMatrix && fVMM.vulkanMemoryModel &&
                        fF16.shaderInt8 && f8Store.storageBuffer8BitAccess &&
                        fSgsc.subgroupSizeControl && fSgsc.computeFullSubgroups;

        if (!canFloat && !canInt) {
            rep.line("");
            rep.line("Required features for the GEMM path are not all present; "
                     "reporting capabilities only.");
            vkDestroyInstance(instance, nullptr);
            return rep.text;
        }

        // Compute queue family.
        uint32_t numQF = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &numQF, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(numQF);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &numQF, qfs.data());
        int qfi = -1;
        for (uint32_t i = 0; i < numQF; ++i)
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = (int)i; break; }
        if (qfi < 0) {
            rep.line("ERROR: no compute queue family");
            vkDestroyInstance(instance, nullptr);
            return rep.text;
        }

        // --- Create device, enabling only supported features ---
        // Re-zero feature structs and turn on what we need + is supported.
        fCoop = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR };
        fVMM = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_MEMORY_MODEL_FEATURES };
        fF16 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES };
        f16Store = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES };
        f8Store = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES };
        fSgsc = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES };

        fCoop.cooperativeMatrix = VK_TRUE;
        fVMM.vulkanMemoryModel = VK_TRUE;
        fF16.shaderFloat16 = canFloat ? VK_TRUE : VK_FALSE;
        fF16.shaderInt8 = canInt ? VK_TRUE : VK_FALSE;
        f16Store.storageBuffer16BitAccess = canFloat ? VK_TRUE : VK_FALSE;
        f8Store.storageBuffer8BitAccess = canInt ? VK_TRUE : VK_FALSE;
        fSgsc.subgroupSizeControl = VK_TRUE;
        fSgsc.computeFullSubgroups = VK_TRUE;

        fCoop.pNext = &fVMM;
        fVMM.pNext = &fF16;
        fF16.pNext = &f16Store;
        f16Store.pNext = &f8Store;
        f8Store.pNext = &fSgsc;

        std::vector<const char *> devExts = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };

        // Enable pipeline-executable info so we can dump the compiled GPU ISA.
        VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR fPipeExecEn = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_EXECUTABLE_PROPERTIES_FEATURES_KHR };
        if (wantPipeExec) {
            fPipeExecEn.pipelineExecutableInfo = VK_TRUE;
            fSgsc.pNext = &fPipeExecEn;
            devExts.push_back(VK_KHR_PIPELINE_EXECUTABLE_PROPERTIES_EXTENSION_NAME);
        }

        // Enable integer dot product (dp4a) for the optimized int8 FMA GEMM.
        VkPhysicalDeviceShaderIntegerDotProductFeatures fIDotEn = {
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES };
        if (hasIntDot) {
            fIDotEn.shaderIntegerDotProduct = VK_TRUE;
            if (wantPipeExec) fPipeExecEn.pNext = &fIDotEn;
            else fSgsc.pNext = &fIDotEn;
            bool hasIDotExt = false;
            uint32_t ne = 0;
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, nullptr);
            std::vector<VkExtensionProperties> ex(ne);
            vkEnumerateDeviceExtensionProperties(pd, nullptr, &ne, ex.data());
            for (auto &e : ex)
                if (strcmp(e.extensionName, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) == 0) hasIDotExt = true;
            if (hasIDotExt) devExts.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
        }
        rep.line("  shaderIntegerDotProduct (dp4a) = %u", hasIntDot);

        float prio = 1.0f;
        VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qci.queueFamilyIndex = (uint32_t)qfi;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dci.pNext = &fCoop;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = (uint32_t)devExts.size();
        dci.ppEnabledExtensionNames = devExts.data();

        VkResult dr = vkCreateDevice(pd, &dci, nullptr, &device);
        if (dr != VK_SUCCESS) {
            rep.line("ERROR: vkCreateDevice failed (VkResult=%d)", (int)dr);
            vkDestroyInstance(instance, nullptr);
            return rep.text;
        }

        VkQueue queue;
        vkGetDeviceQueue(device, (uint32_t)qfi, 0, &queue);

        // Resolve pipeline-executable entry points (only if we enabled them).
        PFN_vkGetPipelineExecutablePropertiesKHR pfnPeProps = nullptr;
        PFN_vkGetPipelineExecutableStatisticsKHR pfnPeStats = nullptr;
        PFN_vkGetPipelineExecutableInternalRepresentationsKHR pfnPeIR = nullptr;
        if (wantPipeExec) {
            pfnPeProps = (PFN_vkGetPipelineExecutablePropertiesKHR)
                vkGetDeviceProcAddr(device, "vkGetPipelineExecutablePropertiesKHR");
            pfnPeStats = (PFN_vkGetPipelineExecutableStatisticsKHR)
                vkGetDeviceProcAddr(device, "vkGetPipelineExecutableStatisticsKHR");
            pfnPeIR = (PFN_vkGetPipelineExecutableInternalRepresentationsKHR)
                vkGetDeviceProcAddr(device, "vkGetPipelineExecutableInternalRepresentationsKHR");
        }

        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(pd, &memProps);

        // Required subgroup size: clamp the device subgroup size into the
        // controllable range (powers of two only).
        uint32_t reqSg = sgProps.subgroupSize ? sgProps.subgroupSize : 32;
        if (sgsc.minSubgroupSize && reqSg < sgsc.minSubgroupSize) reqSg = sgsc.minSubgroupSize;
        if (sgsc.maxSubgroupSize && reqSg > sgsc.maxSubgroupSize) reqSg = sgsc.maxSubgroupSize;

        // Choose the type combination to benchmark: prefer fp16->fp32.
        // For each, find a matching subgroup-scope advertised shape.
        struct Plan {
            const TypeCombo *combo;
            VkCooperativeMatrixPropertiesKHR shape;
            bool found;
        };
        std::vector<Plan> plans;
        for (const auto &combo : kCombos) {
            if (combo.isFloat && !canFloat) continue;
            if (!combo.isFloat && !canInt) continue;
            Plan pl{ &combo, {}, false };
            for (const auto &c : coop) {
                if (c.scope == VK_SCOPE_SUBGROUP_KHR &&
                    c.AType == combo.inputType && c.BType == combo.inputType &&
                    c.CType == combo.outputType && c.ResultType == combo.outputType) {
                    pl.shape = c; pl.found = true; break;
                }
            }
            if (pl.found) plans.push_back(pl);
        }

        if (plans.empty()) {
            rep.line("");
            rep.line("No subgroup-scope shape matches a shader we ship; "
                     "reporting capabilities only.");
            vkDestroyDevice(device, nullptr);
            vkDestroyInstance(instance, nullptr);
            return rep.text;
        }

        // Common Vulkan objects reused across runs.
        VkDescriptorSetLayoutBinding binds[4];
        for (int i = 0; i < 4; ++i) {
            binds[i] = {};
            binds[i].binding = (uint32_t)i;
            binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            binds[i].descriptorCount = 1;
            binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dslci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        dslci.bindingCount = 4;
        dslci.pBindings = binds;
        VkDescriptorSetLayout dsl;
        vkCreateDescriptorSetLayout(device, &dslci, nullptr, &dsl);

        VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsl;
        VkPipelineLayout pipelineLayout;
        vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout);

        VkDescriptorPoolSize psz = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 };
        VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &psz;
        VkDescriptorPool descPool;
        vkCreateDescriptorPool(device, &dpci, nullptr, &descPool);

        VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        dsai.descriptorPool = descPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsl;
        VkDescriptorSet descSet;
        vkAllocateDescriptorSets(device, &dsai, &descSet);

        VkCommandPoolCreateInfo cpci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = (uint32_t)qfi;
        VkCommandPool cmdPool;
        vkCreateCommandPool(device, &cpci, nullptr, &cmdPool);

        VkCommandBufferAllocateInfo cbai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        cbai.commandPool = cmdPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device, &cbai, &cmd);

#if 0   // ===== GEMM dimension-sweep benchmark: DISABLED (WMMA peak test runs instead) =====
        // -------- run each plan --------
        for (const auto &pl : plans) {
            const TypeCombo &combo = *pl.combo;
            const VkCooperativeMatrixPropertiesKHR &shape = pl.shape;
            uint32_t lM = shape.MSize, lN = shape.NSize, lK = shape.KSize;

            rep.line("");
            rep.line("---- %s*%s -> %s   shape %ux%ux%u ----",
                     componentTypeName(combo.inputType), componentTypeName(combo.inputType),
                     componentTypeName(combo.outputType), lM, lN, lK);

            // Dump the compiled GPU ISA once per type (first pipeline built).
            bool dumpedExec = false;

            // Load SPIR-V.
            std::string path = shaderDir + "/" + combo.spvFile;
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) { rep.line("SKIP: shader not found: %s", path.c_str()); continue; }
            std::streamsize sz = f.tellg();
            f.seekg(0);
            std::vector<char> spv(sz);
            f.read(spv.data(), sz);

            VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            smci.codeSize = (size_t)sz;
            smci.pCode = (const uint32_t *)spv.data();
            VkShaderModule module;
            if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS) {
                rep.line("SKIP: failed to create shader module for %s", combo.spvFile);
                continue;
            }

            uint32_t inBits = componentBits(combo.inputType);
            uint32_t outBits = componentBits(combo.outputType);
            size_t inElem = inBits / 8, outElem = outBits / 8;

            // Sweep a few workgroup/tile configurations.
            struct Cfg { uint32_t targetInvocations; uint32_t cRows; uint32_t cCols; };
            const Cfg cfgs[] = {
                { 128, 2, 2 },
                { 256, 2, 2 },
                { 256, 4, 2 },
            };

            for (const auto &cfg : cfgs) {
                uint32_t numSg = std::max(1u, cfg.targetInvocations / reqSg);
                // Arrange subgroups as WG_W x WG_H (WG_W >= WG_H).
                uint32_t wgW = numSg, wgH = 1;
                for (uint32_t h = (uint32_t)std::floor(std::sqrt((double)numSg)); h >= 1; --h) {
                    if (numSg % h == 0) { wgH = h; wgW = numSg / h; break; }
                }
                uint32_t cRows = cfg.cRows, cCols = cfg.cCols;
                uint32_t tileM = wgH * cRows * lM;
                uint32_t tileN = wgW * cCols * lN;
                uint32_t localX = wgW * wgH * reqSg;

                // Sweep the problem size: 256, 512, then 1024..8192 in steps of
                // 1024 (each dim rounded up to tile / lK multiples).
                auto roundUp = [](uint32_t v, uint32_t a) { return (v + a - 1) / a * a; };
                std::vector<uint32_t> dims = { 256, 512 };
                for (uint32_t d = 1024; d <= 8192; d += 1024) dims.push_back(d);
                for (uint32_t dim : dims) {
                uint32_t M = roundUp(dim, tileM);
                uint32_t N = roundUp(dim, tileN);
                uint32_t K = roundUp(dim, lK);

                // Allocate four buffers (host staging + device local).
                auto makeBuf = [&](Buffer &b, VkDeviceSize bytes) -> bool {
                    b.size = bytes;
                    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                    bci.size = bytes;
                    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                    if (vkCreateBuffer(device, &bci, nullptr, &b.host) != VK_SUCCESS) return false;
                    if (vkCreateBuffer(device, &bci, nullptr, &b.dev) != VK_SUCCESS) return false;
                    VkMemoryRequirements mr;
                    vkGetBufferMemoryRequirements(device, b.host, &mr);
                    int32_t hi = findMemoryType(memProps, mr.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                    int32_t di = findMemoryType(memProps, mr.memoryTypeBits,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                    if (di < 0) di = hi;
                    if (hi < 0) return false;
                    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                    mai.allocationSize = mr.size;
                    mai.memoryTypeIndex = (uint32_t)hi;
                    if (vkAllocateMemory(device, &mai, nullptr, &b.hostMem) != VK_SUCCESS) return false;
                    mai.memoryTypeIndex = (uint32_t)di;
                    if (vkAllocateMemory(device, &mai, nullptr, &b.devMem) != VK_SUCCESS) return false;
                    vkBindBufferMemory(device, b.host, b.hostMem, 0);
                    vkBindBufferMemory(device, b.dev, b.devMem, 0);
                    vkMapMemory(device, b.hostMem, 0, bytes, 0, &b.ptr);
                    return true;
                };
                auto freeBuf = [&](Buffer &b) {
                    if (b.host) vkDestroyBuffer(device, b.host, nullptr);
                    if (b.dev) vkDestroyBuffer(device, b.dev, nullptr);
                    if (b.hostMem) vkFreeMemory(device, b.hostMem, nullptr);
                    if (b.devMem) vkFreeMemory(device, b.devMem, nullptr);
                    b = Buffer{};
                };

                Buffer bufA, bufB, bufC, bufD;
                bool ok = makeBuf(bufA, (VkDeviceSize)M * K * inElem) &&
                          makeBuf(bufB, (VkDeviceSize)K * N * inElem) &&
                          makeBuf(bufC, (VkDeviceSize)M * N * outElem) &&
                          makeBuf(bufD, (VkDeviceSize)M * N * outElem);
                if (!ok) {
                    rep.line("  cfg(inv=%u rows=%u cols=%u): buffer alloc failed (skip)",
                             cfg.targetInvocations, cRows, cCols);
                    freeBuf(bufA); freeBuf(bufB); freeBuf(bufC); freeBuf(bufD);
                    continue;
                }

                // Initialize inputs to small values (host buffers).
                auto fillInput = [&](Buffer &b, uint32_t count) {
                    for (uint32_t i = 0; i < count; ++i) {
                        float v = (float)((i * 1664525u + 1013904223u) % 5) - 2.0f; // -2..2
                        if (combo.isFloat) {
                            if (inBits == 16) ((uint16_t *)b.ptr)[i] = floatToHalf(v * 0.5f);
                        } else {
                            ((int8_t *)b.ptr)[i] = (int8_t)((int)v);
                        }
                    }
                };
                fillInput(bufA, M * K);
                fillInput(bufB, K * N);
                // C = 0, D = sentinel.
                memset(bufC.ptr, 0, (size_t)bufC.size);
                memset(bufD.ptr, 0, (size_t)bufD.size);

                // Upload host -> device.
                VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                vkBeginCommandBuffer(cmd, &bi);
                for (Buffer *b : { &bufA, &bufB, &bufC, &bufD }) {
                    VkBufferCopy cp = { 0, 0, b->size };
                    vkCmdCopyBuffer(cmd, b->host, b->dev, 1, &cp);
                }
                vkEndCommandBuffer(cmd);
                VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
                si.commandBufferCount = 1;
                si.pCommandBuffers = &cmd;
                vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
                vkQueueWaitIdle(queue);

                // Update descriptors to point at device buffers.
                VkDescriptorBufferInfo dbi[4] = {
                    { bufA.dev, 0, VK_WHOLE_SIZE },
                    { bufB.dev, 0, VK_WHOLE_SIZE },
                    { bufC.dev, 0, VK_WHOLE_SIZE },
                    { bufD.dev, 0, VK_WHOLE_SIZE },
                };
                VkWriteDescriptorSet wds[4];
                for (int i = 0; i < 4; ++i) {
                    wds[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                    wds[i].dstSet = descSet;
                    wds[i].dstBinding = (uint32_t)i;
                    wds[i].descriptorCount = 1;
                    wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wds[i].pBufferInfo = &dbi[i];
                }
                vkUpdateDescriptorSets(device, 4, wds, 0, nullptr);

                // Specialization constants (must match shader constant_ids).
                float alpha = 1.0f, beta = 0.0f;
                uint32_t spec[13];
                spec[0] = lM; spec[1] = lN; spec[2] = lK;
                spec[3] = M;  spec[4] = N;  spec[5] = K;
                spec[6] = cRows; spec[7] = cCols;
                spec[8] = wgW; spec[9] = wgH;
                memcpy(&spec[10], &alpha, 4);
                memcpy(&spec[11], &beta, 4);
                spec[12] = localX;
                VkSpecializationMapEntry ents[13];
                for (uint32_t i = 0; i < 13; ++i) ents[i] = { i, (uint32_t)(i * 4), 4 };
                VkSpecializationInfo spi = { 13, ents, sizeof(spec), spec };

                VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO };
                rss.requiredSubgroupSize = reqSg;

                VkPipelineShaderStageCreateInfo ssci = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
                ssci.pNext = &rss;
                ssci.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
                ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                ssci.module = module;
                ssci.pName = "main";
                ssci.pSpecializationInfo = &spi;

                VkComputePipelineCreateInfo cpci2 = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
                cpci2.stage = ssci;
                cpci2.layout = pipelineLayout;
                // Ask the driver to keep statistics + ISA for this pipeline.
                if (wantPipeExec && !dumpedExec)
                    cpci2.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                                   VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
                VkPipeline pipeline;
                VkResult pr = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci2, nullptr, &pipeline);
                if (pr != VK_SUCCESS) {
                    rep.line("  cfg(inv=%u rows=%u cols=%u tile=%ux%u): pipeline create failed (VkResult=%d)",
                             localX, cRows, cCols, tileM, tileN, (int)pr);
                    freeBuf(bufA); freeBuf(bufB); freeBuf(bufC); freeBuf(bufD);
                    continue;
                }

                // Dump the compiled GPU ISA/statistics once per type combination.
                if (wantPipeExec && !dumpedExec) {
                    rep.line("  -- compiled GPU executable (ISA / statistics) --");
                    dumpPipelineExecutables(device, pipeline, rep, pfnPeProps, pfnPeStats, pfnPeIR);
                    dumpedExec = true;
                }

                uint32_t gx = N / tileN, gy = M / tileM;
                si.pCommandBuffers = &cmd;

                // A single long submission trips the GPU watchdog
                // (VK_ERROR_DEVICE_LOST, VkResult=-4): when one submit runs too
                // long the driver kills the device and everything after fails.
                // So we (1) time a single dispatch, (2) skip the dim if even one
                // dispatch is already over the cutoff, and (3) run the requested
                // iteration count in small batches that each stay well under the
                // watchdog.
                VkResult waitRes = VK_SUCCESS;
                auto recordDispatches = [&](uint32_t n) {
                    vkBeginCommandBuffer(cmd, &bi);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                    // No barrier between dispatches: A/B are unchanged and D is not
                    // verified, so iterations overlap (steady-state throughput).
                    for (uint32_t i = 0; i < n; ++i) vkCmdDispatch(cmd, gx, gy, 1);
                    vkEndCommandBuffer(cmd);
                };
                auto timedSubmit = [&]() -> double {
                    auto a = std::chrono::high_resolution_clock::now();
                    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
                    waitRes = vkQueueWaitIdle(queue);
                    auto b = std::chrono::high_resolution_clock::now();
                    return std::chrono::duration<double>(b - a).count();
                };
                auto cleanup = [&]() {
                    vkDestroyPipeline(device, pipeline, nullptr);
                    freeBuf(bufA); freeBuf(bufB); freeBuf(bufC); freeBuf(bufD);
                };

                const char *opUnit   = combo.isFloat ? "GFLOP"  : "GOP";
                const char *rateUnit = combo.isFloat ? "TFLOPS" : "TOPS";
                auto reportRun = [&](double sec, uint32_t reps) {
                    double ops = 2.0 * (double)M * (double)N * (double)K * (double)reps;
                    // Modeled buffer_load traffic: with no shared-memory tiling, A
                    // is re-read per output column-block, B per row-block, C once.
                    double aLoadElems = (double)M * (double)N * (double)K / ((double)cCols * (double)lN);
                    double bLoadElems = (double)M * (double)N * (double)K / ((double)cRows * (double)lM);
                    double cLoadElems = (double)M * (double)N;
                    double loadBytes = ((aLoadElems + bLoadElems) * (double)inElem +
                                        cLoadElems * (double)outElem) * (double)reps;
                    rep.line("  dim=%ux%ux%u tile=%ux%u (sg=%ux%u x%u, inv=%u, %u iters):",
                             M, N, K, tileM, tileN, wgW, wgH, reqSg, localX, reps);
                    rep.line("      time    = %.3f ms (%.4f ms/iter)", sec * 1e3, sec * 1e3 / reps);
                    rep.line("      compute = %.1f %s", ops / 1e9, opUnit);
                    rep.line("      load    = %.2f GB (buffer_load traffic, modeled)", loadBytes / 1e9);
                    rep.line("      => %.3f %s, %.1f GB/s",
                             ops / sec / 1e12, rateUnit, loadBytes / sec / 1e9);
                };

                // (1) Calibrate with a single dispatch (first call also warms up).
                recordDispatches(1);
                timedSubmit();
                double oneSec = timedSubmit();
                double oneMs = oneSec * 1e3;
                if (waitRes != VK_SUCCESS) {
                    rep.line("  dim=%ux%ux%u tile=%ux%u: device-lost on a single dispatch "
                             "(VkResult=%d) -> skipping this and larger dims",
                             M, N, K, tileM, tileN, (int)waitRes);
                    cleanup();
                    break;
                }

                // (2) One dispatch already over the cutoff: report it and stop.
                if (oneMs > 250.0) {
                    reportRun(oneSec, 1);
                    rep.line("      (ms/iter %.1f exceeds 250 ms -> skipping larger dims for this config)", oneMs);
                    cleanup();
                    break;
                }

                // (3) Run the requested iterations in batches that each stay
                // ~<=200 ms (well under the watchdog), up to a ~3 s wall budget.
                uint32_t targetIters = (dim <= 2048) ? 1000 : 100;
                uint32_t batch = (oneMs > 0.001) ? (uint32_t)(200.0 / oneMs) : targetIters;
                if (batch < 1) batch = 1;
                if (batch > targetIters) batch = targetIters;
                recordDispatches(batch);
                timedSubmit();   // warm up the batched command buffer

                double totalSec = 0.0;
                uint32_t iters = 0;
                while (iters < targetIters && totalSec < 3.0) {
                    double s = timedSubmit();
                    if (waitRes != VK_SUCCESS) break;
                    totalSec += s;
                    iters += batch;
                }
                if (waitRes != VK_SUCCESS) {
                    rep.line("  dim=%ux%ux%u tile=%ux%u: device-lost mid-measurement "
                             "(VkResult=%d) -> skipping larger dims", M, N, K, tileM, tileN, (int)waitRes);
                    cleanup();
                    break;
                }

                reportRun(totalSec, iters ? iters : 1);
                cleanup();
                } // dim sweep
            } // cfg

            vkDestroyShaderModule(device, module, nullptr);
        } // plans
#endif  // ===== end disabled GEMM dimension-sweep benchmark =====

        // ======================= WMMA peak-throughput test =======================
        // Goal: the matrix engine's peak compute, isolated from DRAM. Every
        // subgroup loads one shared 16x16 A/B tile (cache/register resident) and
        // repeats NACC cross-fed MulAdds REPEATS times, for REPEATS in
        // {1024, 4096, 8196}. Data is generated per operation type.
#if 0   // ===== WMMA peak test DISABLED (only the optimized FMA GEMM runs) =====
        {
            rep.line("");
            rep.line("==== WMMA peak throughput (16x16 tile loaded once, repeated MulAdd) ====");

            const uint32_t NACC = 4;
            const uint32_t SG_PER_WG = 4;
            const uint32_t repeatCounts[] = { 1024, 2048 };
            const uint32_t BUF_TILES = 16384;   // 16x16 tiles available for reloads
            uint32_t localX = reqSg * SG_PER_WG;
            // A submission longer than the GPU watchdog makes vkQueueWaitIdle
            // block forever (~38 ms ran safely; 4x hung). Keep each dispatch near
            // that budget, and divide it down further as reuse drops, since the
            // extra memory loads make each iteration slower.
            const uint64_t WORK_BUDGET = (uint64_t)8192 * 1024;
            const uint32_t MAX_SUBGROUPS = 8192;   // D is sized for this
            rep.line("config: subgroups/wg=%u, accumulators=%u, subgroupSize=%u, reload buffer=%u tiles",
                     SG_PER_WG, NACC, reqSg, BUF_TILES);
            rep.line("note: reuse%% = 100 - reloads*10. 100%% = pure compute (no DRAM),");
            rep.line("      0%% = every MulAdd reloads a fresh tile. TFLOPS/TOPS count a");
            rep.line("      multiply-add as 2 ops; TMAC/s (vendor-style) is half.");

            auto makeBufP = [&](Buffer &b, VkDeviceSize bytes) -> bool {
                b.size = bytes;
                VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.size = bytes;
                bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (vkCreateBuffer(device, &bci, nullptr, &b.dev) != VK_SUCCESS) return false;
                VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, b.dev, &mr);
                int32_t mi = findMemoryType(memProps, mr.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if (mi < 0) return false;
                VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                mai.allocationSize = mr.size; mai.memoryTypeIndex = (uint32_t)mi;
                if (vkAllocateMemory(device, &mai, nullptr, &b.devMem) != VK_SUCCESS) return false;
                vkBindBufferMemory(device, b.dev, b.devMem, 0);
                vkMapMemory(device, b.devMem, 0, bytes, 0, &b.ptr);
                return true;
            };
            auto freeBufP = [&](Buffer &b) {
                if (b.dev) vkDestroyBuffer(device, b.dev, nullptr);
                if (b.devMem) vkFreeMemory(device, b.devMem, nullptr);
                b = Buffer{};
            };

            for (const auto &combo : kCombos) {
                if (combo.isFloat && !canFloat) continue;
                if (!combo.isFloat && !canInt) continue;

                bool have16 = false;
                for (const auto &c : coop)
                    if (c.scope == VK_SCOPE_SUBGROUP_KHR &&
                        c.MSize == 16 && c.NSize == 16 && c.KSize == 16 &&
                        c.AType == combo.inputType && c.BType == combo.inputType &&
                        c.CType == combo.outputType && c.ResultType == combo.outputType) {
                        have16 = true; break;
                    }

                rep.line("");
                rep.line("---- %s*%s -> %s [WMMA peak, 16x16x16] ----",
                         componentTypeName(combo.inputType), componentTypeName(combo.inputType),
                         componentTypeName(combo.outputType));
                if (!have16) { rep.line("  no 16x16x16 subgroup shape advertised; skip"); continue; }

                // Show the exact shader source for this data type.
                dumpTypedShaderSource(shaderDir, "wmma_peak.comp",
                                      glslTypeName(combo.inputType),
                                      glslTypeName(combo.outputType), rep);

                std::string path = shaderDir + "/" + combo.peakSpvFile;
                std::ifstream f(path, std::ios::binary | std::ios::ate);
                if (!f) { rep.line("  SKIP: shader not found: %s", path.c_str()); continue; }
                std::streamsize sz = f.tellg(); f.seekg(0);
                std::vector<char> spv(sz); f.read(spv.data(), sz);
                VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                smci.codeSize = (size_t)sz; smci.pCode = (const uint32_t *)spv.data();
                VkShaderModule module;
                if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS) {
                    rep.line("  SKIP: shader module create failed"); continue;
                }

                size_t inElem = componentBits(combo.inputType) / 8;
                size_t outElem = componentBits(combo.outputType) / 8;
                const VkDeviceSize tileElems = 16 * 16;
                const VkDeviceSize abElems = (VkDeviceSize)BUF_TILES * tileElems;

                Buffer bufA, bufB, bufC, bufD;
                bool ok = makeBufP(bufA, abElems * inElem) &&
                          makeBufP(bufB, abElems * inElem) &&
                          makeBufP(bufC, tileElems * outElem) &&
                          makeBufP(bufD, (VkDeviceSize)MAX_SUBGROUPS * tileElems * outElem);
                if (!ok) {
                    rep.line("  buffer alloc failed; skip");
                    freeBufP(bufA); freeBufP(bufB); freeBufP(bufC); freeBufP(bufD);
                    vkDestroyShaderModule(device, module, nullptr); continue;
                }

                // Generate per-type input data (all reload tiles). Small magnitudes
                // so fp16 does not overflow; the value is irrelevant to throughput.
                for (VkDeviceSize i = 0; i < abElems; ++i) {
                    if (combo.isFloat) {
                        ((uint16_t *)bufA.ptr)[i] = floatToHalf(0.01f);
                        ((uint16_t *)bufB.ptr)[i] = floatToHalf(0.01f);
                    } else {
                        ((int8_t *)bufA.ptr)[i] = 1;
                        ((int8_t *)bufB.ptr)[i] = 1;
                    }
                }
                memset(bufC.ptr, 0, (size_t)bufC.size);

                VkDescriptorBufferInfo dbi[4] = {
                    { bufA.dev, 0, VK_WHOLE_SIZE }, { bufB.dev, 0, VK_WHOLE_SIZE },
                    { bufC.dev, 0, VK_WHOLE_SIZE }, { bufD.dev, 0, VK_WHOLE_SIZE } };
                VkWriteDescriptorSet wds[4];
                for (int i = 0; i < 4; ++i) {
                    wds[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                    wds[i].dstSet = descSet; wds[i].dstBinding = (uint32_t)i;
                    wds[i].descriptorCount = 1;
                    wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    wds[i].pBufferInfo = &dbi[i];
                }
                vkUpdateDescriptorSets(device, 4, wds, 0, nullptr);

                const char *opUnit   = combo.isFloat ? "GFLOP"  : "GOP";
                const char *rateUnit = combo.isFloat ? "TFLOPS" : "TOPS";
                bool dumpedExec = false;   // dump GPU ISA once per type

                for (uint32_t repeats : repeatCounts) {
                    rep.line("  repeats=%u  --  data-reuse sweep (boost-warmed, best of 3):", repeats);

                    // Sweep reuse 100% -> 0% in 10% steps.
                    for (int reuse = 100; reuse >= 0; reuse -= 10) {
                        uint32_t reloadsPer10 = (uint32_t)((100 - reuse) / 10);

                        // Scale subgroups so the dispatch stays short; divide the
                        // budget down as reuse drops (memory loads slow each iter).
                        uint64_t budget = WORK_BUDGET / (1u + reloadsPer10);
                        uint32_t workgroups = (uint32_t)(budget / ((uint64_t)repeats * SG_PER_WG));
                        if (workgroups < 16) workgroups = 16;
                        if ((uint64_t)workgroups * SG_PER_WG > MAX_SUBGROUPS)
                            workgroups = MAX_SUBGROUPS / SG_PER_WG;
                        uint64_t totalSubgroups = (uint64_t)workgroups * SG_PER_WG;

                        uint32_t spec[8] = { 16, 16, 16, repeats, NACC, localX, reloadsPer10, BUF_TILES };
                        VkSpecializationMapEntry ents[8];
                        for (uint32_t i = 0; i < 8; ++i) ents[i] = { i, (uint32_t)(i * 4), 4 };
                        VkSpecializationInfo spi = { 8, ents, sizeof(spec), spec };

                        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {
                            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO };
                        rss.requiredSubgroupSize = reqSg;
                        VkPipelineShaderStageCreateInfo ssci = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
                        ssci.pNext = &rss;
                        ssci.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
                        ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                        ssci.module = module; ssci.pName = "main"; ssci.pSpecializationInfo = &spi;
                        VkComputePipelineCreateInfo cpci2 = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
                        cpci2.stage = ssci; cpci2.layout = pipelineLayout;
                        // Capture the compiled GPU executable (ISA/sp3) once per type.
                        if (wantPipeExec && !dumpedExec)
                            cpci2.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                                           VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
                        VkPipeline pipeline;
                        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci2, nullptr, &pipeline) != VK_SUCCESS) {
                            rep.line("    reuse=%3d%%: pipeline create failed", reuse); continue;
                        }
                        if (wantPipeExec && !dumpedExec) {
                            rep.line("  -- compiled GPU executable (ISA / sp3, statistics) --");
                            dumpPipelineExecutables(device, pipeline, rep, pfnPeProps, pfnPeStats, pfnPeIR);
                            dumpedExec = true;
                        }

                        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                        vkBeginCommandBuffer(cmd, &bi);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                        vkCmdDispatch(cmd, workgroups, 1, 1);
                        vkEndCommandBuffer(cmd);
                        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
                        si.commandBufferCount = 1; si.pCommandBuffers = &cmd;

                        auto once = [&]() -> double {
                            auto a = std::chrono::high_resolution_clock::now();
                            vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
                            VkResult wr = vkQueueWaitIdle(queue);
                            auto b = std::chrono::high_resolution_clock::now();
                            if (wr != VK_SUCCESS) return -1.0;
                            return std::chrono::duration<double>(b - a).count();
                        };
                        double ops = 2.0 * 16.0 * 16.0 * 16.0 * (double)NACC * (double)repeats * (double)totalSubgroups;

                        // Boost warmup (~300 ms) so the measurement is at peak clock.
                        bool lost = false;
                        double warm = 0.0;
                        while (warm < 0.3) { double s = once(); if (s < 0) { lost = true; break; } warm += s; }
                        double best = 1e30;
                        if (!lost) for (int t = 0; t < 3; ++t) {
                            double s = once();
                            if (s < 0) { lost = true; break; }
                            if (s < best) best = s;
                        }
                        vkDestroyPipeline(device, pipeline, nullptr);
                        if (lost) { rep.line("    reuse=%3d%%: device-lost (VkResult=-4)", reuse); continue; }

                        double rate = ops / best / 1e12;   // TFLOPS/TOPS (2-op)
                        // Bandwidth of the A/B tile loads (2 tiles per reload, plus
                        // the one initial load per subgroup). At low reuse the
                        // footprint exceeds cache, so this approaches real DRAM BW.
                        double reloadsPerSg = (double)repeats * (double)reloadsPer10 / 10.0;
                        double loadBytes = (double)totalSubgroups * 2.0 * (1.0 + reloadsPerSg) * 256.0 * (double)inElem;
                        double bw = loadBytes / best / 1e9;   // GB/s
                        rep.line("    reuse=%3d%% (reloads=%u/10, sg=%llu): time=%.3f ms | "
                                 "%.2f %s = %.2f TMAC/s | A/B load BW=%.1f GB/s",
                                 reuse, reloadsPer10, (unsigned long long)totalSubgroups,
                                 best * 1e3, rate, rateUnit, rate / 2.0, bw);
                    }
                }

                freeBufP(bufA); freeBufP(bufB); freeBufP(bufC); freeBufP(bufD);
                vkDestroyShaderModule(device, module, nullptr);
            }
        }
#endif  // ===== end disabled WMMA peak test =====

        // ---- Unified summary collected across the benchmarks ----
        // Three tests x three data types. Each test stores its best rate here and a
        // grouped, per-data-type summary is printed at the very end of the run.
        //   Test 1: Peak WMMA Throughput  (matrix engine, register-resident loop)
        //   Test 2: MATMUL with WMMA      (real 1024^3 GEMM on the matrix engine)
        //   Test 3: MATMUL with FMA       (real 1024^3 GEMM on the vector ALU)
        enum { TY_FP16_FP32 = 0, TY_FP16_FP16 = 1, TY_S8_S32 = 2, NUM_TY = 3 };
        const char *kTyLabel[NUM_TY] = { "fp16*fp16 -> fp32", "fp16*fp16 -> fp16", "s8*s8 -> s32" };
        const bool  kTyFloat[NUM_TY] = { true, true, false };
        double sumPeakWmma[NUM_TY] = { -1, -1, -1 };
        double sumMatWmma [NUM_TY] = { -1, -1, -1 };
        double sumMatFma  [NUM_TY] = { -1, -1, -1 };
        auto tyIndexOf = [](VkComponentTypeKHR in, VkComponentTypeKHR ot) -> int {
            if (in == VK_COMPONENT_TYPE_FLOAT16_KHR && ot == VK_COMPONENT_TYPE_FLOAT32_KHR) return TY_FP16_FP32;
            if (in == VK_COMPONENT_TYPE_FLOAT16_KHR && ot == VK_COMPONENT_TYPE_FLOAT16_KHR) return TY_FP16_FP16;
            if (in == VK_COMPONENT_TYPE_SINT8_KHR   && ot == VK_COMPONENT_TYPE_SINT32_KHR)  return TY_S8_S32;
            return -1;   // u8 etc.: benchmarked-but-not-summarized -> skipped below
        };

        // ===================== 1024x1024x1024 matmul (WMMA GEMM) =====================
        // Real C = A*B at M=N=K=1024 using the cooperative-matrix GEMM shader, for
        // fp16 and int8. Device-local buffers; one matmul is tiny (~2.1 GFLOP) so
        // each timed run dispatches R back-to-back (R calibrated to ~40 ms) and the
        // throughput is the best of 3 boost-warmed runs across a small tile sweep.
        {
            rep.line("");
            rep.line("==== 1024x1024x1024 matmul (WMMA GEMM) ====");
            const uint32_t MM = 1024, NN = 1024, KK = 1024;
            const uint32_t lM = 16, lN = 16, lK = 16;

            auto makeBufDL = [&](Buffer &b, VkDeviceSize bytes) -> bool {
                b.size = bytes;
                VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.size = bytes;
                bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (vkCreateBuffer(device, &bci, nullptr, &b.host) != VK_SUCCESS) return false;
                if (vkCreateBuffer(device, &bci, nullptr, &b.dev) != VK_SUCCESS) return false;
                VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, b.host, &mr);
                int32_t hi = findMemoryType(memProps, mr.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                int32_t di = findMemoryType(memProps, mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                if (di < 0) di = hi;
                if (hi < 0) return false;
                VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                mai.allocationSize = mr.size; mai.memoryTypeIndex = (uint32_t)hi;
                if (vkAllocateMemory(device, &mai, nullptr, &b.hostMem) != VK_SUCCESS) return false;
                mai.memoryTypeIndex = (uint32_t)di;
                if (vkAllocateMemory(device, &mai, nullptr, &b.devMem) != VK_SUCCESS) return false;
                vkBindBufferMemory(device, b.host, b.hostMem, 0);
                vkBindBufferMemory(device, b.dev, b.devMem, 0);
                vkMapMemory(device, b.hostMem, 0, bytes, 0, &b.ptr);
                return true;
            };
            auto freeBufDL = [&](Buffer &b) {
                if (b.host) vkDestroyBuffer(device, b.host, nullptr);
                if (b.dev) vkDestroyBuffer(device, b.dev, nullptr);
                if (b.hostMem) vkFreeMemory(device, b.hostMem, nullptr);
                if (b.devMem) vkFreeMemory(device, b.devMem, nullptr);
                b = Buffer{};
            };

            auto benchGemm = [&](bool tiled) {
              rep.line("");
              rep.line(tiled ? "########## shared-memory tiled GEMM ##########"
                             : "########## simple GEMM (streams A/B from global) ##########");
              for (const auto &combo : kCombos) {
                if (combo.isFloat && !canFloat) continue;
                if (!combo.isFloat && !canInt) continue;
                int tyIdx = tyIndexOf(combo.inputType, combo.outputType);
                if (tyIdx < 0) continue;   // only the three summarized data types
                bool have16 = false;
                for (const auto &c : coop)
                    if (c.scope == VK_SCOPE_SUBGROUP_KHR && c.MSize == 16 && c.NSize == 16 && c.KSize == 16 &&
                        c.AType == combo.inputType && c.BType == combo.inputType &&
                        c.CType == combo.outputType && c.ResultType == combo.outputType) { have16 = true; break; }

                rep.line("");
                rep.line("---- %s*%s -> %s [%s matmul 1024^3] ----",
                         componentTypeName(combo.inputType), componentTypeName(combo.inputType),
                         componentTypeName(combo.outputType), tiled ? "v07 WMMA tiled" : "simple WMMA");
                if (!have16) { rep.line("  no 16x16x16 subgroup shape; skip"); continue; }

                // (kernel source dump disabled)
                // dumpTypedShaderSource(shaderDir, tiled ? "gemm_v07.comp" : "gemm_coopmat.comp",
                //                       glslTypeName(combo.inputType), glslTypeName(combo.outputType), rep);

                std::string path = shaderDir + "/" + (tiled ? combo.shmemSpvFile : combo.spvFile);
                std::ifstream f(path, std::ios::binary | std::ios::ate);
                if (!f) { rep.line("  shader not found: %s", path.c_str()); continue; }
                std::streamsize sz = f.tellg(); f.seekg(0);
                std::vector<char> spv(sz); f.read(spv.data(), sz);
                VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                smci.codeSize = (size_t)sz; smci.pCode = (const uint32_t *)spv.data();
                VkShaderModule module;
                if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS) {
                    rep.line("  module create failed"); continue;
                }

                size_t inElem = componentBits(combo.inputType) / 8;
                size_t outElem = componentBits(combo.outputType) / 8;
                const char *rateUnit = combo.isFloat ? "TFLOPS" : "TOPS";

                // Occupancy-tuned configs for the target GPU (64 VGPR/thread,
                // 128KB LDS per 64 waves => 2KB/wave budget, 2MB L2, >=512 waves):
                // cR*cC = 4 keeps the accumulator/register footprint to ~half the
                // VGPR file; bigger workgroup tiles raise data reuse, but enough
                // subgroups per workgroup keep LDS per wave <= ~1-2KB so the full
                // 64 waves/unit stay resident. { cR, cC, WG_W, WG_H, BK }.
                // Raising the workgroup output tile raises data reuse: the global
                // A/B load : coopMatMulAdd ratio (tiled) is lM/TILE_M + lN/TILE_N,
                // so a 256x256 tile gives 1/16 + 1/16 = 1/8. cR*cC controls the
                // per-subgroup register footprint; more subgroups enlarge the tile
                // without growing it. { cR, cC, WG_W, WG_H, BK }.
                // v07 configs tuned for wave64 / 1024^3 (64 VGPR, 2KB LDS/wave for
                // 64 waves, >=512 waves). { cR, cC, WG_W, WG_H, BK }.
                //  - {4,2,4,2,16}: TILE 128x128, 8 frag, 8 waves/wg -> 512 waves,
                //    ~1.3KB LDS/wave, ~50 VGPR  (primary)
                // Warp-tile (C_ROWS x C_COLS) sweep -- reuse vs. occupancy.
                // Warp-tile reuse = (cR*cC)/(cR+cC) MACs per loaded fragment.
                // 4x2/2x2/4x4 keep a 128x128 block tile (WG_H*cR = WG_W*cC = 8).
                // 1x1/2x1 would need 64/32 waves/wg (>1024 threads) to keep
                // 128x128, so they use a smaller tile at max occupancy (16 waves)
                // -- their block-level reuse is also lower (note when comparing).
                //   1x1 :  1 frag/warp, 16 waves/wg, ~18 VGPR (reuse 0.50) TILE 64x64
                //   2x1 :  2 frag/warp, 16 waves/wg, ~22 VGPR (reuse 0.67) TILE 128x64
                //   2x2 :  4 frag/warp, 16 waves/wg, ~30 VGPR (reuse 1.00) TILE 128x128
                //   4x2 :  8 frag/warp,  8 waves/wg, ~50 VGPR (reuse 1.33) TILE 128x128
                //   4x4 : 16 frag/warp,  4 waves/wg, ~80 VGPR (reuse 2.00) TILE 128x128
                // { cR, cC, WG_W, WG_H, BK }
                struct Cfg { uint32_t cR, cC, wgW, wgH, bk; };
                const Cfg cfgs[] = {
                    { 1, 1, 4, 4, 16 },   // 1x1: TILE  64x64,   1 frag/warp, 16 waves/wg
                    { 2, 1, 4, 4, 16 },   // 2x1: TILE 128x64,   2 frag/warp, 16 waves/wg
                    { 2, 2, 4, 4, 16 },   // 2x2: TILE 128x128,  4 frag/warp, 16 waves/wg
                    { 4, 2, 4, 2, 16 },   // 4x2: TILE 128x128,  8 frag/warp,  8 waves/wg
                    { 4, 4, 2, 2, 16 },   // 4x4: TILE 128x128, 16 frag/warp,  4 waves/wg
                };
                double bestRate = 0.0, bestMs = 0.0, bestBW = 0.0; uint32_t bTileM = 0, bTileN = 0;
                bool dumpedExec = false;   // dump GPU ISA (sp3) once per type

                for (const auto &cfg : cfgs) {
                    uint32_t cR = cfg.cR, cC = cfg.cC, wgW = cfg.wgW, wgH = cfg.wgH;
                    uint32_t BKc = tiled ? cfg.bk : lK;   // K-block (shared) for tiled
                    uint32_t tileM = wgH * cR * lM, tileN = wgW * cC * lN, localX = wgW * wgH * reqSg;
                    auto rup = [](uint32_t v, uint32_t a) { return (v + a - 1) / a * a; };
                    uint32_t M = rup(MM, tileM), N = rup(NN, tileN), K = rup(KK, BKc);

                    // Resource / occupancy estimate (128KB LDS shared by up to 64 waves).
                    uint32_t wavesPerWG = wgW * wgH;
                    double ldsBytes = tiled ? (double)(tileM * BKc + BKc * tileN) * (double)inElem : 0.0;
                    double ldsPerWave = wavesPerWG ? ldsBytes / wavesPerWG : 0.0;
                    uint64_t dispatchedWaves = (uint64_t)(M / tileM) * (uint64_t)(N / tileN) * wavesPerWG;
                    // Matrix load : coopMatMulAdd ratio. Tiled: A/B staged once per
                    // workgroup -> lM/TILE_M + lN/TILE_N. Simple: each subgroup
                    // loads its own -> (cR+cC)/(cR*cC).
                    double loadRatio = tiled ? ((double)lM / tileM + (double)lN / tileN)
                                             : ((double)(cR + cC) / ((double)cR * cC));
                    rep.line("  [cR%u cC%u wg%ux%u bk%u] tile=%ux%u, LDS=%.0f B/wave, waves/wg=%u, "
                             "dispatched=%llu waves, load:MulAdd=%.3f (1/%.1f)",
                             cR, cC, wgW, wgH, BKc, tileM, tileN, ldsPerWave, wavesPerWG,
                             (unsigned long long)dispatchedWaves, loadRatio,
                             loadRatio > 0.0 ? 1.0 / loadRatio : 0.0);

                    Buffer bA, bB, bC, bD;
                    bool ok = makeBufDL(bA, (VkDeviceSize)M * K * inElem) &&
                              makeBufDL(bB, (VkDeviceSize)K * N * inElem) &&
                              makeBufDL(bC, (VkDeviceSize)M * N * outElem) &&
                              makeBufDL(bD, (VkDeviceSize)M * N * outElem);
                    if (!ok) { rep.line("  cfg alloc failed"); freeBufDL(bA); freeBufDL(bB); freeBufDL(bC); freeBufDL(bD); continue; }

                    auto fill = [&](Buffer &b, VkDeviceSize cnt) {
                        for (VkDeviceSize i = 0; i < cnt; ++i) {
                            if (combo.isFloat) ((uint16_t *)b.ptr)[i] = floatToHalf(0.01f);
                            else ((int8_t *)b.ptr)[i] = 1;
                        }
                    };
                    fill(bA, (VkDeviceSize)M * K); fill(bB, (VkDeviceSize)K * N);
                    memset(bC.ptr, 0, (size_t)bC.size); memset(bD.ptr, 0, (size_t)bD.size);

                    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
                    si.commandBufferCount = 1; si.pCommandBuffers = &cmd;

                    vkBeginCommandBuffer(cmd, &bi);
                    for (Buffer *bp : { &bA, &bB, &bC, &bD }) { VkBufferCopy cp = { 0, 0, bp->size }; vkCmdCopyBuffer(cmd, bp->host, bp->dev, 1, &cp); }
                    vkEndCommandBuffer(cmd);
                    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(queue);

                    VkDescriptorBufferInfo dbi[4] = { { bA.dev, 0, VK_WHOLE_SIZE }, { bB.dev, 0, VK_WHOLE_SIZE },
                                                      { bC.dev, 0, VK_WHOLE_SIZE }, { bD.dev, 0, VK_WHOLE_SIZE } };
                    VkWriteDescriptorSet wds[4];
                    for (int i = 0; i < 4; ++i) {
                        wds[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                        wds[i].dstSet = descSet; wds[i].dstBinding = (uint32_t)i; wds[i].descriptorCount = 1;
                        wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[i].pBufferInfo = &dbi[i];
                    }
                    vkUpdateDescriptorSets(device, 4, wds, 0, nullptr);

                    float alpha = 1.0f, beta = 0.0f;
                    uint32_t spec[14];
                    spec[0] = lM; spec[1] = lN; spec[2] = lK; spec[3] = M; spec[4] = N; spec[5] = K;
                    spec[6] = cR; spec[7] = cC; spec[8] = wgW; spec[9] = wgH;
                    memcpy(&spec[10], &alpha, 4); memcpy(&spec[11], &beta, 4);
                    uint32_t numSpec;
                    if (tiled) { spec[12] = BKc; spec[13] = localX; numSpec = 14; }  // BK then localX(id13)
                    else       { spec[12] = localX; numSpec = 13; }                  // localX at id12
                    VkSpecializationMapEntry ents[14];
                    for (uint32_t i = 0; i < numSpec; ++i) ents[i] = { i, (uint32_t)(i * 4), 4 };
                    VkSpecializationInfo spi = { numSpec, ents, (size_t)numSpec * 4, spec };
                    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = {
                        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO };
                    rss.requiredSubgroupSize = reqSg;
                    VkPipelineShaderStageCreateInfo ssci = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
                    ssci.pNext = &rss; ssci.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
                    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT; ssci.module = module; ssci.pName = "main";
                    ssci.pSpecializationInfo = &spi;
                    VkComputePipelineCreateInfo cpci2 = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
                    cpci2.stage = ssci; cpci2.layout = pipelineLayout;
                    if (wantPipeExec) cpci2.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                                                      VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
                    VkPipeline pipeline;
                    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci2, nullptr, &pipeline) != VK_SUCCESS) {
                        rep.line("  cfg pipeline create failed"); freeBufDL(bA); freeBufDL(bB); freeBufDL(bC); freeBufDL(bD); continue;
                    }
                    // sp3 / ISA dump disabled. Keep only the per-config VGPR/SGPR
                    // statistics (lightweight, no ISA text) for the occupancy
                    // comparison between 4x2 / 2x2 / 4x4.
                    if (wantPipeExec)
                        dumpPipelineStatsOnly(device, pipeline, rep, "stat", pfnPeProps, pfnPeStats);
                    (void)dumpedExec;

                    uint32_t gx = N / tileN, gy = M / tileM;
                    VkResult wr = VK_SUCCESS;
                    auto recordR = [&](uint32_t R) {
                        vkBeginCommandBuffer(cmd, &bi);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                        for (uint32_t i = 0; i < R; ++i) vkCmdDispatch(cmd, gx, gy, 1);
                        vkEndCommandBuffer(cmd);
                    };
                    auto timed = [&]() -> double {
                        auto a = std::chrono::high_resolution_clock::now();
                        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
                        wr = vkQueueWaitIdle(queue);
                        auto b = std::chrono::high_resolution_clock::now();
                        if (wr != VK_SUCCESS) return -1.0;
                        return std::chrono::duration<double>(b - a).count();
                    };

                    // Calibrate R so one timed submit is ~40 ms (watchdog-safe).
                    recordR(1); timed(); double t1 = timed();
                    if (t1 < 0) { rep.line("  cfg device-lost"); vkDestroyPipeline(device, pipeline, nullptr); freeBufDL(bA); freeBufDL(bB); freeBufDL(bC); freeBufDL(bD); continue; }
                    uint32_t R = (uint32_t)std::max(1.0, 0.04 / t1); if (R > 4000) R = 4000;
                    recordR(R);
                    double w = 0.0; bool lost = false;       // boost warmup ~200 ms
                    while (w < 0.2) { double s = timed(); if (s < 0) { lost = true; break; } w += s; }
                    double best = 1e30;
                    if (!lost) for (int t = 0; t < 3; ++t) { double s = timed(); if (s < 0) { lost = true; break; } if (s < best) best = s; }
                    vkDestroyPipeline(device, pipeline, nullptr);

                    if (!lost) {
                        double ops = 2.0 * (double)M * (double)N * (double)K * (double)R;
                        double rate = ops / best / 1e12;
                        double msPerMatmul = best / (double)R;
                        // Algorithmic per-matmul DRAM bandwidth: A and B read once,
                        // C read, D written (the minimum traffic a single matmul
                        // needs if perfectly cached internally).
                        double uniqueBytes = ((double)M * K + (double)K * N) * inElem + 2.0 * (double)M * N * outElem;
                        double minBW = uniqueBytes / msPerMatmul / 1e9;   // GB/s
                        rep.line("  tile %ux%u (R=%u, dim=%ux%ux%u): %.4f ms/matmul | %.2f %s = %.2f TMAC/s "
                                 "| min DRAM BW=%.1f GB/s",
                                 tileM, tileN, R, M, N, K, msPerMatmul * 1e3, rate, rateUnit, rate / 2.0, minBW);
                        if (rate > bestRate) { bestRate = rate; bestMs = msPerMatmul * 1e3; bTileM = tileM; bTileN = tileN; bestBW = minBW; }
                    } else rep.line("  cfg device-lost mid-measurement");

                    freeBufDL(bA); freeBufDL(bB); freeBufDL(bC); freeBufDL(bD);
                }
                vkDestroyShaderModule(device, module, nullptr);
                if (bestRate > 0.0)
                    rep.line("  BEST: tile %ux%u  %.4f ms/matmul | %.2f %s = %.2f TMAC/s "
                             "| min DRAM BW=%.1f GB/s",
                             bTileM, bTileN, bestMs, bestRate, rateUnit, bestRate / 2.0, bestBW);
                if (tiled && bestRate > 0.0) sumMatWmma[tyIdx] = bestRate;   // Test 3 result
              }
            };
            // v06-style FMA/MAD GEMM (no WMMA): warp/thread tiled, transposed A.
            auto benchFma = [&]() {
                rep.line("");
                rep.line("########## v06 FMA GEMM (no WMMA; scalar multiply-add) ##########");
                // Tuned for wave64 / 1024^3 (64 VGPR, 2KB LDS/wave for 64 waves,
                // >=512 waves). { BM, BN, BK, TM, TN }.
                struct FCfg { uint32_t BM, BN, BK, TM, TN; };
                const FCfg cfgs[] = {
                    { 128, 128, 16, 8, 4 },  // 512 threads/wg (8 waves) -> 512 waves, ~1KB LDS/wave
                    { 128, 128, 8,  8, 4 },  // smaller LDS
                    { 128, 64, 16, 8, 4 },   // 256 threads/wg, 128 wg -> 512 waves
                };
                for (const auto &combo : kCombos) {
                    if (combo.isFloat && !canFloat) continue;
                    if (!combo.isFloat && !canInt) continue;
                    int tyIdx = tyIndexOf(combo.inputType, combo.outputType);
                    if (tyIdx < 0) continue;   // only the three summarized data types
                    rep.line("");
                    rep.line("---- %s*%s -> %s [optimized FMA matmul 1024^3] ----",
                             componentTypeName(combo.inputType), componentTypeName(combo.inputType),
                             componentTypeName(combo.outputType));
                    // Show the shader source for this data type.
                    dumpTypedShaderSource(shaderDir, "fma_v06.comp",
                                          glslTypeName(combo.inputType),
                                          glslTypeName(combo.outputType), rep);
                    std::string path = shaderDir + "/" + combo.fmaSpvFile;
                    std::ifstream f(path, std::ios::binary | std::ios::ate);
                    if (!f) { rep.line("  shader not found: %s", path.c_str()); continue; }
                    std::streamsize sz = f.tellg(); f.seekg(0);
                    std::vector<char> spv(sz); f.read(spv.data(), sz);
                    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                    smci.codeSize = (size_t)sz; smci.pCode = (const uint32_t *)spv.data();
                    VkShaderModule module;
                    if (vkCreateShaderModule(device, &smci, nullptr, &module) != VK_SUCCESS) { rep.line("  module create failed"); continue; }

                    size_t inElem = componentBits(combo.inputType) / 8;
                    size_t outElem = componentBits(combo.outputType) / 8;
                    const char *rateUnit = combo.isFloat ? "TFLOPS" : "TOPS";
                    double bestRate = 0.0, bestMs = 0.0, bestBW = 0.0; uint32_t bBM = 0, bBN = 0;
                    bool dumpedIsa = false;   // dump the GPU ISA (sp3) once per type

                    for (const auto &cfg : cfgs) {
                        uint32_t BM = cfg.BM, BN = cfg.BN, BK = cfg.BK, TM = cfg.TM, TN = cfg.TN;
                        uint32_t localX = (BM / TM) * (BN / TN);
                        auto rup = [](uint32_t v, uint32_t a) { return (v + a - 1) / a * a; };
                        uint32_t M = rup(MM, BM), N = rup(NN, BN), K = rup(KK, BK);

                        Buffer bA, bB, bC, bD;
                        bool ok = makeBufDL(bA, (VkDeviceSize)M * K * inElem) && makeBufDL(bB, (VkDeviceSize)K * N * inElem) &&
                                  makeBufDL(bC, (VkDeviceSize)M * N * outElem) && makeBufDL(bD, (VkDeviceSize)M * N * outElem);
                        if (!ok) { rep.line("  cfg alloc failed"); freeBufDL(bA); freeBufDL(bB); freeBufDL(bC); freeBufDL(bD); continue; }
                        auto fill = [&](Buffer &b, VkDeviceSize cnt) { for (VkDeviceSize i = 0; i < cnt; ++i) { if (combo.isFloat) ((uint16_t *)b.ptr)[i] = floatToHalf(0.01f); else ((int8_t *)b.ptr)[i] = 1; } };
                        fill(bA, (VkDeviceSize)M * K); fill(bB, (VkDeviceSize)K * N);
                        memset(bC.ptr, 0, (size_t)bC.size); memset(bD.ptr, 0, (size_t)bD.size);

                        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
                        vkBeginCommandBuffer(cmd, &bi);
                        for (Buffer *bp : { &bA, &bB, &bC, &bD }) { VkBufferCopy cp = { 0, 0, bp->size }; vkCmdCopyBuffer(cmd, bp->host, bp->dev, 1, &cp); }
                        vkEndCommandBuffer(cmd);
                        vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE); vkQueueWaitIdle(queue);

                        VkDescriptorBufferInfo dbi[4] = { { bA.dev, 0, VK_WHOLE_SIZE }, { bB.dev, 0, VK_WHOLE_SIZE }, { bC.dev, 0, VK_WHOLE_SIZE }, { bD.dev, 0, VK_WHOLE_SIZE } };
                        VkWriteDescriptorSet wds[4]; for (int i = 0; i < 4; ++i) { wds[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; wds[i].dstSet = descSet; wds[i].dstBinding = (uint32_t)i; wds[i].descriptorCount = 1; wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[i].pBufferInfo = &dbi[i]; }
                        vkUpdateDescriptorSets(device, 4, wds, 0, nullptr);

                        uint32_t spec[9] = { M, N, K, BM, BN, BK, TM, TN, localX };
                        VkSpecializationMapEntry ents[9]; for (uint32_t i = 0; i < 9; ++i) ents[i] = { i, (uint32_t)(i * 4), 4 };
                        VkSpecializationInfo spi = { 9, ents, sizeof(spec), spec };
                        VkPipelineShaderStageCreateInfo ssci = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
                        ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT; ssci.module = module; ssci.pName = "main"; ssci.pSpecializationInfo = &spi;
                        VkComputePipelineCreateInfo cpci2 = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
                        cpci2.stage = ssci; cpci2.layout = pipelineLayout;
                        if (wantPipeExec) cpci2.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR |
                                                          VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR;
                        VkPipeline pipeline;
                        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci2, nullptr, &pipeline) != VK_SUCCESS) {
                            rep.line("  [BM%u BN%u BK%u TM%u TN%u] pipeline create failed", BM, BN, BK, TM, TN);
                            freeBufDL(bA); freeBufDL(bB); freeBufDL(bC); freeBufDL(bD); continue;
                        }
                        rep.line("  [BM%u BN%u BK%u TM%u TN%u] threads/wg=%u", BM, BN, BK, TM, TN, localX);
                        if (wantPipeExec) {
                            if (!dumpedIsa) {   // full ISA (sp3) once per type
                                rep.line("    -- compiled GPU executable (ISA / sp3, statistics) --");
                                dumpPipelineExecutables(device, pipeline, rep, pfnPeProps, pfnPeStats, pfnPeIR);
                                dumpedIsa = true;
                            } else {
                                dumpPipelineStatsOnly(device, pipeline, rep, "stat", pfnPeProps, pfnPeStats);
                            }
                        }

                        uint32_t gx = N / BN, gy = M / BM;
                        VkResult wr = VK_SUCCESS;
                        auto recordR = [&](uint32_t R) { vkBeginCommandBuffer(cmd, &bi); vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr); vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline); for (uint32_t i = 0; i < R; ++i) vkCmdDispatch(cmd, gx, gy, 1); vkEndCommandBuffer(cmd); };
                        auto timed = [&]() -> double { auto a = std::chrono::high_resolution_clock::now(); vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE); wr = vkQueueWaitIdle(queue); auto b = std::chrono::high_resolution_clock::now(); if (wr != VK_SUCCESS) return -1.0; return std::chrono::duration<double>(b - a).count(); };
                        recordR(1); timed(); double t1 = timed();
                        if (t1 < 0) { rep.line("    device-lost"); vkDestroyPipeline(device, pipeline, nullptr); freeBufDL(bA); freeBufDL(bB); freeBufDL(bC); freeBufDL(bD); continue; }
                        uint32_t R = (uint32_t)std::max(1.0, 0.04 / t1); if (R > 4000) R = 4000;
                        recordR(R);
                        double w = 0.0; bool lost = false; while (w < 0.2) { double s = timed(); if (s < 0) { lost = true; break; } w += s; }
                        double best = 1e30; if (!lost) for (int t = 0; t < 3; ++t) { double s = timed(); if (s < 0) { lost = true; break; } if (s < best) best = s; }
                        vkDestroyPipeline(device, pipeline, nullptr);
                        if (!lost) {
                            double ops = 2.0 * (double)M * (double)N * (double)K * (double)R;
                            double rate = ops / best / 1e12;
                            double msPerMatmul = best / (double)R;
                            double uniqueBytes = ((double)M * K + (double)K * N) * inElem + 2.0 * (double)M * N * outElem;
                            double minBW = uniqueBytes / msPerMatmul / 1e9;
                            rep.line("    -> %.4f ms/matmul | %.2f %s = %.2f TMAC/s | min DRAM BW=%.1f GB/s",
                                     msPerMatmul * 1e3, rate, rateUnit, rate / 2.0, minBW);
                            if (rate > bestRate) { bestRate = rate; bestMs = msPerMatmul * 1e3; bBM = BM; bBN = BN; bestBW = minBW; }
                        } else rep.line("    device-lost mid-measurement");
                        freeBufDL(bA); freeBufDL(bB); freeBufDL(bC); freeBufDL(bD);
                    }
                    vkDestroyShaderModule(device, module, nullptr);
                    if (bestRate > 0.0)
                        rep.line("  BEST: BM%u BN%u  %.4f ms/matmul | %.2f %s = %.2f TMAC/s | min DRAM BW=%.1f GB/s",
                                 bBM, bBN, bestMs, bestRate, rateUnit, bestRate / 2.0, bestBW);
                    if (bestRate > 0.0) sumMatFma[tyIdx] = bestRate;   // Test 4 result
                }
            };

            // Only the WMMA matmul runs, sweeping warp-tile 4x2 / 2x2 / 4x4.
            benchGemm(true);   // MATMUL with WMMA (v07 tiled GEMM, 1024^3)
            // benchFma();     // FMA GEMM disabled
            (void)benchGemm; (void)benchFma;   // benchGemm(false)/benchFma not run
        }

#if 0   // ===== WMMA peak throughput DISABLED (only the WMMA matmul sweep runs) =====
        // ===================== WMMA peak throughput =====================
        // Pure-compute peak on the cooperative-matrix (matrix) unit: load a 16x16
        // tile once into registers and repeat coopMatMulAdd REPEATS times (no
        // memory traffic), isolating the matrix unit's issue rate.
        {
            rep.line("");
            rep.line("==== WMMA peak throughput ====");
            const uint32_t REPEATS = 1024;
            const uint32_t NACC = 4;            // WMMA accumulators per subgroup
            const uint32_t SG_PER_WG = 4;
            const uint32_t WORKGROUPS = 512;
            const uint32_t BUF_TILES_W = 256;   // wmma_peak tile buffer (cache resident)
            uint32_t localX = reqSg * SG_PER_WG;
            uint64_t totalSubgroups = (uint64_t)WORKGROUPS * SG_PER_WG;
            rep.line("config: workgroups=%u, localX=%u, waves=%llu, REPEATS=%u, NACC=%u",
                     WORKGROUPS, localX, (unsigned long long)totalSubgroups, REPEATS, NACC);

            struct T { const char *label, *suffix; VkComponentTypeKHR in, ot; bool isFloat; };
            const T types[] = {
                { "fp16*fp16->fp32", "fp16_fp32", VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT32_KHR, true },
                { "fp16*fp16->fp16", "fp16_fp16", VK_COMPONENT_TYPE_FLOAT16_KHR, VK_COMPONENT_TYPE_FLOAT16_KHR, true },
                { "s8*s8->s32",      "s8_s32",    VK_COMPONENT_TYPE_SINT8_KHR,   VK_COMPONENT_TYPE_SINT32_KHR,  false },
            };
            // Results stored into the shared summary array sumPeakWmma
            // (index ti maps directly to TY_FP16_FP32/TY_FP16_FP16/TY_S8_S32).

            auto makeHV = [&](Buffer &b, VkDeviceSize bytes) -> bool {
                b.size = bytes;
                VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
                bci.size = bytes; bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
                if (vkCreateBuffer(device, &bci, nullptr, &b.dev) != VK_SUCCESS) return false;
                VkMemoryRequirements mr; vkGetBufferMemoryRequirements(device, b.dev, &mr);
                int32_t mi = findMemoryType(memProps, mr.memoryTypeBits,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
                if (mi < 0) return false;
                VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
                mai.allocationSize = mr.size; mai.memoryTypeIndex = (uint32_t)mi;
                if (vkAllocateMemory(device, &mai, nullptr, &b.devMem) != VK_SUCCESS) return false;
                vkBindBufferMemory(device, b.dev, b.devMem, 0);
                vkMapMemory(device, b.devMem, 0, bytes, 0, &b.ptr);
                return true;
            };
            auto freeHV = [&](Buffer &b) {
                if (b.dev) vkDestroyBuffer(device, b.dev, nullptr);
                if (b.devMem) vkFreeMemory(device, b.devMem, nullptr);
                b = Buffer{};
            };
            auto bindAll = [&](Buffer *bufs4) {
                VkDescriptorBufferInfo dbi[4]; VkWriteDescriptorSet wds[4];
                for (int i = 0; i < 4; ++i) {
                    dbi[i] = { bufs4[i].dev, 0, VK_WHOLE_SIZE };
                    wds[i] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                    wds[i].dstSet = descSet; wds[i].dstBinding = (uint32_t)i; wds[i].descriptorCount = 1;
                    wds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wds[i].pBufferInfo = &dbi[i];
                }
                vkUpdateDescriptorSets(device, 4, wds, 0, nullptr);
            };
            // Calibrate-batch + boost-warm + best-of-3, return 2-op rate (T/s).
            auto measure = [&](VkPipeline pipeline, double opsPerDispatch) -> double {
                VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
                VkResult wr = VK_SUCCESS;
                auto recordR = [&](uint32_t R) {
                    vkBeginCommandBuffer(cmd, &bi);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                    for (uint32_t i = 0; i < R; ++i) vkCmdDispatch(cmd, WORKGROUPS, 1, 1);
                    vkEndCommandBuffer(cmd);
                };
                auto timed = [&]() -> double {
                    auto a = std::chrono::high_resolution_clock::now();
                    vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE);
                    wr = vkQueueWaitIdle(queue);
                    auto b = std::chrono::high_resolution_clock::now();
                    if (wr != VK_SUCCESS) return -1.0;
                    return std::chrono::duration<double>(b - a).count();
                };
                recordR(1); timed(); double t1 = timed(); if (t1 < 0) return -1.0;
                uint32_t R = (uint32_t)std::max(1.0, 0.04 / t1); if (R > 4000) R = 4000;
                recordR(R);
                double w = 0.0; while (w < 0.2) { double s = timed(); if (s < 0) return -1.0; w += s; }
                double best = 1e30; for (int t = 0; t < 3; ++t) { double s = timed(); if (s < 0) return -1.0; if (s < best) best = s; }
                return opsPerDispatch * (double)R / best / 1e12;
            };
            auto loadModule = [&](const std::string &path) -> VkShaderModule {
                std::ifstream f(path, std::ios::binary | std::ios::ate);
                if (!f) return VK_NULL_HANDLE;
                std::streamsize sz = f.tellg(); f.seekg(0);
                std::vector<char> spv(sz); f.read(spv.data(), sz);
                VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                smci.codeSize = (size_t)sz; smci.pCode = (const uint32_t *)spv.data();
                VkShaderModule m; if (vkCreateShaderModule(device, &smci, nullptr, &m) != VK_SUCCESS) return VK_NULL_HANDLE;
                return m;
            };

            for (uint32_t ti = 0; ti < 3; ++ti) {
                const T &t = types[ti];
                size_t inElem = componentBits(t.in) / 8, outElem = componentBits(t.ot) / 8;
                rep.line("");
                rep.line("---- %s ----", t.label);

                // ---- WMMA peak (needs a 16x16x16 subgroup shape) ----
                bool have16 = false;
                for (const auto &c : coop)
                    if (c.scope == VK_SCOPE_SUBGROUP_KHR && c.MSize == 16 && c.NSize == 16 && c.KSize == 16 &&
                        c.AType == t.in && c.BType == t.in && c.CType == t.ot && c.ResultType == t.ot) { have16 = true; break; }
                if (have16) {
                    VkShaderModule mod = loadModule(shaderDir + "/wmma_peak_" + std::string(t.suffix) + ".spv");
                    if (mod) {
                        Buffer bufs[4];
                        bool ok = makeHV(bufs[0], (VkDeviceSize)BUF_TILES_W * 256 * inElem) &&
                                  makeHV(bufs[1], (VkDeviceSize)BUF_TILES_W * 256 * inElem) &&
                                  makeHV(bufs[2], (VkDeviceSize)256 * outElem) &&
                                  makeHV(bufs[3], (VkDeviceSize)totalSubgroups * 256 * outElem);
                        if (ok) {
                            for (VkDeviceSize i = 0; i < (VkDeviceSize)BUF_TILES_W * 256; ++i) {
                                if (t.isFloat) { ((uint16_t *)bufs[0].ptr)[i] = floatToHalf(0.01f); ((uint16_t *)bufs[1].ptr)[i] = floatToHalf(0.01f); }
                                else { ((int8_t *)bufs[0].ptr)[i] = 1; ((int8_t *)bufs[1].ptr)[i] = 1; }
                            }
                            bindAll(bufs);
                            uint32_t spec[8] = { 16, 16, 16, REPEATS, NACC, localX, 0u, BUF_TILES_W };
                            VkSpecializationMapEntry ents[8]; for (uint32_t i = 0; i < 8; ++i) ents[i] = { i, (uint32_t)(i * 4), 4 };
                            VkSpecializationInfo spi = { 8, ents, sizeof(spec), spec };
                            VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO };
                            rss.requiredSubgroupSize = reqSg;
                            VkPipelineShaderStageCreateInfo ssci = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
                            ssci.pNext = &rss; ssci.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
                            ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT; ssci.module = mod; ssci.pName = "main"; ssci.pSpecializationInfo = &spi;
                            VkComputePipelineCreateInfo cpci2 = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
                            cpci2.stage = ssci; cpci2.layout = pipelineLayout;
                            VkPipeline pipe;
                            if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpci2, nullptr, &pipe) == VK_SUCCESS) {
                                double ops = 2.0 * 16.0 * 16.0 * 16.0 * (double)NACC * (double)REPEATS * (double)totalSubgroups;
                                sumPeakWmma[ti] = measure(pipe, ops);
                                vkDestroyPipeline(device, pipe, nullptr);
                                rep.line("  WMMA: %.2f %s (%.2f TMAC/s)", sumPeakWmma[ti], t.isFloat ? "TFLOPS" : "TOPS", sumPeakWmma[ti] / 2.0);
                            } else rep.line("  WMMA: pipeline create failed");
                        }
                        freeHV(bufs[0]); freeHV(bufs[1]); freeHV(bufs[2]); freeHV(bufs[3]);
                        vkDestroyShaderModule(device, mod, nullptr);
                    }
                } else rep.line("  WMMA: no 16x16x16 shape (skip)");
            }

        }
#endif  // ===== end disabled WMMA peak throughput =====
        (void)sumPeakWmma; (void)sumMatFma;   // tests disabled; arrays unused

        // ============================ FINAL SUMMARY ============================
        // Best WMMA matmul (across the 4x2 / 2x2 / 4x4 warp-tile sweep) per type.
        {
            rep.line("");
            rep.line("============================ SUMMARY ============================");
            rep.line("(fp16 -> TFLOPS, s8 -> TOPS; 2 ops per multiply-add; best of 4x2/2x2/4x4)");
            for (int ti = 0; ti < NUM_TY; ++ti) {
                const char *unit = kTyFloat[ti] ? "TFLOPS" : "TOPS";
                if (sumMatWmma[ti] >= 0.0)
                    rep.line("  %-18s  MATMUL with WMMA : %8.2f %s", kTyLabel[ti], sumMatWmma[ti], unit);
                else
                    rep.line("  %-18s  MATMUL with WMMA : %8s", kTyLabel[ti], "n/a");
            }
            rep.line("================================================================");
        }

        vkDestroyCommandPool(device, cmdPool, nullptr);
        vkDestroyDescriptorPool(device, descPool, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, dsl, nullptr);
    }

    rep.line("");
    rep.line("done.");

    if (device) { vkDeviceWaitIdle(device); vkDestroyDevice(device, nullptr); }
    if (instance) vkDestroyInstance(instance, nullptr);
    return rep.text;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_exynos_vkcoopmat_MainActivity_runBenchmark(JNIEnv *env, jobject /*thiz*/, jstring shaderDir) {
    const char *dir = env->GetStringUTFChars(shaderDir, nullptr);
    std::string result;
    try {
        result = runBenchmark(dir ? dir : "");
    } catch (const std::exception &e) {
        result = std::string("EXCEPTION: ") + e.what();
        ALOGE("exception: %s", e.what());
    } catch (...) {
        result = "EXCEPTION: unknown";
    }
    env->ReleaseStringUTFChars(shaderDir, dir);
    return env->NewStringUTF(result.c_str());
}
