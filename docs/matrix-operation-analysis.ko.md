# 행렬 연산 & 데이터 타입 분석 (한글)

> English version: [matrix-operation-analysis.md](matrix-operation-analysis.md)

이 문서는 이 벤치마크가 **어떤 행렬 연산을 측정**하고 **어떤 데이터 타입을
다루는지**를 실제 소스(`shaders/gemm_coopmat.comp`,
`app/src/main/cpp/native-bench.cpp`) 기준으로 설명합니다.

## 핵심 연산: GEMM (행렬 곱-누산)

측정하는 연산은 **일반 행렬 곱셈(General Matrix Multiply, GEMM)** 입니다:

```
D = alpha * (A x B) + beta * C
```

- `A` = `M x K`, `B` = `K x N`, `C` / `D` = `M x N`, 모두 **row-major**
  (`gemm_coopmat.comp:13`).
- 벤치마크 실행 시 호스트가 `alpha = 1.0`, `beta = 0.0`으로 설정하므로
  (`native-bench.cpp:708`), 측정 대상은 실질적으로 **순수 행렬곱 `D = A x B`**
  입니다.
- 문제 크기는 약 `2048 x 2048 x 2048`이며 타일 / `lK` 배수로 올림되고
  (`native-bench.cpp:603-606`), dispatch마다 **10회 반복** 실행됩니다
  (`repeats = 10`).

하드웨어 기본 단위는 **subgroup scope의 cooperative-matrix** 명령
`coopMatMulAdd` 입니다 (`gemm_coopmat.comp:90`). 이 명령은 GPU의 매트릭스/텐서
엔진에서 작은 `lM x lN x lK` 타일을 곱-누산하며, 셰이더는 이 타일들을 subgroup
당 `C_ROWS x C_COLS` 누산 그리드로 쌓고 `K` 축을 따라 진행하여 전체 GEMM을
구성합니다.

### 성능 지표

처리량은 **TFLOPS**로 보고됩니다 (`native-bench.cpp:778-779`):

```
flops  = 2 * M * N * K * repeats     // 원소당 곱 1 + 덧셈 1 = 2 FLOP
tflops = flops / seconds / 1e12
```

## 데이터 타입: 4가지 조합

벤치마크 대상 조합은 `kCombos`에 정의되어 있습니다
(`native-bench.cpp:199-204`). 각 조합은 **입력 타입 (A, B)** 과
**출력 / 누산 타입 (C, D)** 을 짝지웁니다:

| # | 입력 (A, B)          | 출력 / 누산 (C, D)      | 분류       | 셰이더                 |
|---|----------------------|-------------------------|------------|------------------------|
| 1 | **fp16** (float16)   | **fp32** (float32)      | 부동소수   | `gemm_fp16_fp32.spv`   |
| 2 | **fp16**             | **fp16**                | 부동소수   | `gemm_fp16_fp16.spv`   |
| 3 | **s8** (부호 int8)   | **s32** (부호 int32)    | 정수       | `gemm_s8_s32.spv`      |
| 4 | **u8** (무부호 int8) | **u32** (무부호 int32)  | 정수       | `gemm_u8_u32.spv`      |

이 4가지는 ML / 추론 가속에서 가장 중요한 조합입니다:

- **fp16 -> fp32** — 16비트 반정밀도 입력으로 메모리/대역폭을 절약하면서 32비트
  누산기로 정밀도를 유지합니다. 딥러닝 학습/추론의 표준 패턴입니다.
- **fp16 -> fp16** — 누산까지 16비트: 더 빠르지만 정밀도 손실 위험이 있습니다.
- **s8 -> s32 / u8 -> u32** — **INT8 양자화 추론(quantized inference)** 의 표준:
  8비트 입력에 오버플로 방지를 위한 32비트 누산기.

## 타입 선택은 정적이 아니라 디바이스가 결정한다

위 4가지 조합은 **후보**이지 무조건 실행되는 목록이 아닙니다. 실제 측정 대상은
런타임에 결정됩니다:

1. **GPU가 광고하는 것을 덤프.**
   `vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR`로 지원되는 모든
   `(M, N, K, A/B/C/D 타입, scope)` 구성을 열거합니다
   (`native-bench.cpp:333-351`).

2. **Feature 게이팅** (`native-bench.cpp:381-386`):
   - **float** 경로(조합 1, 2)는 `shaderFloat16` + `storageBuffer16BitAccess`
     필요.
   - **int** 경로(조합 3, 4)는 `shaderInt8` + `storageBuffer8BitAccess` 필요.
   - 공통으로 `cooperativeMatrix`, `vulkanMemoryModel`, `subgroupSizeControl`,
     `computeFullSubgroups` 필요.

3. **Shape 매칭.** 살아남은 각 조합에 대해 A/B/C/D 타입이 정확히 일치하는
   **subgroup-scope** 광고 shape를 찾고 (`native-bench.cpp:480-486`), 그 shape의
   `lM` / `lN` / `lK`를 specialization constant로 셰이더에 주입합니다.

4. **Graceful degradation.** 매칭되는 게 없으면 GEMM을 건너뛰고 실패 대신
   **capability 리포트만** 출력합니다
   (`native-bench.cpp:388-394`, `:490-497`).

따라서 Exynos 2600에서 실제로 측정되는 집합은 드라이버가 매트릭스 엔진에 어떤
타입을 노출하느냐에 따라 달라집니다.

## 각 타입의 측정 방식

- **타일 스윕** — 타입마다 3가지 workgroup/타일 구성을 시도합니다
  (`native-bench.cpp:583-587`): `{invocations 128, 2x2}`, `{256, 2x2}`,
  `{256, 4x2}`. subgroup 당 누산 그리드(`C_ROWS x C_COLS`)와 workgroup 크기를
  바꿔가며 좋은 구성을 탐색합니다.
- **Subgroup 제어** — subgroup 크기를 `requiredSubgroupSize`로 고정하고 full
  subgroups를 강제합니다 (`native-bench.cpp:721-727`).
- **타이밍** — warmup 3회 후 1회 측정 submit; 각 dispatch는 내부적으로 10회
  반복하며 반복 사이에 메모리 배리어를 둡니다 (`native-bench.cpp:752-775`).
- **입력값** — fp16 오버플로 / 정밀도 문제를 피하기 위해 `-2..2`의 작은 정수로
  채웁니다 (`native-bench.cpp:661-666`).

## 요약

**GPU의 subgroup-scope cooperative-matrix 엔진을 이용한 GEMM
(`D = alpha*A*B + beta*C`)의 처리량(TFLOPS)** 을, **4가지 데이터 타입 조합
(fp16/fp16, fp16/fp32, s8/s32, u8/u32)** 에 대해, 타일 구성을 바꿔가며, 그리고
디바이스가 실제 지원하는 범위로 한정하여 측정하는 벤치마크입니다.
