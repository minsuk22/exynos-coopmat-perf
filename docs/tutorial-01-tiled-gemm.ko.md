# 튜토리얼 1 — 타일드 GEMM 커널 읽기 (`gemm_tiled_scalar.comp`)

GPU 커널(컴퓨트 셰이더) 코드가 처음인 분을 위한 첫 튜토리얼입니다. 가장 단순한
타일드 행렬곱 커널 `shaders/gemm_tiled_scalar.comp`를 **"커널 코드를 읽는 법"** 부터
한 줄씩 풀어봅니다.

> 이 튜토리얼은 FMA GEMM 학습 로드맵의 1단계입니다. 다음 단계(`fma_v06`, 외적)는
> 이 커널의 확장입니다.

---

## 0. 가장 중요한 개념 — 커널은 "스레드 1개의 일"을 적은 것

CPU 코드였다면 행렬곱은 3중 루프입니다:
```c
for (row = 0; row < M; row++)
  for (col = 0; col < N; col++)
    for (k = 0; k < K; k++)
      C[row][col] += A[row][k] * B[k][col];
```

**GPU(커널)는 다릅니다.** GPU는 이 코드를 **수천 개 스레드가 동시에** 실행하고, **각
스레드는 자기 좌표만 다릅니다.** 그래서 커널 코드에는:

- `row`, `col` 루프가 **없습니다** → 내 좌표는 **내 스레드 ID로 주어짐**
- 나는 **내 출력 `C[row][col]` 딱 1개**만 계산 (k 루프만 돎)
- `row`/`col`을 도는 일은 **"수천 스레드가 병렬로"** 대신함

> 커널을 읽을 때는 **"나는 스레드 하나다. 내 좌표는 뭐고, 나는 뭘 계산하지?"** 관점으로
> 봅니다. 이게 커널 코드를 읽는 첫 번째 열쇠입니다.

---

## 1. GPU 스레드 계층 (좌표 체계)

```
Grid (전체)
 └─ Workgroup (스레드 묶음, 이 커널은 16×16 = 256개)
     └─ Thread (스레드 1개)
```

- `gl_LocalInvocationID.x/y` = **workgroup 안에서** 내 번호 (0~15)
- `gl_WorkGroupID.x/y` = 내가 **몇 번째 workgroup**인지
- **내 전역 좌표** = workgroup번호 × 크기 + local번호

이 커널은 workgroup 하나가 **출력의 16×16 타일**을 담당합니다.

---

## 2. 선언부 한 줄씩

```glsl
#version 450 core                                            // GLSL 버전 (형식적)
#extension GL_EXT_shader_explicit_arithmetic_types : enable  // fp16/int8 타입용
```

### spec 상수 — 실행 직전에 값을 꽂는 상수
```glsl
layout(constant_id = 0) const uint M = 2048;
layout(constant_id = 1) const uint N = 2048;
layout(constant_id = 2) const uint K = 2048;
layout(constant_id = 3) const uint TILE = 16;
```
코드엔 기본값만 있고, 실제 값은 파이프라인 생성 시 **호스트(C++)가 주입**합니다.

### workgroup 크기
```glsl
layout(local_size_x_id = 4, local_size_y = 1, local_size_z = 1) in;
```
스레드 몇 개짜리 묶음인지. 이 커널은 2D라 실제로는 x, y 둘 다 TILE로 설정됩니다.

### 버퍼(SSBO) — GPU 메모리의 큰 배열
```glsl
layout(set = 0, binding = 0) readonly  buffer BufA { A_TYPE a[]; };
layout(set = 0, binding = 1) readonly  buffer BufB { A_TYPE b[]; };
layout(set = 0, binding = 3) writeonly buffer BufD { C_TYPE d[]; };
```
- `binding` = 호스트가 그 버퍼를 꽂는 "슬롯 번호"
- `readonly`/`writeonly` = 읽기/쓰기 전용
- `a[]` = `a[인덱스]`로 접근. **행렬 A가 1차원으로 펼쳐져** 들어있음

> **행렬을 1차원으로 저장하는 법 (row-major):**
> 행렬 `A[i][j]` (가로 K칸) 는 배열에서 `a[i*K + j]` 위치. (i번째 행 = 앞에 i×K개 건너뜀)

### 공유메모리 (LDS)
```glsl
shared A_TYPE As[TILE * TILE];   // 16×16
shared A_TYPE Bs[TILE * TILE];
```
workgroup 안 256개 스레드가 **함께 쓰는 빠른 온칩 메모리**. A·B의 16×16 조각을 여기
올려놓고 재사용합니다.

---

## 3. 알고리즘 개념 (큰 그림)

내 목표: 출력 **`C[row][col]` 하나**. 정의상:
```
C[row][col] = A[row][0]·B[0][col] + A[row][1]·B[1][col] + ...  (K개 항)
```

A의 row 전체(K개)와 B의 col 전체(K개)를 한 번에 못 올림 → **K를 TILE(16)씩 잘라서**
조금씩 처리:

```
K를 16씩 나눠 순회:
  ① workgroup 전체가 A·B의 16×16 조각을 공유메모리에 올림 (협력)
  ② barrier (다 올릴 때까지 대기)
  ③ 나는 내 조각의 부분 내적 16개를 sum에 누적
  ④ barrier (다 계산할 때까지, 다음 조각 덮어쓰기 전에)
끝나면 C[row][col] = sum 저장
```

---

## 4. main() 코드 한 줄씩

```glsl
uint tx = gl_LocalInvocationID.x;          // workgroup 안 내 열번호 (0~15)
uint ty = gl_LocalInvocationID.y;          // workgroup 안 내 행번호 (0~15)
uint row = gl_WorkGroupID.y * TILE + ty;   // 내가 맡은 출력의 전역 행
uint col = gl_WorkGroupID.x * TILE + tx;   // 내가 맡은 출력의 전역 열
```
→ **내 좌표 확정.** 나는 `C[row][col]` 하나 담당.

```glsl
C_TYPE sum = C_TYPE(0);              // 내 출력의 누산기
uint nt = (K + TILE - 1u) / TILE;    // K를 16씩 나누면 몇 조각? (2048/16 = 128)
for (uint t = 0; t < nt; ++t) {      // 128개 조각 순회
```

```glsl
    uint a_col = t * TILE + tx;
    As[ty*TILE + tx] = a[row*K + a_col];   // A[row][t조각+tx] → 공유메모리 [ty][tx]
    uint b_row = t * TILE + ty;
    Bs[ty*TILE + tx] = b[b_row*N + col];   // B[t조각+ty][col] → 공유메모리 [ty][tx]
```
→ **협력 로드.** 256개 스레드가 각자 한 칸씩 → 16×16 조각이 통째로 올라감. (내가 올린
건 한 칸이지만, 옆 스레드들이 나머지를 채움)
*(원본엔 `row<M` 같은 경계 체크가 있음 — 크기가 TILE의 배수가 아닐 때 0으로 채우는
안전장치. 처음엔 무시해도 됨)*

```glsl
    barrier();   // ② 모두가 로드 끝낼 때까지 대기
```

```glsl
    [[unroll]] for (uint k = 0; k < TILE; ++k)
        sum += C_TYPE(As[ty*TILE + k]) * C_TYPE(Bs[k*TILE + tx]);
```
→ **부분 내적.** As의 **내 행(ty)** 을 가로로, Bs의 **내 열(tx)** 을 세로로 읽어 16개
곱을 누적. (`[[unroll]]` = "이 루프 펼쳐라"는 최적화 힌트)

```glsl
    barrier();   // ④ 모두 계산 끝낼 때까지 (다음 조각 덮어쓰기 전에)
}
```

```glsl
if (row < M && col < N)
    d[row*N + col] = sum;   // 128조각 다 끝난 sum이 최종 C[row][col]
```

---

## 5. 스레드 1개 추적 예시 (TILE=16)

**workgroup (0,0)의 스레드 (tx=2, ty=3)** 라면:

- `row = 0*16 + 3 = 3`, `col = 0*16 + 2 = 2` → **출력 `C[3][2]` 담당**
- 조각 `t=0`에서:
  - `As[3*16+2]`에 `A[3][2]` 올림
  - `Bs[3*16+2]`에 `B[3][2]` 올림
- barrier 후 내적:
  `sum += As[3][0]·Bs[0][2] + As[3][1]·Bs[1][2] + ... + As[3][15]·Bs[15][2]`
  → 즉 `A[3][0..15] · B[0..15][2]` (16개 항)
- 조각 `t=1,2,...127` 반복 → K=2048 전체 내적 완성
- 최종 `d[3*2048+2] = sum`

---

## 정리

> - 커널은 **"스레드 1개 = 출력 1개"** 관점으로 짜여 있고, `row/col`은 **내 스레드
>   ID**에서 나옵니다.
> - 행렬은 GPU 메모리에 **1차원 row-major**(`a[i*K+j]`)로 저장됩니다.
> - K를 **TILE(16)씩 잘라**, 조각마다 **① 협력 로드 → barrier → ② 부분 내적 →
>   barrier** 를 반복해 sum을 완성합니다.
> - `shared`(공유메모리)가 **재사용**의 핵심: 한 번 올린 16×16 조각을 workgroup의 256
>   스레드가 나눠 씁니다.

이 커널은 **1스레드 = 1출력, 내적** 방식입니다.

---

## 다음 단계

- **튜토리얼 2 (예정)**: `fma_v06.comp` — 1스레드가 **TM×TN개 출력**을 맡고, 내적을
  **외적(outer product)** 으로 바꿔 레지스터 재사용을 높이는 확장.
- 개념 배경: [fma-matmul-explained.ko.md](fma-matmul-explained.ko.md) (외적·재사용),
  [wmma-matmul-explained.ko.md](wmma-matmul-explained.ko.md) (WMMA 대조)
