#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "dd_f32.h"

using f32 = float;
using f64 = double;
using dd_f32 = dd::dd_f32;

static void hip_check(hipError_t e, const char* msg) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP error %s: %s (%d)\n", msg, hipGetErrorString(e), (int)e);
    std::exit(1);
  }
}

template <class Launch>
static float gpu_ms(hipStream_t stream, Launch launch, int warmup = 2, int iters = 10) {
  for (int i = 0; i < warmup; ++i) launch();
  hipEvent_t st, en;
  hip_check(hipEventCreate(&st), "event create");
  hip_check(hipEventCreate(&en), "event create");
  hip_check(hipEventRecord(st, stream), "event record");
  for (int i = 0; i < iters; ++i) launch();
  hip_check(hipEventRecord(en, stream), "event record");
  hip_check(hipEventSynchronize(en), "event sync");
  float ms = 0;
  hip_check(hipEventElapsedTime(&ms, st, en), "event elapsed");
  hip_check(hipEventDestroy(st), "event destroy");
  hip_check(hipEventDestroy(en), "event destroy");
  return ms / (float)iters;
}

__device__ __forceinline__ f32 chain_f32(f32 x, f32 y, int iters) {
  for (int k = 0; k < iters; ++k) {
    x = fmaf(x, 1.0001f, y);
    y = fmaf(-y, 0.99991f, x);
    f32 t = fmaf(x, y, 0.1f);
    x = t / (fabsf(y) + 1.0f);
    x = sqrtf(fabsf(x) + 0.001f);
    f32 s, c;
    sincosf(x, &s, &c);
    x = s + c;
    x = logf(fabsf(x) + 1.001f);
    x = expf(x * 0.1f) - 1.0f;
    x = rsqrtf(fabsf(x) + 1.0f) * x;
  }
  return x + y;
}

__device__ __forceinline__ f64 chain_f64(f64 x, f64 y, int iters) {
  for (int k = 0; k < iters; ++k) {
    x = fma(x, 1.0001, y);
    y = fma(-y, 0.99991, x);
    f64 t = fma(x, y, 0.1);
    x = t / (fabs(y) + 1.0);
    x = sqrt(fabs(x) + 0.001);
    f64 s, c;
    sincos(x, &s, &c);
    x = s + c;
    x = log(fabs(x) + 1.001);
    x = exp(x * 0.1) - 1.0;
    x = (1.0 / sqrt(fabs(x) + 1.0)) * x;
  }
  return x + y;
}

__host__ __device__ __forceinline__ dd_f32 dd_abs(dd_f32 a) {
  return (a.hi < 0.0f) ? dd::dd_neg(a) : a;
}

__device__ __forceinline__ dd_f32 chain_dd(dd_f32 x, dd_f32 y, int iters) {
  for (int k = 0; k < iters; ++k) {
    x = dd::dd_add(dd::dd_mul(x, 1.0001f), y);
    y = dd::dd_add(dd::dd_mul(y, -0.99991f), x);
    dd_f32 t = dd::dd_add(dd::dd_mul(x, y), 0.1f);
    x = dd::dd_div(t, dd::dd_add(dd_abs(y), 1.0f));
    x = dd::dd_sqrt(dd::dd_add(dd_abs(x), 0.001f));
    dd_f32 s, c;
    dd::dd_sincos(x, &s, &c);
    x = dd::dd_add(s, c);
    x = dd::dd_ln(dd::dd_add(dd_abs(x), 1.001f));
    x = dd::dd_sub(dd::dd_exp(dd::dd_mul(x, 0.1f)), 1.0f);
    x = dd::dd_mul(dd::dd_inv_sqrt(dd::dd_add(dd_abs(x), 1.0f)), x);
  }
  return dd::dd_add(x, y);
}

__global__ void k_complex_f32(const f32* a, const f32* b, f32* out, int n, int chain_iters) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  out[i] = chain_f32(a[i], b[i], chain_iters);
}

__global__ void k_complex_f64(const f64* a, const f64* b, f64* out, int n, int chain_iters) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  out[i] = chain_f64(a[i], b[i], chain_iters);
}

__global__ void k_complex_dd(const dd_f32* a, const dd_f32* b, dd_f32* out, int n, int chain_iters) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= n) return;
  out[i] = chain_dd(a[i], b[i], chain_iters);
}

static volatile double g_sink = 0.0;

int main(int argc, char** argv) {
  int n = 10'000'000;
  int chain_iters = 64;

  if (argc >= 2) n = std::max(1, std::atoi(argv[1]));
  if (argc >= 3) chain_iters = std::max(1, std::atoi(argv[2]));

  hipDeviceProp_t prop{};
  hip_check(hipGetDeviceProperties(&prop, 0), "get device props");
  std::printf("GPU: %s\n", prop.name);
  std::printf("Complex chain benchmark (fp32, fp64, dd_f32)\n");
  std::printf("N=%d, chain_iters=%d\n", n, chain_iters);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

  std::vector<f32> ha(n), hb(n);
  for (int i = 0; i < n; ++i) {
    ha[i] = dist(rng);
    hb[i] = dist(rng);
    if (fabsf(hb[i]) < 1e-3f) hb[i] = 1.0f;
  }

  std::vector<f64> ha64(n), hb64(n);
  std::vector<dd_f32> hadd(n), hbdd(n);
  for (int i = 0; i < n; ++i) {
    ha64[i] = (f64)ha[i];
    hb64[i] = (f64)hb[i];
    hadd[i] = dd::dd_f32_from_f32(ha[i]);
    hbdd[i] = dd::dd_f32_from_f32(hb[i]);
  }

  f32 *da = nullptr, *db = nullptr, *do32 = nullptr;
  f64 *da64 = nullptr, *db64 = nullptr, *do64 = nullptr;
  dd_f32 *dadd = nullptr, *dbdd = nullptr, *dodd = nullptr;

  hip_check(hipMalloc(&da, n * sizeof(f32)), "malloc");
  hip_check(hipMalloc(&db, n * sizeof(f32)), "malloc");
  hip_check(hipMalloc(&do32, n * sizeof(f32)), "malloc");

  hip_check(hipMalloc(&da64, n * sizeof(f64)), "malloc");
  hip_check(hipMalloc(&db64, n * sizeof(f64)), "malloc");
  hip_check(hipMalloc(&do64, n * sizeof(f64)), "malloc");

  hip_check(hipMalloc(&dadd, n * sizeof(dd_f32)), "malloc");
  hip_check(hipMalloc(&dbdd, n * sizeof(dd_f32)), "malloc");
  hip_check(hipMalloc(&dodd, n * sizeof(dd_f32)), "malloc");

  hip_check(hipMemcpy(da, ha.data(), n * sizeof(f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(db, hb.data(), n * sizeof(f32), hipMemcpyHostToDevice), "cpy");

  hip_check(hipMemcpy(da64, ha64.data(), n * sizeof(f64), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(db64, hb64.data(), n * sizeof(f64), hipMemcpyHostToDevice), "cpy");

  hip_check(hipMemcpy(dadd, hadd.data(), n * sizeof(dd_f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(dbdd, hbdd.data(), n * sizeof(dd_f32), hipMemcpyHostToDevice), "cpy");

  hipStream_t stream;
  hip_check(hipStreamCreate(&stream), "stream");

  int block = 256;
  int grid = (n + block - 1) / block;

  auto ms32 = gpu_ms(stream, [&] {
    hipLaunchKernelGGL(k_complex_f32, dim3(grid), dim3(block), 0, stream, da, db, do32, n, chain_iters);
  });
  hip_check(hipStreamSynchronize(stream), "sync");

  auto ms64 = gpu_ms(stream, [&] {
    hipLaunchKernelGGL(k_complex_f64, dim3(grid), dim3(block), 0, stream, da64, db64, do64, n, chain_iters);
  });
  hip_check(hipStreamSynchronize(stream), "sync");

  auto msdd = gpu_ms(stream, [&] {
    hipLaunchKernelGGL(k_complex_dd, dim3(grid), dim3(block), 0, stream, dadd, dbdd, dodd, n, chain_iters);
  });
  hip_check(hipStreamSynchronize(stream), "sync");

  f32 one32 = 0;
  f64 one64 = 0;
  dd_f32 onedd{};
  hip_check(hipMemcpy(&one32, do32, sizeof(f32), hipMemcpyDeviceToHost), "cpy");
  hip_check(hipMemcpy(&one64, do64, sizeof(f64), hipMemcpyDeviceToHost), "cpy");
  hip_check(hipMemcpy(&onedd, dodd, sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");
  g_sink += (double)one32 + (double)one64 + (double)dd::dd_f32_as_f64(onedd);

  const double evals = (double)n * (double)chain_iters;
  auto geval = [&](float ms) {
    double s = (double)ms * 1e-3;
    return evals / s / 1e9;
  };

  std::printf("Throughput (GEval/s): fp32=%.3f  fp64=%.3f  dd_f32=%.3f\n",
              geval(ms32), geval(ms64), geval(msdd));
  std::printf("Kernel time (ms):      fp32=%.3f  fp64=%.3f  dd_f32=%.3f\n", ms32, ms64, msdd);

  hip_check(hipStreamDestroy(stream), "stream destroy");
  hip_check(hipFree(da), "free");
  hip_check(hipFree(db), "free");
  hip_check(hipFree(do32), "free");
  hip_check(hipFree(da64), "free");
  hip_check(hipFree(db64), "free");
  hip_check(hipFree(do64), "free");
  hip_check(hipFree(dadd), "free");
  hip_check(hipFree(dbdd), "free");
  hip_check(hipFree(dodd), "free");

  if (g_sink == 0.123) std::printf("sink=%f\n", g_sink);
  return 0;
}
