#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "dd_f32.h"

using dd::dd_f32;
using dd::f32;
using dd::f64;

#ifndef DD_N
#define DD_N 100000
#endif

#ifndef DD_ITERS
#define DD_ITERS 1024
#endif

#ifndef DD_ILP
#define DD_ILP 4
#endif

static void hip_check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP error (%s): %s\n", what, hipGetErrorString(e));
    std::exit(1);
  }
}

template <typename T>
__global__ void kernel_add(const T* a, const T* b, T* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

__global__ void kernel_add_dd(const dd_f32* a, const dd_f32* b, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dd::dd_add(a[i], b[i]);
}

template <typename T>
__global__ void kernel_mul(const T* a, const T* b, T* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] * b[i];
}

__global__ void kernel_mul_dd(const dd_f32* a, const dd_f32* b, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dd::dd_mul(a[i], b[i]);
}

template <typename T>
__global__ void kernel_div(const T* a, const T* b, T* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] / b[i];
}

__global__ void kernel_div_dd(const dd_f32* a, const dd_f32* b, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dd::dd_div(a[i], b[i]);
}

__global__ void kernel_sqrt_f32(const f32* a, f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sqrtf(a[i]);
}

__global__ void kernel_sqrt_f64(const f64* a, f64* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sqrt(a[i]);
}

__global__ void kernel_sqrt_dd(const dd_f32* a, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dd::dd_sqrt(a[i]);
}

__global__ void kernel_ln_f32(const f32* a, f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = logf(a[i]);
}

__global__ void kernel_ln_f64(const f64* a, f64* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = log(a[i]);
}

__global__ void kernel_ln_dd(const dd_f32* a, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dd::dd_ln(a[i]);
}

__global__ void kernel_exp_f32(const f32* a, f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = expf(a[i]);
}

__global__ void kernel_exp_f64(const f64* a, f64* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = exp(a[i]);
}

__global__ void kernel_exp_dd(const dd_f32* a, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dd::dd_exp(a[i]);
}

__global__ void kernel_sin_f32(const f32* a, f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sinf(a[i]);
}

__global__ void kernel_sin_f64(const f64* a, f64* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = sin(a[i]);
}

__global__ void kernel_sin_dd(const dd_f32* a, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dd::dd_sin(a[i]);
}

__global__ void kernel_cos_f32(const f32* a, f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cosf(a[i]);
}

__global__ void kernel_cos_f64(const f64* a, f64* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cos(a[i]);
}

__global__ void kernel_cos_dd(const dd_f32* a, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = dd::dd_cos(a[i]);
}

// ---------------- Perf kernels (loop to amortize launch/memory) --------------

template <typename T>
__global__ void kernel_add_perf(const T* a, const T* b, T* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  T x0 = a[i];
  T x1 = x0 + (T)1;
  T x2 = x0 + (T)2;
  T x3 = x0 + (T)3;
  T y = b[i];
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = x0 + y;
    x1 = x1 + y;
    x2 = x2 + y;
    x3 = x3 + y;
  }
  out[i] = x0 + x1 + x2 + x3;
}

__global__ void kernel_add_dd_perf(const dd_f32* a, const dd_f32* b, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  dd_f32 x1 = dd::dd_add(x0, (f32)1);
  dd_f32 x2 = dd::dd_add(x0, (f32)2);
  dd_f32 x3 = dd::dd_add(x0, (f32)3);
  dd_f32 y = b[i];
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_add(x0, y);
    x1 = dd::dd_add(x1, y);
    x2 = dd::dd_add(x2, y);
    x3 = dd::dd_add(x3, y);
  }
  out[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

template <typename T>
__global__ void kernel_mul_perf(const T* a, const T* b, T* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  T x0 = a[i];
  T x1 = x0 + (T)1;
  T x2 = x0 + (T)2;
  T x3 = x0 + (T)3;
  T y = b[i];
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = x0 * y;
    x1 = x1 * y;
    x2 = x2 * y;
    x3 = x3 * y;
  }
  out[i] = x0 + x1 + x2 + x3;
}

__global__ void kernel_mul_dd_perf(const dd_f32* a, const dd_f32* b, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  dd_f32 x1 = dd::dd_add(x0, (f32)1);
  dd_f32 x2 = dd::dd_add(x0, (f32)2);
  dd_f32 x3 = dd::dd_add(x0, (f32)3);
  dd_f32 y = b[i];
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_mul(x0, y);
    x1 = dd::dd_mul(x1, y);
    x2 = dd::dd_mul(x2, y);
    x3 = dd::dd_mul(x3, y);
  }
  out[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

template <typename T>
__global__ void kernel_div_perf(const T* a, const T* b, T* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  T x0 = a[i];
  T x1 = x0 + (T)1;
  T x2 = x0 + (T)2;
  T x3 = x0 + (T)3;
  T y = b[i];
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = x0 / y;
    x1 = x1 / y;
    x2 = x2 / y;
    x3 = x3 / y;
  }
  out[i] = x0 + x1 + x2 + x3;
}

__global__ void kernel_div_dd_perf(const dd_f32* a, const dd_f32* b, dd_f32* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  dd_f32 x1 = dd::dd_add(x0, (f32)1);
  dd_f32 x2 = dd::dd_add(x0, (f32)2);
  dd_f32 x3 = dd::dd_add(x0, (f32)3);
  dd_f32 y = b[i];
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_div(x0, y);
    x1 = dd::dd_div(x1, y);
    x2 = dd::dd_div(x2, y);
    x3 = dd::dd_div(x3, y);
  }
  out[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
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

static double rel_bits(long double ref, long double got) {
  long double denom = fabsl(ref);
  long double err = fabsl(got - ref);
  if (denom == 0.0L) return (err == 0.0L) ? 200.0 : 0.0;
  long double rel = err / denom;
  if (rel == 0.0L) return 200.0;
  return -std::log2((double)rel);
}

static const char* fmt_bits(double bits, char out[16]) {
  if (bits >= 199.0) {
    std::snprintf(out, 16, "exact");
  } else {
    std::snprintf(out, 16, "%.2f", bits);
  }
  return out;
}

int main(int argc, char** argv) {
  int n = DD_N;
  if (argc >= 2) n = std::max(1, std::atoi(argv[1]));

  hipDeviceProp_t prop{};
  hip_check(hipGetDeviceProperties(&prop, 0), "get device props");
  std::printf("GPU: %s\n", prop.name);
  std::printf("N=%d, ITERS=%d\n", n, (int)DD_ITERS);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  std::uniform_real_distribution<float> dist_exp(-10.0f, 10.0f);
  std::uniform_real_distribution<float> dist_pos(1e-3f, 1000.0f);

  std::vector<f32> ha(n), hb(n);
  for (int i = 0; i < n; ++i) {
    ha[i] = dist(rng);
    hb[i] = dist(rng);
    if (fabsf(hb[i]) < 1e-3f) hb[i] = 1.0f;
  }

  // Variants for ops that require positivity / limited dynamic range.
  std::vector<f32> ha_pos(n);
  std::vector<f32> ha_exp(n);
  for (int i = 0; i < n; ++i) {
    ha_pos[i] = dist_pos(rng);
    ha_exp[i] = dist_exp(rng);
  }

  std::vector<dd_f32> ha_dd(n), hb_dd(n), ha_pos_dd(n), ha_exp_dd(n);
  for (int i = 0; i < n; ++i) {
    ha_dd[i] = dd::dd_f32_from_f32(ha[i]);
    hb_dd[i] = dd::dd_f32_from_f32(hb[i]);
    ha_pos_dd[i] = dd::dd_f32_from_f32(ha_pos[i]);
    ha_exp_dd[i] = dd::dd_f32_from_f32(ha_exp[i]);
  }

  // Device buffers
  f32 *da_f32, *db_f32, *do_f32;
  f64 *da_f64, *db_f64, *do_f64;
  dd_f32 *da_dd, *db_dd, *do_dd;
  f32 *da_pos_f32;
  f64 *da_pos_f64;
  dd_f32 *da_pos_dd;
  f32 *da_exp_f32;
  f64 *da_exp_f64;
  dd_f32 *da_exp_dd;

  hip_check(hipMalloc(&da_f32, n * sizeof(f32)), "malloc f32 a");
  hip_check(hipMalloc(&db_f32, n * sizeof(f32)), "malloc f32 b");
  hip_check(hipMalloc(&do_f32, n * sizeof(f32)), "malloc f32 out");
  hip_check(hipMalloc(&da_f64, n * sizeof(f64)), "malloc f64 a");
  hip_check(hipMalloc(&db_f64, n * sizeof(f64)), "malloc f64 b");
  hip_check(hipMalloc(&do_f64, n * sizeof(f64)), "malloc f64 out");
  hip_check(hipMalloc(&da_dd, n * sizeof(dd_f32)), "malloc dd a");
  hip_check(hipMalloc(&db_dd, n * sizeof(dd_f32)), "malloc dd b");
  hip_check(hipMalloc(&do_dd, n * sizeof(dd_f32)), "malloc dd out");

  hip_check(hipMalloc(&da_pos_f32, n * sizeof(f32)), "malloc f32 a_pos");
  hip_check(hipMalloc(&da_pos_f64, n * sizeof(f64)), "malloc f64 a_pos");
  hip_check(hipMalloc(&da_pos_dd, n * sizeof(dd_f32)), "malloc dd a_pos");
  hip_check(hipMalloc(&da_exp_f32, n * sizeof(f32)), "malloc f32 a_exp");
  hip_check(hipMalloc(&da_exp_f64, n * sizeof(f64)), "malloc f64 a_exp");
  hip_check(hipMalloc(&da_exp_dd, n * sizeof(dd_f32)), "malloc dd a_exp");

  hip_check(hipMemcpy(da_f32, ha.data(), n * sizeof(f32), hipMemcpyHostToDevice), "cpy f32 a");
  hip_check(hipMemcpy(db_f32, hb.data(), n * sizeof(f32), hipMemcpyHostToDevice), "cpy f32 b");

  std::vector<f64> ha64(n), hb64(n), ha_pos64(n), ha_exp64(n);
  for (int i = 0; i < n; ++i) {
    ha64[i] = (f64)ha[i];
    hb64[i] = (f64)hb[i];
    ha_pos64[i] = (f64)ha_pos[i];
    ha_exp64[i] = (f64)ha_exp[i];
  }
  hip_check(hipMemcpy(da_f64, ha64.data(), n * sizeof(f64), hipMemcpyHostToDevice), "cpy f64 a");
  hip_check(hipMemcpy(db_f64, hb64.data(), n * sizeof(f64), hipMemcpyHostToDevice), "cpy f64 b");

  hip_check(hipMemcpy(da_dd, ha_dd.data(), n * sizeof(dd_f32), hipMemcpyHostToDevice), "cpy dd a");
  hip_check(hipMemcpy(db_dd, hb_dd.data(), n * sizeof(dd_f32), hipMemcpyHostToDevice), "cpy dd b");

  hip_check(hipMemcpy(da_pos_f32, ha_pos.data(), n * sizeof(f32), hipMemcpyHostToDevice), "cpy f32 a_pos");
  hip_check(hipMemcpy(da_pos_f64, ha_pos64.data(), n * sizeof(f64), hipMemcpyHostToDevice), "cpy f64 a_pos");
  hip_check(hipMemcpy(da_pos_dd, ha_pos_dd.data(), n * sizeof(dd_f32), hipMemcpyHostToDevice), "cpy dd a_pos");
  hip_check(hipMemcpy(da_exp_f32, ha_exp.data(), n * sizeof(f32), hipMemcpyHostToDevice), "cpy f32 a_exp");
  hip_check(hipMemcpy(da_exp_f64, ha_exp64.data(), n * sizeof(f64), hipMemcpyHostToDevice), "cpy f64 a_exp");
  hip_check(hipMemcpy(da_exp_dd, ha_exp_dd.data(), n * sizeof(dd_f32), hipMemcpyHostToDevice), "cpy dd a_exp");

  hipStream_t stream;
  hip_check(hipStreamCreate(&stream), "stream create");

  int block = 256;
  int grid = (n + block - 1) / block;

  // ---------------- Accuracy checks -----------------------------------------

  auto launch_add_f32 = [&]() {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_add<f32>), dim3(grid), dim3(block), 0, stream, da_f32, db_f32, do_f32, n);
    hip_check(hipStreamSynchronize(stream), "sync add f32");
  };
  auto launch_add_f64 = [&]() {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_add<f64>), dim3(grid), dim3(block), 0, stream, da_f64, db_f64, do_f64, n);
    hip_check(hipStreamSynchronize(stream), "sync add f64");
  };
  auto launch_add_dd = [&]() {
    hipLaunchKernelGGL(kernel_add_dd, dim3(grid), dim3(block), 0, stream, da_dd, db_dd, do_dd, n);
    hip_check(hipStreamSynchronize(stream), "sync add dd");
  };

  auto launch_mul_f32 = [&]() {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_mul<f32>), dim3(grid), dim3(block), 0, stream, da_f32, db_f32, do_f32, n);
    hip_check(hipStreamSynchronize(stream), "sync mul f32");
  };
  auto launch_mul_f64 = [&]() {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_mul<f64>), dim3(grid), dim3(block), 0, stream, da_f64, db_f64, do_f64, n);
    hip_check(hipStreamSynchronize(stream), "sync mul f64");
  };
  auto launch_mul_dd = [&]() {
    hipLaunchKernelGGL(kernel_mul_dd, dim3(grid), dim3(block), 0, stream, da_dd, db_dd, do_dd, n);
    hip_check(hipStreamSynchronize(stream), "sync mul dd");
  };

  auto launch_div_f32 = [&]() {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_div<f32>), dim3(grid), dim3(block), 0, stream, da_f32, db_f32, do_f32, n);
    hip_check(hipStreamSynchronize(stream), "sync div f32");
  };
  auto launch_div_f64 = [&]() {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_div<f64>), dim3(grid), dim3(block), 0, stream, da_f64, db_f64, do_f64, n);
    hip_check(hipStreamSynchronize(stream), "sync div f64");
  };
  auto launch_div_dd = [&]() {
    hipLaunchKernelGGL(kernel_div_dd, dim3(grid), dim3(block), 0, stream, da_dd, db_dd, do_dd, n);
    hip_check(hipStreamSynchronize(stream), "sync div dd");
  };

  {
    auto ref = [](long double a, long double b) { return a + b; };
    std::vector<f32> out_f32(n);
    std::vector<f64> out_f64(n);
    std::vector<dd_f32> out_dd(n);
    launch_add_f32(); hip_check(hipMemcpy(out_f32.data(), do_f32, n * sizeof(f32), hipMemcpyDeviceToHost), "cpy");
    launch_add_f64(); hip_check(hipMemcpy(out_f64.data(), do_f64, n * sizeof(f64), hipMemcpyDeviceToHost), "cpy");
    launch_add_dd();  hip_check(hipMemcpy(out_dd.data(),  do_dd, n * sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");
    double minf32=1e9,minf64=1e9,mindd=1e9;
    for(int i=0;i<n;++i){
      long double r=ref((long double)ha[i],(long double)hb[i]);
      minf32=std::min(minf32,rel_bits(r,(long double)out_f32[i]));
      minf64=std::min(minf64,rel_bits(r,(long double)out_f64[i]));
      mindd=std::min(mindd,rel_bits(r,(long double)dd::dd_f32_as_f64(out_dd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Accuracy (min rel bits) add   | f32: %6s  f64: %6s  dd_f32: %6s\n",
                fmt_bits(minf32, b32), fmt_bits(minf64, b64), fmt_bits(mindd, bdd));
  }
  {
    auto ref = [](long double a, long double b) { return a * b; };
    std::vector<f32> out_f32(n);
    std::vector<f64> out_f64(n);
    std::vector<dd_f32> out_dd(n);
    launch_mul_f32(); hip_check(hipMemcpy(out_f32.data(), do_f32, n * sizeof(f32), hipMemcpyDeviceToHost), "cpy");
    launch_mul_f64(); hip_check(hipMemcpy(out_f64.data(), do_f64, n * sizeof(f64), hipMemcpyDeviceToHost), "cpy");
    launch_mul_dd();  hip_check(hipMemcpy(out_dd.data(),  do_dd, n * sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");
    double minf32=1e9,minf64=1e9,mindd=1e9;
    for(int i=0;i<n;++i){
      long double r=ref((long double)ha[i],(long double)hb[i]);
      minf32=std::min(minf32,rel_bits(r,(long double)out_f32[i]));
      minf64=std::min(minf64,rel_bits(r,(long double)out_f64[i]));
      mindd=std::min(mindd,rel_bits(r,(long double)dd::dd_f32_as_f64(out_dd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Accuracy (min rel bits) mul   | f32: %6s  f64: %6s  dd_f32: %6s\n",
                fmt_bits(minf32, b32), fmt_bits(minf64, b64), fmt_bits(mindd, bdd));
  }
  {
    auto ref = [](long double a, long double b) { return a / b; };
    std::vector<f32> out_f32(n);
    std::vector<f64> out_f64(n);
    std::vector<dd_f32> out_dd(n);
    launch_div_f32(); hip_check(hipMemcpy(out_f32.data(), do_f32, n * sizeof(f32), hipMemcpyDeviceToHost), "cpy");
    launch_div_f64(); hip_check(hipMemcpy(out_f64.data(), do_f64, n * sizeof(f64), hipMemcpyDeviceToHost), "cpy");
    launch_div_dd();  hip_check(hipMemcpy(out_dd.data(),  do_dd, n * sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");
    double minf32=1e9,minf64=1e9,mindd=1e9;
    for(int i=0;i<n;++i){
      long double r=ref((long double)ha[i],(long double)hb[i]);
      minf32=std::min(minf32,rel_bits(r,(long double)out_f32[i]));
      minf64=std::min(minf64,rel_bits(r,(long double)out_f64[i]));
      mindd=std::min(mindd,rel_bits(r,(long double)dd::dd_f32_as_f64(out_dd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Accuracy (min rel bits) div   | f32: %6s  f64: %6s  dd_f32: %6s\n",
                fmt_bits(minf32, b32), fmt_bits(minf64, b64), fmt_bits(mindd, bdd));
  }

  // sqrt accuracy on positive range
  {
    std::vector<f32> out_f32(n);
    std::vector<f64> out_f64(n);
    std::vector<dd_f32> out_dd(n);

    hipLaunchKernelGGL(kernel_sqrt_f32, dim3(grid), dim3(block), 0, stream, da_pos_f32, do_f32, n);
    hip_check(hipStreamSynchronize(stream), "sync sqrt f32");
    hip_check(hipMemcpy(out_f32.data(), do_f32, n * sizeof(f32), hipMemcpyDeviceToHost), "cpy");

    hipLaunchKernelGGL(kernel_sqrt_f64, dim3(grid), dim3(block), 0, stream, da_pos_f64, do_f64, n);
    hip_check(hipStreamSynchronize(stream), "sync sqrt f64");
    hip_check(hipMemcpy(out_f64.data(), do_f64, n * sizeof(f64), hipMemcpyDeviceToHost), "cpy");

    hipLaunchKernelGGL(kernel_sqrt_dd, dim3(grid), dim3(block), 0, stream, da_pos_dd, do_dd, n);
    hip_check(hipStreamSynchronize(stream), "sync sqrt dd");
    hip_check(hipMemcpy(out_dd.data(), do_dd, n * sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");

    double minf32=1e9,minf64=1e9,mindd=1e9;
    for (int i = 0; i < n; ++i) {
      long double ref = sqrtl((long double)ha_pos[i]);
      minf32 = std::min(minf32, rel_bits(ref, (long double)out_f32[i]));
      minf64 = std::min(minf64, rel_bits(ref, (long double)out_f64[i]));
      mindd  = std::min(mindd,  rel_bits(ref, (long double)dd::dd_f32_as_f64(out_dd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Accuracy (min rel bits) sqrt  | f32: %6s  f64: %6s  dd_f32: %6s\n",
                fmt_bits(minf32, b32), fmt_bits(minf64, b64), fmt_bits(mindd, bdd));
  }

  // ln / exp / sin / cos accuracy (correctness-first paths)
  {
    std::vector<f32> out_f32(n);
    std::vector<f64> out_f64(n);
    std::vector<dd_f32> out_dd(n);

    hipLaunchKernelGGL(kernel_ln_f32, dim3(grid), dim3(block), 0, stream, da_pos_f32, do_f32, n);
    hip_check(hipStreamSynchronize(stream), "sync ln f32");
    hip_check(hipMemcpy(out_f32.data(), do_f32, n * sizeof(f32), hipMemcpyDeviceToHost), "cpy");

    hipLaunchKernelGGL(kernel_ln_f64, dim3(grid), dim3(block), 0, stream, da_pos_f64, do_f64, n);
    hip_check(hipStreamSynchronize(stream), "sync ln f64");
    hip_check(hipMemcpy(out_f64.data(), do_f64, n * sizeof(f64), hipMemcpyDeviceToHost), "cpy");

    hipLaunchKernelGGL(kernel_ln_dd, dim3(grid), dim3(block), 0, stream, da_pos_dd, do_dd, n);
    hip_check(hipStreamSynchronize(stream), "sync ln dd");
    hip_check(hipMemcpy(out_dd.data(), do_dd, n * sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");

    double minf32=1e9,minf64=1e9,mindd=1e9;
    for (int i = 0; i < n; ++i) {
      long double ref = logl((long double)ha_pos[i]);
      minf32 = std::min(minf32, rel_bits(ref, (long double)out_f32[i]));
      minf64 = std::min(minf64, rel_bits(ref, (long double)out_f64[i]));
      mindd  = std::min(mindd,  rel_bits(ref, (long double)dd::dd_f32_as_f64(out_dd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Accuracy (min rel bits) ln    | f32: %6s  f64: %6s  dd_f32: %6s\n",
                fmt_bits(minf32, b32), fmt_bits(minf64, b64), fmt_bits(mindd, bdd));
  }

  {
    std::vector<f32> out_f32(n);
    std::vector<f64> out_f64(n);
    std::vector<dd_f32> out_dd(n);

    hipLaunchKernelGGL(kernel_exp_f32, dim3(grid), dim3(block), 0, stream, da_exp_f32, do_f32, n);
    hip_check(hipStreamSynchronize(stream), "sync exp f32");
    hip_check(hipMemcpy(out_f32.data(), do_f32, n * sizeof(f32), hipMemcpyDeviceToHost), "cpy");

    hipLaunchKernelGGL(kernel_exp_f64, dim3(grid), dim3(block), 0, stream, da_exp_f64, do_f64, n);
    hip_check(hipStreamSynchronize(stream), "sync exp f64");
    hip_check(hipMemcpy(out_f64.data(), do_f64, n * sizeof(f64), hipMemcpyDeviceToHost), "cpy");

    hipLaunchKernelGGL(kernel_exp_dd, dim3(grid), dim3(block), 0, stream, da_exp_dd, do_dd, n);
    hip_check(hipStreamSynchronize(stream), "sync exp dd");
    hip_check(hipMemcpy(out_dd.data(), do_dd, n * sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");

    double minf32=1e9,minf64=1e9,mindd=1e9;
    for (int i = 0; i < n; ++i) {
      long double ref = expl((long double)ha_exp[i]);
      minf32 = std::min(minf32, rel_bits(ref, (long double)out_f32[i]));
      minf64 = std::min(minf64, rel_bits(ref, (long double)out_f64[i]));
      mindd  = std::min(mindd,  rel_bits(ref, (long double)dd::dd_f32_as_f64(out_dd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Accuracy (min rel bits) exp   | f32: %6s  f64: %6s  dd_f32: %6s\n",
                fmt_bits(minf32, b32), fmt_bits(minf64, b64), fmt_bits(mindd, bdd));
  }

  {
    std::vector<f32> out_f32(n);
    std::vector<f64> out_f64(n);
    std::vector<dd_f32> out_dd(n);

    hipLaunchKernelGGL(kernel_sin_f32, dim3(grid), dim3(block), 0, stream, da_f32, do_f32, n);
    hip_check(hipStreamSynchronize(stream), "sync sin f32");
    hip_check(hipMemcpy(out_f32.data(), do_f32, n * sizeof(f32), hipMemcpyDeviceToHost), "cpy");

    hipLaunchKernelGGL(kernel_sin_f64, dim3(grid), dim3(block), 0, stream, da_f64, do_f64, n);
    hip_check(hipStreamSynchronize(stream), "sync sin f64");
    hip_check(hipMemcpy(out_f64.data(), do_f64, n * sizeof(f64), hipMemcpyDeviceToHost), "cpy");

    hipLaunchKernelGGL(kernel_sin_dd, dim3(grid), dim3(block), 0, stream, da_dd, do_dd, n);
    hip_check(hipStreamSynchronize(stream), "sync sin dd");
    hip_check(hipMemcpy(out_dd.data(), do_dd, n * sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");

    double minf32=1e9,minf64=1e9,mindd=1e9;
    for (int i = 0; i < n; ++i) {
      long double ref = sinl((long double)ha[i]);
      minf32 = std::min(minf32, rel_bits(ref, (long double)out_f32[i]));
      minf64 = std::min(minf64, rel_bits(ref, (long double)out_f64[i]));
      mindd  = std::min(mindd,  rel_bits(ref, (long double)dd::dd_f32_as_f64(out_dd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Accuracy (min rel bits) sin   | f32: %6s  f64: %6s  dd_f32: %6s\n",
                fmt_bits(minf32, b32), fmt_bits(minf64, b64), fmt_bits(mindd, bdd));
  }

  {
    std::vector<f32> out_f32(n);
    std::vector<f64> out_f64(n);
    std::vector<dd_f32> out_dd(n);

    hipLaunchKernelGGL(kernel_cos_f32, dim3(grid), dim3(block), 0, stream, da_f32, do_f32, n);
    hip_check(hipStreamSynchronize(stream), "sync cos f32");
    hip_check(hipMemcpy(out_f32.data(), do_f32, n * sizeof(f32), hipMemcpyDeviceToHost), "cpy");

    hipLaunchKernelGGL(kernel_cos_f64, dim3(grid), dim3(block), 0, stream, da_f64, do_f64, n);
    hip_check(hipStreamSynchronize(stream), "sync cos f64");
    hip_check(hipMemcpy(out_f64.data(), do_f64, n * sizeof(f64), hipMemcpyDeviceToHost), "cpy");

    hipLaunchKernelGGL(kernel_cos_dd, dim3(grid), dim3(block), 0, stream, da_dd, do_dd, n);
    hip_check(hipStreamSynchronize(stream), "sync cos dd");
    hip_check(hipMemcpy(out_dd.data(), do_dd, n * sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");

    double minf32=1e9,minf64=1e9,mindd=1e9;
    for (int i = 0; i < n; ++i) {
      long double ref = cosl((long double)ha[i]);
      minf32 = std::min(minf32, rel_bits(ref, (long double)out_f32[i]));
      minf64 = std::min(minf64, rel_bits(ref, (long double)out_f64[i]));
      mindd  = std::min(mindd,  rel_bits(ref, (long double)dd::dd_f32_as_f64(out_dd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Accuracy (min rel bits) cos   | f32: %6s  f64: %6s  dd_f32: %6s\n",
                fmt_bits(minf32, b32), fmt_bits(minf64, b64), fmt_bits(mindd, bdd));
  }

  // ---------------- Performance ---------------------------------------------
  auto print_perf = [&](const char* name, float ms_f32, float ms_f64, float ms_dd) {
    const double ops = (double)n * (double)DD_ITERS * (double)DD_ILP;
    double t32 = (double)ms_f32 * 1e-3;
    double t64 = (double)ms_f64 * 1e-3;
    double tdd = (double)ms_dd  * 1e-3;
    double gops32 = ops / t32 / 1e9;
    double gops64 = ops / t64 / 1e9;
    double gopsdd = ops / tdd / 1e9;
    std::printf("Perf %-4s | f32: %8.2f Gop/s  f64: %8.2f Gop/s  dd_f32: %8.2f Gop/s  (kernel ms: %.3f / %.3f / %.3f)\n",
                name, gops32, gops64, gopsdd, ms_f32, ms_f64, ms_dd);
  };

  auto launch_add_perf_f32 = [&](hipStream_t s) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_add_perf<f32>), dim3(grid), dim3(block), 0, s, da_f32, db_f32, do_f32, n);
  };
  auto launch_add_perf_f64 = [&](hipStream_t s) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_add_perf<f64>), dim3(grid), dim3(block), 0, s, da_f64, db_f64, do_f64, n);
  };
  auto launch_add_perf_dd = [&](hipStream_t s) {
    hipLaunchKernelGGL(kernel_add_dd_perf, dim3(grid), dim3(block), 0, s, da_dd, db_dd, do_dd, n);
  };

  auto launch_mul_perf_f32 = [&](hipStream_t s) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_mul_perf<f32>), dim3(grid), dim3(block), 0, s, da_f32, db_f32, do_f32, n);
  };
  auto launch_mul_perf_f64 = [&](hipStream_t s) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_mul_perf<f64>), dim3(grid), dim3(block), 0, s, da_f64, db_f64, do_f64, n);
  };
  auto launch_mul_perf_dd = [&](hipStream_t s) {
    hipLaunchKernelGGL(kernel_mul_dd_perf, dim3(grid), dim3(block), 0, s, da_dd, db_dd, do_dd, n);
  };

  auto launch_div_perf_f32 = [&](hipStream_t s) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_div_perf<f32>), dim3(grid), dim3(block), 0, s, da_f32, db_f32, do_f32, n);
  };
  auto launch_div_perf_f64 = [&](hipStream_t s) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel_div_perf<f64>), dim3(grid), dim3(block), 0, s, da_f64, db_f64, do_f64, n);
  };
  auto launch_div_perf_dd = [&](hipStream_t s) {
    hipLaunchKernelGGL(kernel_div_dd_perf, dim3(grid), dim3(block), 0, s, da_dd, db_dd, do_dd, n);
  };

  auto ms_add_f32 = gpu_ms(stream, [&] { launch_add_perf_f32(stream); });
  auto ms_add_f64 = gpu_ms(stream, [&] { launch_add_perf_f64(stream); });
  auto ms_add_dd  = gpu_ms(stream, [&] { launch_add_perf_dd(stream);  });
  print_perf("add", ms_add_f32, ms_add_f64, ms_add_dd);

  auto ms_mul_f32 = gpu_ms(stream, [&] { launch_mul_perf_f32(stream); });
  auto ms_mul_f64 = gpu_ms(stream, [&] { launch_mul_perf_f64(stream); });
  auto ms_mul_dd  = gpu_ms(stream, [&] { launch_mul_perf_dd(stream);  });
  print_perf("mul", ms_mul_f32, ms_mul_f64, ms_mul_dd);

  auto ms_div_f32 = gpu_ms(stream, [&] { launch_div_perf_f32(stream); });
  auto ms_div_f64 = gpu_ms(stream, [&] { launch_div_perf_f64(stream); });
  auto ms_div_dd  = gpu_ms(stream, [&] { launch_div_perf_dd(stream);  });
  print_perf("div", ms_div_f32, ms_div_f64, ms_div_dd);

  // CPU baseline (fp64) for add/mul/div
  auto cpu_bench = [&](const char* name, auto op) {
    std::vector<double> out(n);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) {
      double x = ha64[i];
      double y = hb64[i];
      for (int it = 0; it < DD_ITERS; ++it) x = op(x, y);
      out[i] = x;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    double ops = (double)n * (double)DD_ITERS;
    double gops = ops / sec / 1e9;
    std::printf("CPU fp64 perf %-4s: %8.3f Gop/s (%.3fs)\n", name, gops, sec);
  };

  cpu_bench("add", [](double x, double y) { return x + y; });
  cpu_bench("mul", [](double x, double y) { return x * y; });
  cpu_bench("div", [](double x, double y) { return x / y; });

  hip_check(hipStreamDestroy(stream), "stream destroy");
  hip_check(hipFree(da_f32), "free");
  hip_check(hipFree(db_f32), "free");
  hip_check(hipFree(do_f32), "free");
  hip_check(hipFree(da_f64), "free");
  hip_check(hipFree(db_f64), "free");
  hip_check(hipFree(do_f64), "free");
  hip_check(hipFree(da_dd), "free");
  hip_check(hipFree(db_dd), "free");
  hip_check(hipFree(do_dd), "free");
  hip_check(hipFree(da_pos_f32), "free");
  hip_check(hipFree(da_pos_f64), "free");
  hip_check(hipFree(da_pos_dd), "free");
  hip_check(hipFree(da_exp_f32), "free");
  hip_check(hipFree(da_exp_f64), "free");
  hip_check(hipFree(da_exp_dd), "free");

  return 0;
}
