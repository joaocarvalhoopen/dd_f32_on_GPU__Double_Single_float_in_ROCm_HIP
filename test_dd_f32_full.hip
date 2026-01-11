#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <random>
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

#ifndef DD_POW_INT_EXP
#define DD_POW_INT_EXP 7
#endif

static void hip_check(hipError_t e, const char* what) {
  if (e != hipSuccess) {
    std::fprintf(stderr, "HIP error (%s): %s\n", what, hipGetErrorString(e));
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

static volatile double g_cpu_sink = 0.0;

// ----------------- Accuracy kernels -----------------------------------------

template <typename T>
__global__ void k_add(const T* a, const T* b, T* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]+b[i]; }

template <typename T>
__global__ void k_sub(const T* a, const T* b, T* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]-b[i]; }

template <typename T>
__global__ void k_mul(const T* a, const T* b, T* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]*b[i]; }

template <typename T>
__global__ void k_div(const T* a, const T* b, T* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]/b[i]; }

__global__ void k_add_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_add(a[i],b[i]); }
__global__ void k_sub_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_sub(a[i],b[i]); }
__global__ void k_mul_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_mul(a[i],b[i]); }
__global__ void k_div_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_div(a[i],b[i]); }

__global__ void k_inv_f32(const f32* a, f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=1.0f/a[i]; }
__global__ void k_inv_f64(const f64* a, f64* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=1.0/a[i]; }
__global__ void k_inv_dd (const dd_f32* a, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_inverse(a[i]); }

__global__ void k_sqrt_f32(const f32* a, f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=sqrtf(a[i]); }
__global__ void k_sqrt_f64(const f64* a, f64* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=sqrt(a[i]); }
__global__ void k_sqrt_dd (const dd_f32* a, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_sqrt(a[i]); }

__global__ void k_invsqrt_f32(const f32* a, f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=rsqrtf(a[i]); }
__global__ void k_invsqrt_f64(const f64* a, f64* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=1.0/sqrt(a[i]); }
__global__ void k_invsqrt_dd (const dd_f32* a, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_inv_sqrt(a[i]); }

__global__ void k_ln_f32(const f32* a, f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=logf(a[i]); }
__global__ void k_ln_f64(const f64* a, f64* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=log(a[i]); }
__global__ void k_ln_dd (const dd_f32* a, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_ln(a[i]); }

__global__ void k_exp_f32(const f32* a, f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=expf(a[i]); }
__global__ void k_exp_f64(const f64* a, f64* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=exp(a[i]); }
__global__ void k_exp_dd (const dd_f32* a, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_exp(a[i]); }

__global__ void k_sin_f32(const f32* a, f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=sinf(a[i]); }
__global__ void k_sin_f64(const f64* a, f64* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=sin(a[i]); }
__global__ void k_sin_dd (const dd_f32* a, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_sin(a[i]); }

__global__ void k_cos_f32(const f32* a, f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=cosf(a[i]); }
__global__ void k_cos_f64(const f64* a, f64* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=cos(a[i]); }
__global__ void k_cos_dd (const dd_f32* a, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_cos(a[i]); }

__global__ void k_tan_f32(const f32* a, f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=tanf(a[i]); }
__global__ void k_tan_f64(const f64* a, f64* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=tan(a[i]); }
__global__ void k_tan_dd (const dd_f32* a, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_tan(a[i]); }

__global__ void k_cotan_f32(const f32* a, f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ f32 t=tanf(a[i]); o[i]=1.0f/t; } }
__global__ void k_cotan_f64(const f64* a, f64* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ f64 t=tan(a[i]); o[i]=1.0/t; } }
__global__ void k_cotan_dd (const dd_f32* a, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_cotan(a[i]); }

__global__ void k_pow_int_dd(const dd_f32* a, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_pow_int(a[i], DD_POW_INT_EXP); }
__global__ void k_pow_int_f32(const f32* a, f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=powf(a[i], (f32)DD_POW_INT_EXP); }
__global__ void k_pow_int_f64(const f64* a, f64* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=pow(a[i], (f64)DD_POW_INT_EXP); }

template <typename T>
__global__ void k_pow(const T* a, const T* b, T* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=pow(a[i], b[i]); }

__global__ void k_pow_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_pow(a[i], b[i]); }

template <typename T>
__global__ void k_log_base(const T* a, const T* b, T* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=log(a[i])/log(b[i]); }

__global__ void k_log_base_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=dd::dd_log(a[i], b[i]); }

// ----------------- Perf kernels (ILP=4) -------------------------------------

#define ILP4_INIT_F32(x0) f32 x1=x0+1.0f, x2=x0+2.0f, x3=x0+3.0f
#define ILP4_INIT_F64(x0) f64 x1=x0+1.0,  x2=x0+2.0,  x3=x0+3.0
#define ILP4_INIT_DD(x0)  dd_f32 x1=dd::dd_add(x0,(f32)1), x2=dd::dd_add(x0,(f32)2), x3=dd::dd_add(x0,(f32)3)

template <typename T>
__global__ void p_add(const T* a, const T* b, T* o, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  T x0=a[i]; T y=b[i]; T x1=x0+(T)1, x2=x0+(T)2, x3=x0+(T)3;
  #pragma unroll 1
  for(int it=0; it<DD_ITERS; ++it){ x0=x0+y; x1=x1+y; x2=x2+y; x3=x3+y; }
  o[i]=x0+x1+x2+x3;
}

__global__ void p_add_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  dd_f32 x0=a[i], y=b[i]; ILP4_INIT_DD(x0);
  #pragma unroll 1
  for(int it=0; it<DD_ITERS; ++it){ x0=dd::dd_add(x0,y); x1=dd::dd_add(x1,y); x2=dd::dd_add(x2,y); x3=dd::dd_add(x3,y); }
  o[i]=dd::dd_add(dd::dd_add(x0,x1), dd::dd_add(x2,x3));
}

template <typename T>
__global__ void p_sub(const T* a, const T* b, T* o, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  T x0=a[i]; T y=b[i]; T x1=x0+(T)1, x2=x0+(T)2, x3=x0+(T)3;
  #pragma unroll 1
  for(int it=0; it<DD_ITERS; ++it){ x0=x0-y; x1=x1-y; x2=x2-y; x3=x3-y; }
  o[i]=x0+x1+x2+x3;
}

__global__ void p_sub_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  dd_f32 x0=a[i], y=b[i]; ILP4_INIT_DD(x0);
  #pragma unroll 1
  for(int it=0; it<DD_ITERS; ++it){ x0=dd::dd_sub(x0,y); x1=dd::dd_sub(x1,y); x2=dd::dd_sub(x2,y); x3=dd::dd_sub(x3,y); }
  o[i]=dd::dd_add(dd::dd_add(x0,x1), dd::dd_add(x2,x3));
}

template <typename T>
__global__ void p_mul(const T* a, const T* b, T* o, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  T x0=a[i]; T y=b[i]; T x1=x0+(T)1, x2=x0+(T)2, x3=x0+(T)3;
  #pragma unroll 1
  for(int it=0; it<DD_ITERS; ++it){ x0=x0*y; x1=x1*y; x2=x2*y; x3=x3*y; }
  o[i]=x0+x1+x2+x3;
}

__global__ void p_mul_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  dd_f32 x0=a[i], y=b[i]; ILP4_INIT_DD(x0);
  #pragma unroll 1
  for(int it=0; it<DD_ITERS; ++it){ x0=dd::dd_mul(x0,y); x1=dd::dd_mul(x1,y); x2=dd::dd_mul(x2,y); x3=dd::dd_mul(x3,y); }
  o[i]=dd::dd_add(dd::dd_add(x0,x1), dd::dd_add(x2,x3));
}

template <typename T>
__global__ void p_div(const T* a, const T* b, T* o, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  T x0=a[i]; T y=b[i]; T x1=x0+(T)1, x2=x0+(T)2, x3=x0+(T)3;
  #pragma unroll 1
  for(int it=0; it<DD_ITERS; ++it){ x0=x0/y; x1=x1/y; x2=x2/y; x3=x3/y; }
  o[i]=x0+x1+x2+x3;
}

__global__ void p_div_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
  dd_f32 x0=a[i], y=b[i]; ILP4_INIT_DD(x0);
  #pragma unroll 1
  for(int it=0; it<DD_ITERS; ++it){ x0=dd::dd_div(x0,y); x1=dd::dd_div(x1,y); x2=dd::dd_div(x2,y); x3=dd::dd_div(x3,y); }
  o[i]=dd::dd_add(dd::dd_add(x0,x1), dd::dd_add(x2,x3));
}

// Unary perf helpers
__global__ void p_inv_f32(const f32* a, f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f32 x0 = a[i];
  ILP4_INIT_F32(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = 1.0f / x0; x1 = 1.0f / x1; x2 = 1.0f / x2; x3 = 1.0f / x3;
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_inv_f64(const f64* a, f64* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f64 x0 = a[i];
  ILP4_INIT_F64(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = 1.0 / x0; x1 = 1.0 / x1; x2 = 1.0 / x2; x3 = 1.0 / x3;
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_inv_dd(const dd_f32* a, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_inverse(x0);
    x1 = dd::dd_inverse(x1);
    x2 = dd::dd_inverse(x2);
    x3 = dd::dd_inverse(x3);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

__global__ void p_sqrt_f32(const f32* a, f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f32 x0 = a[i];
  ILP4_INIT_F32(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = sqrtf(x0); x1 = sqrtf(x1); x2 = sqrtf(x2); x3 = sqrtf(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_sqrt_f64(const f64* a, f64* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f64 x0 = a[i];
  ILP4_INIT_F64(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = sqrt(x0); x1 = sqrt(x1); x2 = sqrt(x2); x3 = sqrt(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_sqrt_dd(const dd_f32* a, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_sqrt(x0);
    x1 = dd::dd_sqrt(x1);
    x2 = dd::dd_sqrt(x2);
    x3 = dd::dd_sqrt(x3);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

__global__ void p_invsqrt_f32(const f32* a, f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f32 x0 = a[i];
  ILP4_INIT_F32(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = rsqrtf(x0); x1 = rsqrtf(x1); x2 = rsqrtf(x2); x3 = rsqrtf(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_invsqrt_f64(const f64* a, f64* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f64 x0 = a[i];
  ILP4_INIT_F64(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = 1.0 / sqrt(x0); x1 = 1.0 / sqrt(x1); x2 = 1.0 / sqrt(x2); x3 = 1.0 / sqrt(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_invsqrt_dd(const dd_f32* a, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_inv_sqrt(x0);
    x1 = dd::dd_inv_sqrt(x1);
    x2 = dd::dd_inv_sqrt(x2);
    x3 = dd::dd_inv_sqrt(x3);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

// Transcendentals perf: count as "1 op" even though SFU-heavy.
__global__ void p_ln_f32(const f32* a, f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f32 x0 = a[i];
  ILP4_INIT_F32(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = logf(x0); x1 = logf(x1); x2 = logf(x2); x3 = logf(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_ln_f64(const f64* a, f64* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f64 x0 = a[i];
  ILP4_INIT_F64(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = log(x0); x1 = log(x1); x2 = log(x2); x3 = log(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_ln_dd(const dd_f32* a, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_ln(x0);
    x1 = dd::dd_ln(x1);
    x2 = dd::dd_ln(x2);
    x3 = dd::dd_ln(x3);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

__global__ void p_exp_f32(const f32* a, f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f32 x0 = a[i];
  ILP4_INIT_F32(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = expf(x0); x1 = expf(x1); x2 = expf(x2); x3 = expf(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_exp_f64(const f64* a, f64* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f64 x0 = a[i];
  ILP4_INIT_F64(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = exp(x0); x1 = exp(x1); x2 = exp(x2); x3 = exp(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_exp_dd(const dd_f32* a, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_exp(x0);
    x1 = dd::dd_exp(x1);
    x2 = dd::dd_exp(x2);
    x3 = dd::dd_exp(x3);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

__global__ void p_sin_f32(const f32* a, f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f32 x0 = a[i];
  ILP4_INIT_F32(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = sinf(x0); x1 = sinf(x1); x2 = sinf(x2); x3 = sinf(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_sin_f64(const f64* a, f64* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f64 x0 = a[i];
  ILP4_INIT_F64(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = sin(x0); x1 = sin(x1); x2 = sin(x2); x3 = sin(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_sin_dd(const dd_f32* a, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_sin(x0);
    x1 = dd::dd_sin(x1);
    x2 = dd::dd_sin(x2);
    x3 = dd::dd_sin(x3);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

__global__ void p_cos_f32(const f32* a, f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f32 x0 = a[i];
  ILP4_INIT_F32(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = cosf(x0); x1 = cosf(x1); x2 = cosf(x2); x3 = cosf(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_cos_f64(const f64* a, f64* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f64 x0 = a[i];
  ILP4_INIT_F64(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = cos(x0); x1 = cos(x1); x2 = cos(x2); x3 = cos(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_cos_dd(const dd_f32* a, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_cos(x0);
    x1 = dd::dd_cos(x1);
    x2 = dd::dd_cos(x2);
    x3 = dd::dd_cos(x3);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

__global__ void p_tan_f32(const f32* a, f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f32 x0 = a[i];
  ILP4_INIT_F32(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = tanf(x0); x1 = tanf(x1); x2 = tanf(x2); x3 = tanf(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_tan_f64(const f64* a, f64* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f64 x0 = a[i];
  ILP4_INIT_F64(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = tan(x0); x1 = tan(x1); x2 = tan(x2); x3 = tan(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_tan_dd(const dd_f32* a, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_tan(x0);
    x1 = dd::dd_tan(x1);
    x2 = dd::dd_tan(x2);
    x3 = dd::dd_tan(x3);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

__global__ void p_cotan_f32(const f32* a, f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f32 x0 = a[i];
  ILP4_INIT_F32(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = 1.0f / tanf(x0); x1 = 1.0f / tanf(x1); x2 = 1.0f / tanf(x2); x3 = 1.0f / tanf(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_cotan_f64(const f64* a, f64* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f64 x0 = a[i];
  ILP4_INIT_F64(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = 1.0 / tan(x0); x1 = 1.0 / tan(x1); x2 = 1.0 / tan(x2); x3 = 1.0 / tan(x3);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_cotan_dd(const dd_f32* a, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_cotan(x0);
    x1 = dd::dd_cotan(x1);
    x2 = dd::dd_cotan(x2);
    x3 = dd::dd_cotan(x3);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

__global__ void p_pow_int_f32(const f32* a, f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f32 x0 = a[i];
  ILP4_INIT_F32(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = powf(x0, (f32)DD_POW_INT_EXP);
    x1 = powf(x1, (f32)DD_POW_INT_EXP);
    x2 = powf(x2, (f32)DD_POW_INT_EXP);
    x3 = powf(x3, (f32)DD_POW_INT_EXP);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_pow_int_f64(const f64* a, f64* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  f64 x0 = a[i];
  ILP4_INIT_F64(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = pow(x0, (f64)DD_POW_INT_EXP);
    x1 = pow(x1, (f64)DD_POW_INT_EXP);
    x2 = pow(x2, (f64)DD_POW_INT_EXP);
    x3 = pow(x3, (f64)DD_POW_INT_EXP);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_pow_int_dd(const dd_f32* a, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_pow_int(x0, DD_POW_INT_EXP);
    x1 = dd::dd_pow_int(x1, DD_POW_INT_EXP);
    x2 = dd::dd_pow_int(x2, DD_POW_INT_EXP);
    x3 = dd::dd_pow_int(x3, DD_POW_INT_EXP);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

template <typename T>
__global__ void p_pow(const T* a, const T* b, T* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  T x0 = a[i];
  T y = b[i];
  T x1 = x0 + (T)1, x2 = x0 + (T)2, x3 = x0 + (T)3;
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = pow(x0, y);
    x1 = pow(x1, y);
    x2 = pow(x2, y);
    x3 = pow(x3, y);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_pow_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  dd_f32 y = b[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_pow(x0, y);
    x1 = dd::dd_pow(x1, y);
    x2 = dd::dd_pow(x2, y);
    x3 = dd::dd_pow(x3, y);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

template <typename T>
__global__ void p_log_base(const T* a, const T* b, T* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  T x0 = a[i];
  T y = b[i];
  T x1 = x0 + (T)1, x2 = x0 + (T)2, x3 = x0 + (T)3;
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = log(x0) / log(y);
    x1 = log(x1) / log(y);
    x2 = log(x2) / log(y);
    x3 = log(x3) / log(y);
  }
  o[i] = x0 + x1 + x2 + x3;
}

__global__ void p_log_base_dd(const dd_f32* a, const dd_f32* b, dd_f32* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dd_f32 x0 = a[i];
  dd_f32 y = b[i];
  ILP4_INIT_DD(x0);
  #pragma unroll 1
  for (int it = 0; it < DD_ITERS; ++it) {
    x0 = dd::dd_log(x0, y);
    x1 = dd::dd_log(x1, y);
    x2 = dd::dd_log(x2, y);
    x3 = dd::dd_log(x3, y);
  }
  o[i] = dd::dd_add(dd::dd_add(x0, x1), dd::dd_add(x2, x3));
}

// ----------------- CPU benches ----------------------------------------------

template <class T, class Op>
static double cpu_ops_per_s_unary(const std::vector<T>& a, Op op) {
  std::vector<T> out(a.size());
  auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < a.size(); ++i) {
    T x0 = a[i];
    T x1 = x0 + (T)1, x2 = x0 + (T)2, x3 = x0 + (T)3;
    for (int it = 0; it < DD_ITERS; ++it) {
      x0 = op(x0);
      x1 = op(x1);
      x2 = op(x2);
      x3 = op(x3);
    }
    out[i] = x0 + x1 + x2 + x3;
    g_cpu_sink += (double)out[i];
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  double ops = (double)a.size() * (double)DD_ITERS * (double)DD_ILP;
  return ops / sec;
}

template <class T, class Op>
static double cpu_ops_per_s_binary(const std::vector<T>& a, const std::vector<T>& b, Op op) {
  std::vector<T> out(a.size());
  auto t0 = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < a.size(); ++i) {
    T x0 = a[i];
    T x1 = x0 + (T)1, x2 = x0 + (T)2, x3 = x0 + (T)3;
    T y = b[i];
    for (int it = 0; it < DD_ITERS; ++it) {
      x0 = op(x0, y);
      x1 = op(x1, y);
      x2 = op(x2, y);
      x3 = op(x3, y);
    }
    out[i] = x0 + x1 + x2 + x3;
    g_cpu_sink += (double)out[i];
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  double ops = (double)a.size() * (double)DD_ITERS * (double)DD_ILP;
  return ops / sec;
}

int main(int argc, char** argv) {
  int n = DD_N;
  if (argc >= 2) n = std::max(1, std::atoi(argv[1]));

  hipDeviceProp_t prop{};
  hip_check(hipGetDeviceProperties(&prop, 0), "get device props");
  std::printf("GPU: %s\n", prop.name);
  std::printf("N=%d, ITERS=%d, ILP=%d\n", n, (int)DD_ITERS, (int)DD_ILP);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);
  std::uniform_real_distribution<float> dist_pos(1e-3f, 1000.0f);
  std::uniform_real_distribution<float> dist_exp(-10.0f, 10.0f);
  std::uniform_real_distribution<float> dist_angle(-3.0f, 3.0f);
  std::uniform_real_distribution<float> dist_powexp(-3.0f, 3.0f);

  std::vector<f32> a(n), b(n), a_pos(n), b_pos(n), a_exp(n), ang(n), powe(n);
  for (int i=0;i<n;++i) {
    a[i]=dist(rng);
    b[i]=dist(rng);
    if (fabsf(b[i]) < 1e-3f) b[i]=1.0f;
    a_pos[i]=dist_pos(rng);
    b_pos[i]=dist_pos(rng) + 1.0f; // base > 1
    a_exp[i]=dist_exp(rng);
    ang[i]=dist_angle(rng);
    powe[i]=dist_powexp(rng);
  }

  std::vector<f64> a64(n), b64(n), a_pos64(n), b_pos64(n), a_exp64(n), ang64(n), powe64(n);
  for (int i=0;i<n;++i){
    a64[i]=a[i]; b64[i]=b[i]; a_pos64[i]=a_pos[i]; b_pos64[i]=b_pos[i]; a_exp64[i]=a_exp[i]; ang64[i]=ang[i]; powe64[i]=powe[i];
  }

  std::vector<dd_f32> a_dd(n), b_dd(n), a_pos_dd(n), b_pos_dd(n), a_exp_dd(n), ang_dd(n), powe_dd(n);
  for (int i=0;i<n;++i){
    a_dd[i]=dd::dd_f32_from_f32(a[i]);
    b_dd[i]=dd::dd_f32_from_f32(b[i]);
    a_pos_dd[i]=dd::dd_f32_from_f32(a_pos[i]);
    b_pos_dd[i]=dd::dd_f32_from_f32(b_pos[i]);
    a_exp_dd[i]=dd::dd_f32_from_f32(a_exp[i]);
    ang_dd[i]=dd::dd_f32_from_f32(ang[i]);
    powe_dd[i]=dd::dd_f32_from_f32(powe[i]);
  }

  // Device buffers
  f32 *da=nullptr,*db=nullptr,*do32=nullptr;
  f64 *da64d=nullptr,*db64d=nullptr,*do64=nullptr;
  dd_f32 *dadd=nullptr,*dbdd=nullptr,*dodd=nullptr;

  f32 *da_posd=nullptr,*db_posd=nullptr,*da_expd=nullptr,*dangd=nullptr;
  f64 *da_pos64d=nullptr,*db_pos64d=nullptr,*da_exp64d=nullptr,*dang64d=nullptr;
  dd_f32 *da_posddd=nullptr,*db_posddd=nullptr,*da_expddd=nullptr,*dangddd=nullptr;

  f32 *dpowed=nullptr;
  f64 *dpowe64d=nullptr;
  dd_f32 *dpowedd=nullptr;

  hip_check(hipMalloc(&da, n*sizeof(f32)), "malloc");
  hip_check(hipMalloc(&db, n*sizeof(f32)), "malloc");
  hip_check(hipMalloc(&do32, n*sizeof(f32)), "malloc");
  hip_check(hipMalloc(&da64d, n*sizeof(f64)), "malloc");
  hip_check(hipMalloc(&db64d, n*sizeof(f64)), "malloc");
  hip_check(hipMalloc(&do64, n*sizeof(f64)), "malloc");
  hip_check(hipMalloc(&dadd, n*sizeof(dd_f32)), "malloc");
  hip_check(hipMalloc(&dbdd, n*sizeof(dd_f32)), "malloc");
  hip_check(hipMalloc(&dodd, n*sizeof(dd_f32)), "malloc");

  hip_check(hipMalloc(&da_posd, n*sizeof(f32)), "malloc");
  hip_check(hipMalloc(&db_posd, n*sizeof(f32)), "malloc");
  hip_check(hipMalloc(&da_expd, n*sizeof(f32)), "malloc");
  hip_check(hipMalloc(&dangd, n*sizeof(f32)), "malloc");
  hip_check(hipMalloc(&dpowed, n*sizeof(f32)), "malloc");

  hip_check(hipMalloc(&da_pos64d, n*sizeof(f64)), "malloc");
  hip_check(hipMalloc(&db_pos64d, n*sizeof(f64)), "malloc");
  hip_check(hipMalloc(&da_exp64d, n*sizeof(f64)), "malloc");
  hip_check(hipMalloc(&dang64d, n*sizeof(f64)), "malloc");
  hip_check(hipMalloc(&dpowe64d, n*sizeof(f64)), "malloc");

  hip_check(hipMalloc(&da_posddd, n*sizeof(dd_f32)), "malloc");
  hip_check(hipMalloc(&db_posddd, n*sizeof(dd_f32)), "malloc");
  hip_check(hipMalloc(&da_expddd, n*sizeof(dd_f32)), "malloc");
  hip_check(hipMalloc(&dangddd, n*sizeof(dd_f32)), "malloc");
  hip_check(hipMalloc(&dpowedd, n*sizeof(dd_f32)), "malloc");

  hip_check(hipMemcpy(da, a.data(), n*sizeof(f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(db, b.data(), n*sizeof(f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(da64d, a64.data(), n*sizeof(f64), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(db64d, b64.data(), n*sizeof(f64), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(dadd, a_dd.data(), n*sizeof(dd_f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(dbdd, b_dd.data(), n*sizeof(dd_f32), hipMemcpyHostToDevice), "cpy");

  hip_check(hipMemcpy(da_posd, a_pos.data(), n*sizeof(f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(db_posd, b_pos.data(), n*sizeof(f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(da_expd, a_exp.data(), n*sizeof(f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(dangd, ang.data(), n*sizeof(f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(dpowed, powe.data(), n*sizeof(f32), hipMemcpyHostToDevice), "cpy");

  hip_check(hipMemcpy(da_pos64d, a_pos64.data(), n*sizeof(f64), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(db_pos64d, b_pos64.data(), n*sizeof(f64), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(da_exp64d, a_exp64.data(), n*sizeof(f64), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(dang64d, ang64.data(), n*sizeof(f64), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(dpowe64d, powe64.data(), n*sizeof(f64), hipMemcpyHostToDevice), "cpy");

  hip_check(hipMemcpy(da_posddd, a_pos_dd.data(), n*sizeof(dd_f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(db_posddd, b_pos_dd.data(), n*sizeof(dd_f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(da_expddd, a_exp_dd.data(), n*sizeof(dd_f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(dangddd, ang_dd.data(), n*sizeof(dd_f32), hipMemcpyHostToDevice), "cpy");
  hip_check(hipMemcpy(dpowedd, powe_dd.data(), n*sizeof(dd_f32), hipMemcpyHostToDevice), "cpy");

  hipStream_t stream;
  hip_check(hipStreamCreate(&stream), "stream");
  int block=256;
  int grid=(n+block-1)/block;

  auto print_acc_bin = [&](const char* name, auto launch32, auto launch64, auto launchdd, auto ref) {
    std::vector<f32> o32(n); std::vector<f64> o64(n); std::vector<dd_f32> odd(n);
    launch32(); hip_check(hipMemcpy(o32.data(), do32, n*sizeof(f32), hipMemcpyDeviceToHost), "cpy");
    launch64(); hip_check(hipMemcpy(o64.data(), do64, n*sizeof(f64), hipMemcpyDeviceToHost), "cpy");
    launchdd(); hip_check(hipMemcpy(odd.data(), dodd, n*sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");
    double min32=1e9,min64=1e9,mindd=1e9;
    for(int i=0;i<n;++i){
      long double r=ref((long double)a[i], (long double)b[i]);
      min32=std::min(min32, rel_bits(r, (long double)o32[i]));
      min64=std::min(min64, rel_bits(r, (long double)o64[i]));
      mindd=std::min(mindd, rel_bits(r, (long double)dd::dd_f32_as_f64(odd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Prec(bits,min rel) %-7s | CPU ref vs GPU f32:%6s f64:%6s dd:%6s\n",
                name, fmt_bits(min32, b32), fmt_bits(min64, b64), fmt_bits(mindd, bdd));
  };

  auto print_acc_un = [&](const char* name, const std::vector<f32>& ah, const std::vector<f64>& ah64,
                          auto launch32, auto launch64, auto launchdd, auto ref) {
    std::vector<f32> o32(n); std::vector<f64> o64(n); std::vector<dd_f32> odd(n);
    launch32(); hip_check(hipMemcpy(o32.data(), do32, n*sizeof(f32), hipMemcpyDeviceToHost), "cpy");
    launch64(); hip_check(hipMemcpy(o64.data(), do64, n*sizeof(f64), hipMemcpyDeviceToHost), "cpy");
    launchdd(); hip_check(hipMemcpy(odd.data(), dodd, n*sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");
    double min32=1e9,min64=1e9,mindd=1e9;
    for(int i=0;i<n;++i){
      long double r=ref((long double)ah[i]);
      min32=std::min(min32, rel_bits(r, (long double)o32[i]));
      min64=std::min(min64, rel_bits(r, (long double)o64[i]));
      mindd=std::min(mindd, rel_bits(r, (long double)dd::dd_f32_as_f64(odd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Prec(bits,min rel) %-7s | CPU ref vs GPU f32:%6s f64:%6s dd:%6s\n",
                name, fmt_bits(min32, b32), fmt_bits(min64, b64), fmt_bits(mindd, bdd));
  };

  // Accuracy for basic ops (these are exact for float inputs, so you may see 200 bits)
  print_acc_bin("add", [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_add<f32>), dim3(grid), dim3(block), 0, stream, da, db, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_add<f64>), dim3(grid), dim3(block), 0, stream, da64d, db64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [&]{ hipLaunchKernelGGL(k_add_dd, dim3(grid), dim3(block), 0, stream, dadd, dbdd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [](long double x, long double y){ return x+y; });

  print_acc_bin("sub", [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_sub<f32>), dim3(grid), dim3(block), 0, stream, da, db, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_sub<f64>), dim3(grid), dim3(block), 0, stream, da64d, db64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [&]{ hipLaunchKernelGGL(k_sub_dd, dim3(grid), dim3(block), 0, stream, dadd, dbdd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [](long double x, long double y){ return x-y; });

  print_acc_bin("mul", [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_mul<f32>), dim3(grid), dim3(block), 0, stream, da, db, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_mul<f64>), dim3(grid), dim3(block), 0, stream, da64d, db64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [&]{ hipLaunchKernelGGL(k_mul_dd, dim3(grid), dim3(block), 0, stream, dadd, dbdd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [](long double x, long double y){ return x*y; });

  print_acc_bin("div", [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_div<f32>), dim3(grid), dim3(block), 0, stream, da, db, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_div<f64>), dim3(grid), dim3(block), 0, stream, da64d, db64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [&]{ hipLaunchKernelGGL(k_div_dd, dim3(grid), dim3(block), 0, stream, dadd, dbdd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
                      [](long double x, long double y){ return x/y; });

  // Unary accuracy
  print_acc_un("inverse", a, a64,
               [&]{ hipLaunchKernelGGL(k_inv_f32, dim3(grid), dim3(block), 0, stream, da, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_inv_f64, dim3(grid), dim3(block), 0, stream, da64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_inv_dd,  dim3(grid), dim3(block), 0, stream, dadd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [](long double x){ return 1.0L/x; });

  print_acc_un("sqrt", a_pos, a_pos64,
               [&]{ hipLaunchKernelGGL(k_sqrt_f32, dim3(grid), dim3(block), 0, stream, da_posd, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_sqrt_f64, dim3(grid), dim3(block), 0, stream, da_pos64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_sqrt_dd,  dim3(grid), dim3(block), 0, stream, da_posddd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [](long double x){ return sqrtl(x); });

  print_acc_un("inv_sqrt", a_pos, a_pos64,
               [&]{ hipLaunchKernelGGL(k_invsqrt_f32, dim3(grid), dim3(block), 0, stream, da_posd, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_invsqrt_f64, dim3(grid), dim3(block), 0, stream, da_pos64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_invsqrt_dd,  dim3(grid), dim3(block), 0, stream, da_posddd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [](long double x){ return 1.0L/sqrtl(x); });

  print_acc_un("ln", a_pos, a_pos64,
               [&]{ hipLaunchKernelGGL(k_ln_f32, dim3(grid), dim3(block), 0, stream, da_posd, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_ln_f64, dim3(grid), dim3(block), 0, stream, da_pos64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_ln_dd,  dim3(grid), dim3(block), 0, stream, da_posddd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [](long double x){ return logl(x); });

  print_acc_un("exp", a_exp, a_exp64,
               [&]{ hipLaunchKernelGGL(k_exp_f32, dim3(grid), dim3(block), 0, stream, da_expd, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_exp_f64, dim3(grid), dim3(block), 0, stream, da_exp64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_exp_dd,  dim3(grid), dim3(block), 0, stream, da_expddd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [](long double x){ return expl(x); });

  print_acc_un("sin", ang, ang64,
               [&]{ hipLaunchKernelGGL(k_sin_f32, dim3(grid), dim3(block), 0, stream, dangd, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_sin_f64, dim3(grid), dim3(block), 0, stream, dang64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_sin_dd,  dim3(grid), dim3(block), 0, stream, dangddd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [](long double x){ return sinl(x); });

  print_acc_un("cos", ang, ang64,
               [&]{ hipLaunchKernelGGL(k_cos_f32, dim3(grid), dim3(block), 0, stream, dangd, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_cos_f64, dim3(grid), dim3(block), 0, stream, dang64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_cos_dd,  dim3(grid), dim3(block), 0, stream, dangddd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [](long double x){ return cosl(x); });

  print_acc_un("tan", ang, ang64,
               [&]{ hipLaunchKernelGGL(k_tan_f32, dim3(grid), dim3(block), 0, stream, dangd, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_tan_f64, dim3(grid), dim3(block), 0, stream, dang64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_tan_dd,  dim3(grid), dim3(block), 0, stream, dangddd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [](long double x){ return tanl(x); });

  print_acc_un("cotan", ang, ang64,
               [&]{ hipLaunchKernelGGL(k_cotan_f32, dim3(grid), dim3(block), 0, stream, dangd, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_cotan_f64, dim3(grid), dim3(block), 0, stream, dang64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_cotan_dd,  dim3(grid), dim3(block), 0, stream, dangddd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [](long double x){ return 1.0L/tanl(x); });

  print_acc_un("pow_int", a_pos, a_pos64,
               [&]{ hipLaunchKernelGGL(k_pow_int_f32, dim3(grid), dim3(block), 0, stream, da_posd, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_pow_int_f64, dim3(grid), dim3(block), 0, stream, da_pos64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [&]{ hipLaunchKernelGGL(k_pow_int_dd,  dim3(grid), dim3(block), 0, stream, da_posddd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); },
               [](long double x){ return powl(x, (long double)DD_POW_INT_EXP); });

  // pow(a_pos, powe)
  {
    auto launch32 = [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_pow<f32>), dim3(grid), dim3(block), 0, stream, da_posd, dpowed, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); };
    auto launch64 = [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_pow<f64>), dim3(grid), dim3(block), 0, stream, da_pos64d, dpowe64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); };
    auto launchdd = [&]{ hipLaunchKernelGGL(k_pow_dd, dim3(grid), dim3(block), 0, stream, da_posddd, dpowedd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); };
    std::vector<f32> o32(n); std::vector<f64> o64(n); std::vector<dd_f32> odd(n);
    launch32(); hip_check(hipMemcpy(o32.data(), do32, n*sizeof(f32), hipMemcpyDeviceToHost), "cpy");
    launch64(); hip_check(hipMemcpy(o64.data(), do64, n*sizeof(f64), hipMemcpyDeviceToHost), "cpy");
    launchdd(); hip_check(hipMemcpy(odd.data(), dodd, n*sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");
    double min32=1e9,min64=1e9,mindd=1e9;
    for(int i=0;i<n;++i){
      long double r=powl((long double)a_pos[i], (long double)powe[i]);
      min32=std::min(min32, rel_bits(r, (long double)o32[i]));
      min64=std::min(min64, rel_bits(r, (long double)o64[i]));
      mindd=std::min(mindd, rel_bits(r, (long double)dd::dd_f32_as_f64(odd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Prec(bits,min rel) %-7s | CPU ref vs GPU f32:%6s f64:%6s dd:%6s\n",
                "pow", fmt_bits(min32, b32), fmt_bits(min64, b64), fmt_bits(mindd, bdd));
  }

  // log_base(a_pos, b_pos)
  {
    auto launch32 = [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_log_base<f32>), dim3(grid), dim3(block), 0, stream, da_posd, db_posd, do32, n); hip_check(hipStreamSynchronize(stream),"sync"); };
    auto launch64 = [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(k_log_base<f64>), dim3(grid), dim3(block), 0, stream, da_pos64d, db_pos64d, do64, n); hip_check(hipStreamSynchronize(stream),"sync"); };
    auto launchdd = [&]{ hipLaunchKernelGGL(k_log_base_dd, dim3(grid), dim3(block), 0, stream, da_posddd, db_posddd, dodd, n); hip_check(hipStreamSynchronize(stream),"sync"); };
    std::vector<f32> o32(n); std::vector<f64> o64(n); std::vector<dd_f32> odd(n);
    launch32(); hip_check(hipMemcpy(o32.data(), do32, n*sizeof(f32), hipMemcpyDeviceToHost), "cpy");
    launch64(); hip_check(hipMemcpy(o64.data(), do64, n*sizeof(f64), hipMemcpyDeviceToHost), "cpy");
    launchdd(); hip_check(hipMemcpy(odd.data(), dodd, n*sizeof(dd_f32), hipMemcpyDeviceToHost), "cpy");
    double min32=1e9,min64=1e9,mindd=1e9;
    for(int i=0;i<n;++i){
      long double r=logl((long double)a_pos[i]) / logl((long double)b_pos[i]);
      min32=std::min(min32, rel_bits(r, (long double)o32[i]));
      min64=std::min(min64, rel_bits(r, (long double)o64[i]));
      mindd=std::min(mindd, rel_bits(r, (long double)dd::dd_f32_as_f64(odd[i])));
    }
    char b32[16], b64[16], bdd[16];
    std::printf("Prec(bits,min rel) %-7s | CPU ref vs GPU f32:%6s f64:%6s dd:%6s\n",
                "log", fmt_bits(min32, b32), fmt_bits(min64, b64), fmt_bits(mindd, bdd));
  }

  // ---------------- Performance ---------------------------------------------
  auto print_perf = [&](const char* name, float ms32, float ms64, float msdd) {
    double ops = (double)n * (double)DD_ITERS * (double)DD_ILP;
    double t32=ms32*1e-3, t64=ms64*1e-3, tdd=msdd*1e-3;
    double tops32 = ops/t32/1e12;
    double tops64 = ops/t64/1e12;
    double topsdd = ops/tdd/1e12;
    std::printf("Perf(TOp/s) %-7s | GPU f32:%7.3f  f64:%7.3f  dd:%7.3f\n", name, tops32, tops64, topsdd);
  };

  auto bench3 = [&](const char* name, auto l32, auto l64, auto ldd) {
    float ms32=gpu_ms(stream, l32);
    float ms64=gpu_ms(stream, l64);
    float msdd=gpu_ms(stream, ldd);
    print_perf(name, ms32, ms64, msdd);
  };

  bench3("add", [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_add<f32>), dim3(grid), dim3(block), 0, stream, da, db, do32, n); },
               [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_add<f64>), dim3(grid), dim3(block), 0, stream, da64d, db64d, do64, n); },
               [&]{ hipLaunchKernelGGL(p_add_dd, dim3(grid), dim3(block), 0, stream, dadd, dbdd, dodd, n); });

  bench3("sub", [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_sub<f32>), dim3(grid), dim3(block), 0, stream, da, db, do32, n); },
               [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_sub<f64>), dim3(grid), dim3(block), 0, stream, da64d, db64d, do64, n); },
               [&]{ hipLaunchKernelGGL(p_sub_dd, dim3(grid), dim3(block), 0, stream, dadd, dbdd, dodd, n); });

  bench3("mul", [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_mul<f32>), dim3(grid), dim3(block), 0, stream, da, db, do32, n); },
               [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_mul<f64>), dim3(grid), dim3(block), 0, stream, da64d, db64d, do64, n); },
               [&]{ hipLaunchKernelGGL(p_mul_dd, dim3(grid), dim3(block), 0, stream, dadd, dbdd, dodd, n); });

  bench3("div", [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_div<f32>), dim3(grid), dim3(block), 0, stream, da, db, do32, n); },
               [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_div<f64>), dim3(grid), dim3(block), 0, stream, da64d, db64d, do64, n); },
               [&]{ hipLaunchKernelGGL(p_div_dd, dim3(grid), dim3(block), 0, stream, dadd, dbdd, dodd, n); });

  bench3("inverse", [&]{ hipLaunchKernelGGL(p_inv_f32, dim3(grid), dim3(block), 0, stream, da, do32, n); },
                   [&]{ hipLaunchKernelGGL(p_inv_f64, dim3(grid), dim3(block), 0, stream, da64d, do64, n); },
                   [&]{ hipLaunchKernelGGL(p_inv_dd,  dim3(grid), dim3(block), 0, stream, dadd, dodd, n); });

  bench3("sqrt", [&]{ hipLaunchKernelGGL(p_sqrt_f32, dim3(grid), dim3(block), 0, stream, da_posd, do32, n); },
                [&]{ hipLaunchKernelGGL(p_sqrt_f64, dim3(grid), dim3(block), 0, stream, da_pos64d, do64, n); },
                [&]{ hipLaunchKernelGGL(p_sqrt_dd,  dim3(grid), dim3(block), 0, stream, da_posddd, dodd, n); });

  bench3("inv_sqrt", [&]{ hipLaunchKernelGGL(p_invsqrt_f32, dim3(grid), dim3(block), 0, stream, da_posd, do32, n); },
                    [&]{ hipLaunchKernelGGL(p_invsqrt_f64, dim3(grid), dim3(block), 0, stream, da_pos64d, do64, n); },
                    [&]{ hipLaunchKernelGGL(p_invsqrt_dd,  dim3(grid), dim3(block), 0, stream, da_posddd, dodd, n); });

  bench3("ln", [&]{ hipLaunchKernelGGL(p_ln_f32, dim3(grid), dim3(block), 0, stream, da_posd, do32, n); },
              [&]{ hipLaunchKernelGGL(p_ln_f64, dim3(grid), dim3(block), 0, stream, da_pos64d, do64, n); },
              [&]{ hipLaunchKernelGGL(p_ln_dd,  dim3(grid), dim3(block), 0, stream, da_posddd, dodd, n); });

  bench3("exp", [&]{ hipLaunchKernelGGL(p_exp_f32, dim3(grid), dim3(block), 0, stream, da_expd, do32, n); },
               [&]{ hipLaunchKernelGGL(p_exp_f64, dim3(grid), dim3(block), 0, stream, da_exp64d, do64, n); },
               [&]{ hipLaunchKernelGGL(p_exp_dd,  dim3(grid), dim3(block), 0, stream, da_expddd, dodd, n); });

  bench3("sin", [&]{ hipLaunchKernelGGL(p_sin_f32, dim3(grid), dim3(block), 0, stream, dangd, do32, n); },
               [&]{ hipLaunchKernelGGL(p_sin_f64, dim3(grid), dim3(block), 0, stream, dang64d, do64, n); },
               [&]{ hipLaunchKernelGGL(p_sin_dd,  dim3(grid), dim3(block), 0, stream, dangddd, dodd, n); });

  bench3("cos", [&]{ hipLaunchKernelGGL(p_cos_f32, dim3(grid), dim3(block), 0, stream, dangd, do32, n); },
               [&]{ hipLaunchKernelGGL(p_cos_f64, dim3(grid), dim3(block), 0, stream, dang64d, do64, n); },
               [&]{ hipLaunchKernelGGL(p_cos_dd,  dim3(grid), dim3(block), 0, stream, dangddd, dodd, n); });

  bench3("tan", [&]{ hipLaunchKernelGGL(p_tan_f32, dim3(grid), dim3(block), 0, stream, dangd, do32, n); },
               [&]{ hipLaunchKernelGGL(p_tan_f64, dim3(grid), dim3(block), 0, stream, dang64d, do64, n); },
               [&]{ hipLaunchKernelGGL(p_tan_dd,  dim3(grid), dim3(block), 0, stream, dangddd, dodd, n); });

  bench3("cotan", [&]{ hipLaunchKernelGGL(p_cotan_f32, dim3(grid), dim3(block), 0, stream, dangd, do32, n); },
                [&]{ hipLaunchKernelGGL(p_cotan_f64, dim3(grid), dim3(block), 0, stream, dang64d, do64, n); },
                [&]{ hipLaunchKernelGGL(p_cotan_dd,  dim3(grid), dim3(block), 0, stream, dangddd, dodd, n); });

  bench3("pow_int", [&]{ hipLaunchKernelGGL(p_pow_int_f32, dim3(grid), dim3(block), 0, stream, da_posd, do32, n); },
                  [&]{ hipLaunchKernelGGL(p_pow_int_f64, dim3(grid), dim3(block), 0, stream, da_pos64d, do64, n); },
                  [&]{ hipLaunchKernelGGL(p_pow_int_dd,  dim3(grid), dim3(block), 0, stream, da_posddd, dodd, n); });

  bench3("pow", [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_pow<f32>), dim3(grid), dim3(block), 0, stream, da_posd, dpowed, do32, n); },
              [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_pow<f64>), dim3(grid), dim3(block), 0, stream, da_pos64d, dpowe64d, do64, n); },
              [&]{ hipLaunchKernelGGL(p_pow_dd, dim3(grid), dim3(block), 0, stream, da_posddd, dpowedd, dodd, n); });

  bench3("log", [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_log_base<f32>), dim3(grid), dim3(block), 0, stream, da_posd, db_posd, do32, n); },
              [&]{ hipLaunchKernelGGL(HIP_KERNEL_NAME(p_log_base<f64>), dim3(grid), dim3(block), 0, stream, da_pos64d, db_pos64d, do64, n); },
              [&]{ hipLaunchKernelGGL(p_log_base_dd, dim3(grid), dim3(block), 0, stream, da_posddd, db_posddd, dodd, n); });

  // CPU throughput (TOp/s)
  auto print_cpu = [&](const char* name, double ops_s_f32, double ops_s_f64) {
    std::printf("Perf(TOp/s) %-7s | CPU f32:%7.4f  f64:%7.4f\n", name, ops_s_f32/1e12, ops_s_f64/1e12);
  };

  print_cpu("add", cpu_ops_per_s_binary(a, b, [](float x,float y){return x+y;}), cpu_ops_per_s_binary(a64, b64, [](double x,double y){return x+y;}));
  print_cpu("mul", cpu_ops_per_s_binary(a, b, [](float x,float y){return x*y;}), cpu_ops_per_s_binary(a64, b64, [](double x,double y){return x*y;}));
  print_cpu("div", cpu_ops_per_s_binary(a, b, [](float x,float y){return x/y;}), cpu_ops_per_s_binary(a64, b64, [](double x,double y){return x/y;}));
  print_cpu("sqrt", cpu_ops_per_s_unary(a_pos, [](float x){return std::sqrt(x);} ), cpu_ops_per_s_unary(a_pos64, [](double x){return std::sqrt(x);}));
  print_cpu("ln", cpu_ops_per_s_unary(a_pos, [](float x){return std::log(x);} ), cpu_ops_per_s_unary(a_pos64, [](double x){return std::log(x);}));
  print_cpu("exp", cpu_ops_per_s_unary(a_exp, [](float x){return std::exp(x);} ), cpu_ops_per_s_unary(a_exp64, [](double x){return std::exp(x);}));
  print_cpu("sin", cpu_ops_per_s_unary(ang, [](float x){return std::sin(x);} ), cpu_ops_per_s_unary(ang64, [](double x){return std::sin(x);}));
  print_cpu("cos", cpu_ops_per_s_unary(ang, [](float x){return std::cos(x);} ), cpu_ops_per_s_unary(ang64, [](double x){return std::cos(x);}));

  hip_check(hipStreamDestroy(stream), "stream destroy");
  hip_check(hipFree(da), "free da"); hip_check(hipFree(db), "free db"); hip_check(hipFree(do32), "free do32");
  hip_check(hipFree(da64d), "free da64d"); hip_check(hipFree(db64d), "free db64d"); hip_check(hipFree(do64), "free do64");
  hip_check(hipFree(dadd), "free dadd"); hip_check(hipFree(dbdd), "free dbdd"); hip_check(hipFree(dodd), "free dodd");
  hip_check(hipFree(da_posd), "free da_posd"); hip_check(hipFree(db_posd), "free db_posd"); hip_check(hipFree(da_expd), "free da_expd"); hip_check(hipFree(dangd), "free dangd"); hip_check(hipFree(dpowed), "free dpowed");
  hip_check(hipFree(da_pos64d), "free da_pos64d"); hip_check(hipFree(db_pos64d), "free db_pos64d"); hip_check(hipFree(da_exp64d), "free da_exp64d"); hip_check(hipFree(dang64d), "free dang64d"); hip_check(hipFree(dpowe64d), "free dpowe64d");
  hip_check(hipFree(da_posddd), "free da_posddd"); hip_check(hipFree(db_posddd), "free db_posddd"); hip_check(hipFree(da_expddd), "free da_expddd"); hip_check(hipFree(dangddd), "free dangddd"); hip_check(hipFree(dpowedd), "free dpowedd");

  return 0;
}
