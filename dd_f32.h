#pragma once

// dd_f32: double-single (aka df64) implemented as an unevaluated sum of two f32.
// Based on classic Dekker/Knuth error-free transforms and the GPU-focused survey
// in "Extended-Precision Floating-Point Numbers for GPU Computation" (Thall).

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>

namespace dd {

using f32 = float;
using f64 = double;

// Force-inlining is important here: these error-free transforms rely on
// strict evaluation/rounding behavior and should not incur call overhead.
#if defined(__clang__)
#define DD_F32_FORCEINLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define DD_F32_FORCEINLINE __forceinline
#else
#define DD_F32_FORCEINLINE inline __attribute__((always_inline))
#endif

#define DD_F32_HDINLINE __host__ __device__ static DD_F32_FORCEINLINE

struct dd_f32 {
  f32 hi;
  f32 lo;
};

DD_F32_HDINLINE f32 force_eval_f32(f32 x) {
  // Prevent overly-aggressive optimizer algebra (important for error-free transforms)
  // without forcing a device-side local-memory spill.
#if defined(__HIP_DEVICE_COMPILE__)
  asm volatile("" : "+v"(x));
  return x;
#else
  volatile f32 v = x;
  return v;
#endif
}

DD_F32_HDINLINE dd_f32 dd_f32_from_f32(f32 hi, f32 lo) {
  // Normalize to a non-overlapping-ish representation.
  // Use general TwoSum (works regardless of magnitude ordering).
  f32 x = hi + lo;
  f32 bvirtual = x - hi;
  f32 avirtual = x - bvirtual;
  f32 broundoff = lo - bvirtual;
  f32 aroundoff = hi - avirtual;
  f32 y = aroundoff + broundoff;
  return {x, y};
}

DD_F32_HDINLINE dd_f32 dd_f32_from_f32(f32 a) { return {a, 0.0f}; }

DD_F32_HDINLINE dd_f32 dd_f32_from_f64(f64 a) {
  f32 hi = (f32)a;
  f32 lo = (f32)(a - (f64)hi);
  return dd_f32_from_f32(hi, lo);
}

// Compatibility with the user-provided (likely typo) signature.
DD_F32_HDINLINE dd_f32 dd_f32_to_f64(f64 a) { return dd_f32_from_f64(a); }

DD_F32_HDINLINE void dd_f32_to_f32(dd_f32 v, f32* hi, f32* lo) {
  *hi = v.hi;
  *lo = v.lo;
}

DD_F32_HDINLINE f64 dd_f32_as_f64(dd_f32 v) { return (f64)v.hi + (f64)v.lo; }

// --- Error-free transforms ---------------------------------------------------

DD_F32_HDINLINE dd_f32 fast_two_sum(f32 a, f32 b) {
  // Requires |a| >= |b|.
  f32 x = a + b;
  f32 y = b - (x - a);
  return {x, y};
}

DD_F32_HDINLINE dd_f32 two_sum(f32 a, f32 b) {
  f32 x = a + b;
  f32 bvirtual = x - a;
  f32 avirtual = x - bvirtual;
  f32 broundoff = b - bvirtual;
  f32 aroundoff = a - avirtual;
  f32 y = aroundoff + broundoff;
  return {x, y};
}

DD_F32_HDINLINE dd_f32 quick_two_sum(f32 a, f32 b) {
  // Same as fast_two_sum but name matches paper.
  return fast_two_sum(a, b);
}

DD_F32_HDINLINE dd_f32 split(f32 a) {
  // For IEEE-754 float (p=24), a standard splitter is 2^13+1.
  // This is only used if FMA is not available.
  constexpr f32 splitter = (f32)(8193.0f); // 2^13 + 1
  f32 t = splitter * a;
  f32 ahi = t - (t - a);
  f32 alo = a - ahi;
  return {ahi, alo};
}

DD_F32_HDINLINE dd_f32 two_prod_fma(f32 a, f32 b) {
  // Algorithm 5 (Two-Product-FMA) from the paper.
  f32 x = force_eval_f32(a * b);
#if defined(__HIP_DEVICE_COMPILE__)
  f32 y = fmaf(a, b, -x);
#else
  f32 y = std::fmaf(a, b, -x);
#endif
  return {x, y};
}

DD_F32_HDINLINE dd_f32 two_prod_split(f32 a, f32 b) {
  // Algorithm 4 (Two-Product) from the paper (Dekker split based).
  f32 p = force_eval_f32(a * b);
  dd_f32 aS = split(a);
  dd_f32 bS = split(b);
  f32 err = ((aS.hi * bS.hi - p) + aS.hi * bS.lo + aS.lo * bS.hi) + aS.lo * bS.lo;
  return {p, err};
}

// TwoProd selection:
// - On HIP device builds we default to the fused-FMA version (much faster).
// - You can force the conservative split-based version with DD_F32_USE_SPLIT_TWOPROD.
// - On host builds we default to split-based unless DD_F32_USE_FMA_TWOPROD is defined.
DD_F32_HDINLINE dd_f32 two_prod(f32 a, f32 b) {
#if defined(DD_F32_USE_FMA_TWOPROD)
  return two_prod_fma(a, b);
#elif defined(__HIP_DEVICE_COMPILE__) && !defined(DD_F32_USE_SPLIT_TWOPROD)
  return two_prod_fma(a, b);
#else
  return two_prod_split(a, b);
#endif
}

DD_F32_HDINLINE dd_f32 renorm(dd_f32 a) {
  // Ensure |hi| >= |lo| and (somewhat) non-overlapping.
  // If ordering already ok, QuickTwoSum is sufficient.
  f32 ahi = a.hi;
  f32 alo = a.lo;
  if (fabsf(ahi) >= fabsf(alo)) return quick_two_sum(ahi, alo);
  return two_sum(ahi, alo);
}

// --- Core arithmetic ---------------------------------------------------------

DD_F32_HDINLINE dd_f32 dd_add(dd_f32 a, dd_f32 b) {
  // Faster variant: exact high-part sum + accumulate low parts, then renormalize.
  dd_f32 s = two_sum(a.hi, b.hi);
  s.lo += (a.lo + b.lo);
  return quick_two_sum(s.hi, s.lo);
}

// More accurate add (slower). Useful for cancellation-heavy polynomials.
DD_F32_HDINLINE dd_f32 dd_add_accurate(dd_f32 a, dd_f32 b) {
  dd_f32 s = two_sum(a.hi, b.hi);
  dd_f32 t = two_sum(a.lo, b.lo);
  s.lo += t.hi;
  s = quick_two_sum(s.hi, s.lo);
  s.lo += t.lo;
  s = quick_two_sum(s.hi, s.lo);
  return s;
}

DD_F32_HDINLINE dd_f32 dd_add(dd_f32 a, f32 b) { return dd_add(a, dd_f32_from_f32(b)); }
DD_F32_HDINLINE dd_f32 dd_add(f32 a, f32 b) { return two_sum(a, b); }

DD_F32_HDINLINE dd_f32 dd_sub(dd_f32 a, dd_f32 b) {
  dd_f32 s = two_sum(a.hi, -b.hi);
  s.lo += (a.lo - b.lo);
  return quick_two_sum(s.hi, s.lo);
}

DD_F32_HDINLINE dd_f32 dd_sub(dd_f32 a, f32 b) { return dd_add(a, -b); }
DD_F32_HDINLINE dd_f32 dd_sub(f32 a, f32 b) { return two_sum(a, -b); }

DD_F32_HDINLINE dd_f32 dd_mul(dd_f32 a, dd_f32 b) {
  dd_f32 p = two_prod(a.hi, b.hi);
#if defined(__HIP_DEVICE_COMPILE__)
  p.lo = fmaf(a.hi, b.lo, p.lo);
  p.lo = fmaf(a.lo, b.hi, p.lo);
#else
  p.lo = std::fmaf(a.hi, b.lo, p.lo);
  p.lo = std::fmaf(a.lo, b.hi, p.lo);
#endif
  // a.lo*b.lo is ~2^-48 relative; skip for speed.
  return quick_two_sum(p.hi, p.lo);
}

// Full-precision dd multiply (slower): keeps the a.lo*b.lo term.
DD_F32_HDINLINE dd_f32 dd_mul_full(dd_f32 a, dd_f32 b) {
  dd_f32 p = two_prod(a.hi, b.hi);
#if defined(__HIP_DEVICE_COMPILE__)
  p.lo = fmaf(a.hi, b.lo, p.lo);
  p.lo = fmaf(a.lo, b.hi, p.lo);
#else
  p.lo = std::fmaf(a.hi, b.lo, p.lo);
  p.lo = std::fmaf(a.lo, b.hi, p.lo);
#endif
  p.lo = p.lo + a.lo * b.lo;
  return quick_two_sum(p.hi, p.lo);
}

DD_F32_HDINLINE dd_f32 dd_mul(dd_f32 a, f32 b) {
  dd_f32 p = two_prod(a.hi, b);
#if defined(__HIP_DEVICE_COMPILE__)
  p.lo = fmaf(a.lo, b, p.lo);
#else
  p.lo = std::fmaf(a.lo, b, p.lo);
#endif
  return quick_two_sum(p.hi, p.lo);
}

DD_F32_HDINLINE dd_f32 dd_mul(f32 a, f32 b) { return two_prod(a, b); }

// Karp/Markstein mixed-precision division (paper Algorithm 6).
DD_F32_HDINLINE dd_f32 dd_div(dd_f32 B, dd_f32 A) {
  // Karp/Markstein style mixed-precision division (paper Algorithm 6),
  // but we keep the remainder term in dd to reliably reach ~48 bits.
  f32 xn = force_eval_f32(1.0f / A.hi);
  f32 yn = force_eval_f32(B.hi * xn);
  dd_f32 Ayn = dd_mul(A, yn);
  dd_f32 diff = dd_sub(B, Ayn);
  dd_f32 corr = dd_mul(diff, xn);
  return dd_add(dd_f32_from_f32(yn), corr);
}

DD_F32_HDINLINE dd_f32 dd_div(dd_f32 a, f32 b) { return dd_div(a, dd_f32_from_f32(b)); }
DD_F32_HDINLINE dd_f32 dd_div(f32 a, f32 b) { return dd_div(dd_f32_from_f32(a), dd_f32_from_f32(b)); }

// --- Sign / reciprocal -------------------------------------------------------

DD_F32_HDINLINE int dd_sign(dd_f32 a) {
  if (a.hi > 0.0f) return 1;
  if (a.hi < 0.0f) return -1;
  if (a.lo > 0.0f) return 1;
  if (a.lo < 0.0f) return -1;
  return 0;
}

DD_F32_HDINLINE dd_f32 dd_inverse(dd_f32 a) { return dd_div(dd_f32_from_f32(1.0f), a); }

// --- sqrt / inv_sqrt ---------------------------------------------------------

// Karp/Markstein mixed-precision square-root (paper Algorithm 7).
DD_F32_HDINLINE dd_f32 dd_sqrt(dd_f32 A) {
#if defined(__HIP_DEVICE_COMPILE__)
  f32 xn = rsqrtf(A.hi);
#else
  f32 xn = 1.0f / sqrtf(A.hi);
#endif
  f32 yn = force_eval_f32(A.hi * xn);
  dd_f32 yn2 = dd_mul(yn, yn);
  dd_f32 diff = dd_sub(A, yn2);
  dd_f32 corr = dd_mul(diff, xn);
  corr.hi *= 0.5f;
  corr.lo *= 0.5f;
  return dd_add(dd_f32_from_f32(yn), corr);
}

DD_F32_HDINLINE dd_f32 dd_inv_sqrt(dd_f32 a) {
  // Newton-Raphson on y ~= 1/sqrt(a): y <- y*(1.5 - 0.5*a*y*y)
#if defined(__HIP_DEVICE_COMPILE__)
  f32 y0 = rsqrtf(a.hi);
#else
  f32 y0 = 1.0f / sqrtf(a.hi);
#endif
  dd_f32 y = dd_f32_from_f32(y0);
  dd_f32 y2 = dd_mul(y, y);
  dd_f32 ay2 = dd_mul(a, y2);
  dd_f32 t = dd_sub(dd_f32_from_f32(1.5f), dd_mul(ay2, 0.5f));
  y = dd_mul(y, t);
  return y;
}

// --- pow/log/exp/trig (dd-only, no fp64 backend) -----------------------------

DD_F32_HDINLINE dd_f32 dd_neg(dd_f32 a) { return {-a.hi, -a.lo}; }

DD_F32_HDINLINE dd_f32 dd_scalbn(dd_f32 a, int k) {
#if defined(__HIP_DEVICE_COMPILE__)
  return {scalbnf(a.hi, k), scalbnf(a.lo, k)};
#else
  return {std::scalbn(a.hi, k), std::scalbn(a.lo, k)};
#endif
}

// High-precision constants without using fp64 at runtime.
// Split into multiple f32 pieces so cancellation (e.g., near pi/2) remains accurate.
static constexpr f32 DD_LN2_A = 0.6931471825f;
static constexpr f32 DD_LN2_B = -1.904654212e-09f;
static constexpr f32 DD_LN2_C = -1.110223025e-16f;
static constexpr dd_f32 DD_LN2 = {DD_LN2_A, DD_LN2_B};

static constexpr f32 DD_INV_LN2_HI = 1.442695022f;
static constexpr f32 DD_INV_LN2_LO = 1.925963034e-08f;

static constexpr f32 DD_PIO2_A = 1.570796371f;
static constexpr f32 DD_PIO2_B = -4.371138829e-08f;
static constexpr f32 DD_PIO2_C = -1.776356839e-15f;
static constexpr dd_f32 DD_PIO2 = {DD_PIO2_A, DD_PIO2_B};

static constexpr f32 DD_INV_PIO2_HI = 0.6366197467f;
static constexpr f32 DD_INV_PIO2_LO = 2.568255297e-08f;

static constexpr f32 DD_SQRT2_HI = 1.414213538f;

DD_F32_HDINLINE dd_f32 dd_ln(dd_f32 a) {
  // Domain: a > 0 (tests use a_pos in [1e-3, 1000]).
  if (!(a.hi > 0.0f)) return dd_f32_from_f32(NAN);

#if defined(__HIP_DEVICE_COMPILE__)
  int e = ilogbf(a.hi);
  f32 m = scalbnf(a.hi, -e); // m in [1,2)
#else
  int e = std::ilogb(a.hi);
  f32 m = std::scalbn(a.hi, -e);
#endif

  // Bring mantissa into [1/sqrt(2), sqrt(2)] by shifting one bit if needed.
  if (m > DD_SQRT2_HI) {
    m *= 0.5f;
    e += 1;
  }

  dd_f32 md = dd_f32_from_f32(m);
  dd_f32 t = dd_div(dd_sub(md, 1.0f), dd_add(md, 1.0f));
  dd_f32 t2 = dd_mul_full(t, t);

  // ln(m) = 2 * t * (1 + t^2*(1/3 + t^2*(1/5 + ... + t^2*(1/17))))
  static constexpr dd_f32 INV3  = {0.3333333433f, -9.934107759e-09f};
  static constexpr dd_f32 INV5  = {0.200000003f,  -2.980232283e-09f};
  static constexpr dd_f32 INV7  = {0.1428571492f, -6.386212004e-09f};
  static constexpr dd_f32 INV9  = {0.1111111119f, -8.278422947e-10f};
  static constexpr dd_f32 INV11 = {0.09090909362f,-2.709302116e-09f};
  static constexpr dd_f32 INV13 = {0.07692307979f,-2.865607973e-09f};
  static constexpr dd_f32 INV15 = {0.06666667014f,-3.476937627e-09f};
  static constexpr dd_f32 INV17 = {0.05882352963f,-2.191347243e-10f};

  dd_f32 p = INV17;
  p = dd_add(INV15, dd_mul_full(t2, p));
  p = dd_add(INV13, dd_mul_full(t2, p));
  p = dd_add(INV11, dd_mul_full(t2, p));
  p = dd_add(INV9,  dd_mul_full(t2, p));
  p = dd_add(INV7,  dd_mul_full(t2, p));
  p = dd_add(INV5,  dd_mul_full(t2, p));
  p = dd_add(INV3,  dd_mul_full(t2, p));
  p = dd_add(dd_mul_full(t2, p), 1.0f);

  dd_f32 lnm = dd_mul(dd_mul_full(t, p), 2.0f);
  dd_f32 le = dd_mul(DD_LN2, (f32)e);
  le = dd_add(le, (f32)(DD_LN2_C * (f32)e));
  return dd_add(lnm, le);
}

DD_F32_HDINLINE dd_f32 dd_exp(dd_f32 a) {
  // Range in tests: a in [-10, 10].
#if defined(__HIP_DEVICE_COMPILE__)
  f32 zf = fmaf(a.hi, DD_INV_LN2_LO, a.hi * DD_INV_LN2_HI);
  int k = (int)nearbyintf(zf);
#else
  f32 zf = std::fmaf(a.hi, DD_INV_LN2_LO, a.hi * DD_INV_LN2_HI);
  int k = (int)std::nearbyint(zf);
#endif
  dd_f32 r = dd_sub(a, dd_mul(DD_LN2, (f32)k));
  r = dd_sub(r, (f32)(DD_LN2_C * (f32)k));

  // Ensure |r| <= ln2/2 (robust against k rounding error).
  constexpr f32 LN2_HALF_HI = 0.3465735912f;
  if (r.hi > LN2_HALF_HI) {
    k += 1;
    r = dd_sub(dd_sub(r, DD_LN2), DD_LN2_C);
  } else if (r.hi < -LN2_HALF_HI) {
    k -= 1;
    r = dd_add(dd_add(r, DD_LN2), DD_LN2_C);
  }

  // exp(r) on |r| <= ln2/2 via Taylor (up to 12!).
  static constexpr dd_f32 INVFACT2  = {0.5f, 0.0f};
  static constexpr dd_f32 INVFACT3  = {0.1666666716f, -4.967053879e-09f};
  static constexpr dd_f32 INVFACT4  = {0.04166666791f,-1.24176347e-09f};
  static constexpr dd_f32 INVFACT5  = {0.008333333768f,-4.346172033e-10f};
  static constexpr dd_f32 INVFACT6  = {0.001388888923f,-3.363109444e-11f};
  static constexpr dd_f32 INVFACT7  = {0.0001984127011f,-2.725596875e-12f};
  static constexpr dd_f32 INVFACT8  = {2.480158764e-05f,-3.406996094e-13f};
  static constexpr dd_f32 INVFACT9  = {2.755731884e-06f, 3.793571224e-14f};
  static constexpr dd_f32 INVFACT10 = {2.755731998e-07f,-7.575112209e-15f};
  static constexpr dd_f32 INVFACT11 = {2.505210794e-08f, 4.417623045e-16f};
  static constexpr dd_f32 INVFACT12 = {2.087675588e-09f, 1.108283981e-16f};

  dd_f32 p = INVFACT12;
  p = dd_add(INVFACT11, dd_mul_full(r, p));
  p = dd_add(INVFACT10, dd_mul_full(r, p));
  p = dd_add(INVFACT9,  dd_mul_full(r, p));
  p = dd_add(INVFACT8,  dd_mul_full(r, p));
  p = dd_add(INVFACT7,  dd_mul_full(r, p));
  p = dd_add(INVFACT6,  dd_mul_full(r, p));
  p = dd_add(INVFACT5,  dd_mul_full(r, p));
  p = dd_add(INVFACT4,  dd_mul_full(r, p));
  p = dd_add(INVFACT3,  dd_mul_full(r, p));
  p = dd_add(INVFACT2,  dd_mul_full(r, p));

  dd_f32 er = dd_add(dd_mul_full(r, dd_add(dd_mul_full(r, p), 1.0f)), 1.0f);
  return dd_scalbn(er, k);
}

DD_F32_HDINLINE dd_f32 dd_pow_int(dd_f32 a, int exponent) {
  if (exponent == 0) return dd_f32_from_f32(1.0f);
  if (exponent < 0) return dd_inverse(dd_pow_int(a, -exponent));
  dd_f32 base = a;
  dd_f32 res = dd_f32_from_f32(1.0f);
  int e = exponent;
  while (e) {
    if (e & 1) res = dd_mul(res, base);
    e >>= 1;
    if (e) base = dd_mul(base, base);
  }
  return res;
}

DD_F32_HDINLINE dd_f32 dd_pow(dd_f32 a, dd_f32 exponent) {
  // a^b = exp(b*ln(a))
  return dd_exp(dd_mul(exponent, dd_ln(a)));
}

DD_F32_HDINLINE dd_f32 dd_log(dd_f32 a, dd_f32 base) {
  return dd_div(dd_ln(a), dd_ln(base));
}

DD_F32_HDINLINE dd_f32 dd_log_10(dd_f32 a, dd_f32 base) {
  // Kept to match requested signature; intended usage: base = 10.
  return dd_log(a, base);
}

DD_F32_HDINLINE void dd_sincos(dd_f32 angle_rad, dd_f32* s, dd_f32* c) {
#if defined(__HIP_DEVICE_COMPILE__)
  f32 zf = fmaf(angle_rad.hi, DD_INV_PIO2_LO, angle_rad.hi * DD_INV_PIO2_HI);
  int k = (int)nearbyintf(zf);
#else
  f32 zf = std::fmaf(angle_rad.hi, DD_INV_PIO2_LO, angle_rad.hi * DD_INV_PIO2_HI);
  int k = (int)std::nearbyint(zf);
#endif
  dd_f32 r = dd_sub(angle_rad, dd_mul(DD_PIO2, (f32)k));
  r = dd_sub(r, (f32)(DD_PIO2_C * (f32)k));

  // Ensure |r| <= pi/4 (robust against k rounding error).
  constexpr f32 PIO4_HI = 0.7853981853f;
  if (r.hi > PIO4_HI) {
    k += 1;
    r = dd_sub(dd_sub(r, DD_PIO2), DD_PIO2_C);
  } else if (r.hi < -PIO4_HI) {
    k -= 1;
    r = dd_add(dd_add(r, DD_PIO2), DD_PIO2_C);
  }

  // fdlibm-style polynomials for sin/cos on reduced r.
  static constexpr dd_f32 S1 = {-0.1666666716f, 4.967053879e-09f};
  static constexpr dd_f32 S2 = {0.008333333768f,-4.346280558e-10f};
  static constexpr dd_f32 S3 = {-0.0001984127011f, 2.839715659e-12f};
  static constexpr dd_f32 S4 = {2.75573143e-06f,-5.900851639e-14f};
  static constexpr dd_f32 S5 = {-2.505075969e-08f,-5.645685822e-16f};
  static constexpr dd_f32 S6 = {1.589691018e-10f,-2.250000749e-18f};

  // Cos Taylor coefficients up to x^16.
  static constexpr dd_f32 C1 = {0.04166666791f, -1.24176347e-09f};
  static constexpr dd_f32 C2 = {-0.001388888923f, 3.363109444e-11f};
  static constexpr dd_f32 C3 = {2.480158764e-05f, -3.406996094e-13f};
  static constexpr dd_f32 C4 = {-2.755731998e-07f, 7.575112209e-15f};
  static constexpr dd_f32 C5 = {2.087675588e-09f, 1.108283981e-16f};
  static constexpr dd_f32 C6 = {-1.147074536e-11f, -2.372207689e-19f};
  static constexpr dd_f32 C7 = {4.779477256e-14f, 7.625444044e-22f};

  // Use r = x + y with y small (from reduction tail). Evaluate polynomials in x,
  // then apply first-order correction in y to improve accuracy near zeros.
  dd_f32 x = dd_f32_from_f32(r.hi);
  f32 y = r.lo;

  dd_f32 z = dd_mul_full(x, x);

  dd_f32 ps = S6;
  ps = dd_add(S5, dd_mul_full(z, ps));
  ps = dd_add(S4, dd_mul_full(z, ps));
  ps = dd_add(S3, dd_mul_full(z, ps));
  ps = dd_add(S2, dd_mul_full(z, ps));
  ps = dd_add(S1, dd_mul_full(z, ps));
  dd_f32 sin_x = dd_add(x, dd_mul_full(dd_mul_full(x, z), ps));

  dd_f32 pc = C7;
  pc = dd_add(C6, dd_mul_full(z, pc));
  pc = dd_add(C5, dd_mul_full(z, pc));
  pc = dd_add(C4, dd_mul_full(z, pc));
  pc = dd_add(C3, dd_mul_full(z, pc));
  pc = dd_add(C2, dd_mul_full(z, pc));
  pc = dd_add(C1, dd_mul_full(z, pc));
  dd_f32 z2 = dd_mul_full(z, z);
  dd_f32 w = dd_add_accurate(dd_mul(z, -0.5f), dd_f32_from_f32(1.0f));
  dd_f32 cos_x = dd_add_accurate(w, dd_mul_full(z2, pc));

  dd_f32 sin_r = dd_add(sin_x, dd_mul(cos_x, y));
  dd_f32 cos_r = dd_sub(cos_x, dd_mul(sin_x, y));

  int q = k & 3;
  if (q == 0) {
    *s = sin_r;
    *c = cos_r;
  } else if (q == 1) {
    *s = cos_r;
    *c = dd_neg(sin_r);
  } else if (q == 2) {
    *s = dd_neg(sin_r);
    *c = dd_neg(cos_r);
  } else {
    *s = dd_neg(cos_r);
    *c = sin_r;
  }
}

DD_F32_HDINLINE dd_f32 dd_sin(dd_f32 angle_rad) {
  dd_f32 s, c;
  dd_sincos(angle_rad, &s, &c);
  return s;
}

DD_F32_HDINLINE dd_f32 dd_cos(dd_f32 angle_rad) {
  dd_f32 s, c;
  dd_sincos(angle_rad, &s, &c);
  return c;
}

// User asked for dd_sin_cos(angle) returning dd_f32; we return sin(angle).
// Use dd_sincos(angle, &s, &c) if you need both.
DD_F32_HDINLINE dd_f32 dd_sin_cos(dd_f32 angle_rad) { return dd_sin(angle_rad); }

DD_F32_HDINLINE dd_f32 dd_tan(dd_f32 angle_rad) {
  dd_f32 s, c;
  dd_sincos(angle_rad, &s, &c);
  return dd_div(s, c);
}

DD_F32_HDINLINE dd_f32 dd_cotan(dd_f32 angle_rad) {
  dd_f32 s, c;
  dd_sincos(angle_rad, &s, &c);
  return dd_div(c, s);
}

} // namespace dd
