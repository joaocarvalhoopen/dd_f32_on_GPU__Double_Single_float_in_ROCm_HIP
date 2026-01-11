# dd_f32 on GPU - Double Single float in ROCm - HIP
A simple and fast lib to get close to 48 bits of significance in the mantissa on GPU.

This project implements a **double-single** (“df64”) type `dd_f32` for GPUs: a number is represented as `hi + lo`, where both parts are **float32**.
The goal is ~48-bit mantissa precision (vs 24-bit for float32) while primarily using fp32 hardware.

Key reference: *Extended Precision Floating Point Numbers for GPU Computation* (Andrew Thall).

## What’s included

- `dd_f32.h`
  - `dd::dd_f32 { float hi, lo; }`
  - Error-free transforms: TwoSum / QuickTwoSum / Split / TwoProd
  - Core ops: `dd_add`, `dd_sub`, `dd_mul`, `dd_div`, `dd_inverse`, `dd_sqrt`, `dd_inv_sqrt`
  - dd-only transcendentals (no fp64 backend at runtime): `dd_ln`, `dd_exp`, `dd_sincos` (+ `dd_sin`, `dd_cos`), `dd_tan`, `dd_cotan`, `dd_pow`, `dd_log`

- Benchmarks / tests
  - `test_dd_f32.hip`: smaller accuracy + perf summary
  - `test_dd_f32_full.hip`: full accuracy table + throughput table
  - `some_complex_calculation.hip`: long mixed-operation chain benchmark (fp32 vs fp64 vs dd_f32)

## Requirements

- ROCm with HIP (`/opt/rocm/bin/hipcc` expected by default)
- An AMD GPU supported by ROCm
- Linux toolchain (make, clang as used by hipcc)

## Build

```bash
make
```

## Run

```bash
make run
```

Or run individual programs:

```bash
./test_dd_f32
./test_dd_f32_full

# args: N chain_iters
./some_complex_calculation 1000000 64
```

## Understanding the accuracy metric

The test programs report **min relative bits**:

- `bits = -log2(|got - ref| / |ref|)`
- If `ref == 0`, it falls back to absolute error (`exact` if also 0).
- If the relative error is exactly 0 for the sampled inputs, the program prints **`exact`**.

Notes:
- Cancellation-heavy cases (e.g., `cos(x)` near 0) can look worse in *relative* bits even if absolute error is tiny.

## Performance numbers

- `test_dd_f32_full.hip` prints **Perf(TOp/s)** = operations-per-second / 1e12.
- `some_complex_calculation.hip` prints **GEval/s** for a long mixed-op chain, where one “eval” is one loop iteration of the chain.

These are **throughput benchmarks**, not “hardware TFLOPs”.

## Build options / knobs

TwoProd implementation can be selected (trade accuracy robustness vs speed):

- Default behavior:
  - On device: prefers FMA-based TwoProd when available.
  - Split-based TwoProd is available and can be forced.

Common knobs (see `dd_f32.h`):

- `DD_F32_USE_SPLIT_TWOPROD`
  - Forces split-based TwoProd (more conservative/robust).

You can pass defines via `HIPFLAGS`:

```bash
make HIPFLAGS='-O3 -DNDEBUG --offload-arch=native -DDD_F32_USE_SPLIT_TWOPROD'
```

## File map

- `dd_f32.h` — implementation
- `test_dd_f32.hip` — basic correctness + perf
- `test_dd_f32_full.hip` — full tables
- `some_complex_calculation.hip` — long chained mixed-op perf comparison
- `Makefile` — build + run targets

## Notes / known limitations

- dd trig (`cos/tan/cotan`) accuracy is sensitive to range reduction and relative-error reporting near zeros; you may see lower *min relative bits* than for `sin`.
- This repository is focused on GPU-side dd arithmetic performance/accuracy trade-offs; it is not a drop-in IEEE-754 extended precision type.

## Test programs output

### test_dd_f32.hip

```
./test_dd_f32
GPU: AMD Radeon RX 9070 XT
N=100000, ITERS=1024
Accuracy (min rel bits) add   | f32:  24.00  f64:  exact  dd_f32:  exact
Accuracy (min rel bits) mul   | f32:  24.01  f64:  exact  dd_f32:  exact
Accuracy (min rel bits) div   | f32:  24.00  f64:  53.01  dd_f32:  47.12
Accuracy (min rel bits) sqrt  | f32:  24.00  f64:  53.01  dd_f32:  45.81
Accuracy (min rel bits) ln    | f32:  22.57  f64:  52.99  dd_f32:  46.16
Accuracy (min rel bits) exp   | f32:  23.56  f64:  52.73  dd_f32:  45.07
Accuracy (min rel bits) sin   | f32:  23.03  f64:  52.83  dd_f32:  23.48
Accuracy (min rel bits) cos   | f32:  23.05  f64:  52.84  dd_f32:  24.29
Perf add  | f32:  3594.75 Gop/s  f64:   186.03 Gop/s  dd_f32:   626.53 Gop/s  (kernel ms: 0.114 / 2.202 / 0.654)
Perf mul  | f32:  3869.70 Gop/s  f64:   204.87 Gop/s  dd_f32:   881.26 Gop/s  (kernel ms: 0.106 / 1.999 / 0.465)
Perf div  | f32:   469.10 Gop/s  f64:    23.82 Gop/s  dd_f32:   232.19 Gop/s  (kernel ms: 0.873 / 17.197 / 1.764)
CPU fp64 perf add :    3.065 Gop/s (0.033s)
CPU fp64 perf mul :    2.045 Gop/s (0.050s)
CPU fp64 perf div :    0.547 Gop/s (0.187s)

```

### test_dd_f32_full.hip

```
./test_dd_f32_full
GPU: AMD Radeon RX 9070 XT
N=100000, ITERS=1024, ILP=4
Prec(bits,min rel) add     | CPU ref vs GPU f32: 24.00 f64: exact dd: exact
Prec(bits,min rel) sub     | CPU ref vs GPU f32: 24.00 f64: exact dd: exact
Prec(bits,min rel) mul     | CPU ref vs GPU f32: 24.00 f64: exact dd: exact
Prec(bits,min rel) div     | CPU ref vs GPU f32: 24.00 f64: 53.00 dd: 47.12
Prec(bits,min rel) inverse | CPU ref vs GPU f32: 24.01 f64: 53.01 dd: 47.49
Prec(bits,min rel) sqrt    | CPU ref vs GPU f32: 24.01 f64: 53.01 dd: 45.70
Prec(bits,min rel) inv_sqrt | CPU ref vs GPU f32: 23.39 f64: 52.44 dd: 46.07
Prec(bits,min rel) ln      | CPU ref vs GPU f32: 22.61 f64: 53.00 dd: 45.91
Prec(bits,min rel) exp     | CPU ref vs GPU f32: 23.56 f64: 52.74 dd: 45.11
Prec(bits,min rel) sin     | CPU ref vs GPU f32: 23.17 f64: 52.86 dd: 46.82
Prec(bits,min rel) cos     | CPU ref vs GPU f32: 23.00 f64: 52.80 dd: 39.31
Prec(bits,min rel) tan     | CPU ref vs GPU f32: 22.79 f64: 52.28 dd: 39.30
Prec(bits,min rel) cotan   | CPU ref vs GPU f32: 22.53 f64: 52.11 dd: 39.31
Prec(bits,min rel) pow_int | CPU ref vs GPU f32: 22.92 f64: 52.03 dd: 45.71
Prec(bits,min rel) pow     | CPU ref vs GPU f32: 22.89 f64: 52.03 dd: 42.63
Prec(bits,min rel) log     | CPU ref vs GPU f32: 22.07 f64: 51.65 dd: 45.80
Perf(TOp/s) add     | GPU f32:  3.537  f64:  0.186  dd:  0.620
Perf(TOp/s) sub     | GPU f32:  3.750  f64:  0.192  dd:  0.665             
Perf(TOp/s) mul     | GPU f32:  3.879  f64:  0.203  dd:  0.871
Perf(TOp/s) div     | GPU f32:  0.489  f64:  0.025  dd:  0.237
Perf(TOp/s) inverse | GPU f32:  0.606  f64:  0.030  dd:  0.214
Perf(TOp/s) sqrt    | GPU f32:  0.604  f64:  0.031  dd:  0.296
Perf(TOp/s) inv_sqrt | GPU f32:  1.704  f64:  0.016  dd:  0.235
Perf(TOp/s) ln      | GPU f32:  1.004  f64:  0.005  dd:  1.633
Perf(TOp/s) exp     | GPU f32:  0.800  f64:  0.023  dd:  0.045
Perf(TOp/s) sin     | GPU f32:  0.436  f64:  0.010  dd:  0.031
Perf(TOp/s) cos     | GPU f32:  0.487  f64:  0.010  dd:  0.031
Perf(TOp/s) tan     | GPU f32:  0.379  f64:  0.006  dd:  0.027
Perf(TOp/s) cotan   | GPU f32:  0.264  f64:  0.005  dd:  0.027             
Perf(TOp/s) pow_int | GPU f32:  0.058  f64:  0.003  dd:  0.300
Perf(TOp/s) pow     | GPU f32:  0.055  f64:  0.003  dd:  0.019             
Perf(TOp/s) log     | GPU f32:  0.453  f64:  0.005  dd:  0.036
Perf(TOp/s) add     | CPU f32: 0.0104  f64: 0.0104
Perf(TOp/s) mul     | CPU f32: 0.0072  f64: 0.0075
Perf(TOp/s) div     | CPU f32: 0.0029  f64: 0.0014
Perf(TOp/s) sqrt    | CPU f32: 0.0010  f64: 0.0007
Perf(TOp/s) ln      | CPU f32: 0.0006  f64: 0.0005
Perf(TOp/s) exp     | CPU f32: 0.0019  f64: 0.0004
Perf(TOp/s) sin     | CPU f32: 0.0006  f64: 0.0004
Perf(TOp/s) cos     | CPU f32: 0.0006  f64: 0.0004

```

### some_complex_calculation.hip

```
./some_complex_calculation
GPU: AMD Radeon RX 9070 XT
Complex chain benchmark (fp32, fp64, dd_f32)
N=10000000, chain_iters=64
Throughput (GEval/s): fp32=71.408  fp64=2.204  dd_f32=10.159
Kernel time (ms):      fp32=8.963  fp64=290.428  dd_f32=63.000

```

## License
MIT Open Source License

## Have fun
Best regards, <br>
Joao Carvalho
