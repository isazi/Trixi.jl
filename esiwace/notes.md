
# Notes on performance

## Profiling 2026-01-28

Using the tuned values from earlier today.

### reference - A100 (snellius)

```
Device-side activity: GPU was busy for 39.6 s (63.79% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name                                                                                                              ⋯
├──────────┼────────────┼───────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
│   39.71% │    24.65 s │  1000 │  24.65 ms ± 0.27   ( 24.54 ‥ 30.55)  │ gpu__flux_differencing_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo< ⋯
│   10.02% │     6.22 s │  1000 │   6.22 ms ± 0.02   (  6.18 ‥ 6.31)   │ gpu_surface_integral_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<In ⋯
│    5.72% │     3.55 s │  1000 │   3.55 ms ± 0.03   (   3.5 ‥ 3.91)   │ gpu_prolong2interfaces_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo< ⋯
│    2.73% │     1.69 s │  1000 │   1.69 ms ± 0.01   (  1.68 ‥ 1.91)   │ gpu_interface_flux_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int6 ⋯
│    2.28% │     1.41 s │  1000 │   1.41 ms ± 0.02   (  1.41 ‥ 1.81)   │ gpu__apply_jacobian_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int ⋯
│    1.80% │     1.12 s │  1800 │ 622.31 µs ± 2.02   (612.26 ‥ 628.23) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int ⋯
│    0.66% │   412.5 ms │  1000 │  412.5 µs ± 1.48   (409.84 ‥ 417.95) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int ⋯
│    0.36% │  221.81 ms │   200 │   1.11 ms ± 0.0    (   1.1 ‥ 1.16)   │ gpu_max_scaled_speed_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<In ⋯
│    0.30% │  183.92 ms │  1000 │ 183.92 µs ± 2.74   (177.86 ‥ 188.59) │ gpu_fill_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int64>>>, NDRa ⋯
│    0.13% │   79.51 ms │   201 │ 395.58 µs ± 1.46   (393.15 ‥ 409.13) │ [copy device to device memory]                                                                                    ⋯
│    0.08% │   46.92 ms │   199 │  235.8 µs ± 1.9    (231.27 ‥ 241.28) │ partial_mapreduce_grid(INFINITE_OR_GIANT, _, Bool, CartesianIndices<1, Tuple<OneTo<Int64>>>, CartesianIndices<1,  ⋯
│    0.00% │  987.05 µs │   200 │   4.94 µs ± 0.18   (  4.53 ‥ 5.72)   │ partial_mapreduce_grid(identity, max, Float64, CartesianIndices<1, Tuple<OneTo<Int64>>>, CartesianIndices<1, Tupl ⋯
│    0.00% │  861.88 µs │   199 │   4.33 µs ± 0.18   (  4.05 ‥ 4.77)   │ partial_mapreduce_grid(identity, _, Bool, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, CartesianIndice ⋯
│    0.00% │  851.39 µs │   200 │   4.26 µs ± 0.18   (  4.05 ‥ 4.77)   │ partial_mapreduce_grid(identity, max, Float64, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, CartesianI ⋯
│    0.00% │   696.9 µs │   399 │   1.75 µs ± 0.16   (  1.43 ‥ 2.38)   │ [copy device to pageable memory]                                                                                  ⋯
└──────────┴────────────┴───────┴──────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

### ijk_fusedloop - A100 (snellius)

```
Device-side activity: GPU was busy for 31.23 s (55.08% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name                                                                                                              ⋯
├──────────┼────────────┼───────┼──────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
│   28.68% │    16.26 s │  1000 │  16.26 ms ± 0.08   ( 16.24 ‥ 17.52)  │ gpu__exp_ijk_fusedloop_flux_differencing_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndic ⋯
│   10.98% │     6.22 s │  1000 │   6.22 ms ± 0.02   (  6.18 ‥ 6.32)   │ gpu_surface_integral_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<In ⋯
│    6.29% │     3.57 s │  1000 │   3.57 ms ± 0.04   (  3.51 ‥ 3.93)   │ gpu_prolong2interfaces_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo< ⋯
│    2.99% │     1.69 s │  1000 │   1.69 ms ± 0.02   (  1.67 ‥ 1.92)   │ gpu_interface_flux_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int6 ⋯
│    2.49% │     1.41 s │  1000 │   1.41 ms ± 0.03   (   1.4 ‥ 1.81)   │ gpu__apply_jacobian_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int ⋯
│    1.98% │     1.12 s │  1800 │ 622.26 µs ± 2.05   (611.78 ‥ 629.43) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int ⋯
│    0.73% │   412.5 ms │  1000 │  412.5 µs ± 1.46   (409.84 ‥ 416.76) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int ⋯
│    0.39% │   221.9 ms │   200 │   1.11 ms ± 0.01   (   1.1 ‥ 1.24)   │ gpu_max_scaled_speed_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<In ⋯
│    0.32% │  183.92 ms │  1000 │ 183.92 µs ± 2.75   (177.86 ‥ 187.64) │ gpu_fill_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<OneTo<Int64>>>, NDRa ⋯
│    0.14% │    79.5 ms │   201 │ 395.53 µs ± 1.52   (392.91 ‥ 409.84) │ [copy device to device memory]                                                                                    ⋯
│    0.08% │   46.92 ms │   199 │ 235.78 µs ± 1.72   ( 231.5 ‥ 240.33) │ partial_mapreduce_grid(INFINITE_OR_GIANT, _, Bool, CartesianIndices<1, Tuple<OneTo<Int64>>>, CartesianIndices<1,  ⋯
│    0.00% │  982.52 µs │   200 │   4.91 µs ± 0.19   (  4.53 ‥ 5.96)   │ partial_mapreduce_grid(identity, max, Float64, CartesianIndices<1, Tuple<OneTo<Int64>>>, CartesianIndices<1, Tupl ⋯
│    0.00% │  864.98 µs │   199 │   4.35 µs ± 0.17   (  3.81 ‥ 4.77)   │ partial_mapreduce_grid(identity, _, Bool, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, CartesianIndice ⋯
│    0.00% │  852.11 µs │   200 │   4.26 µs ± 0.18   (  4.05 ‥ 5.25)   │ partial_mapreduce_grid(identity, max, Float64, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64>>>, CartesianI ⋯
│    0.00% │  707.63 µs │   399 │   1.77 µs ± 0.17   (  1.43 ‥ 2.38)   │ [copy device to pageable memory]                                                                                  ⋯
└──────────┴────────────┴───────┴──────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

## Tuning 2026-01-28

Using **Julia 1.12** today (and until further notice).

### Raw data

#### A100 (snellius)

```
Tuning reference
        Best time: 0.033542788 s -- workgroupsize: 160
Tuning exp_split
        Best time: 0.032114261 s -- speedup: 1.0444826365458013 -- workgroupsize: 160
Tuning exp_index
        Best time: 0.033648544 s -- speedup: 0.9968570408276802 -- workgroupsize: 160
Tuning exp_ijk
        Best time: 0.016501226 s -- speedup: 2.032745203295803 -- workgroupsize: (32, 8)
Tuning exp_ijk_fusedloop
        Best time: 0.016364023 s -- speedup: 2.049788612494617 -- workgroupsize: (32, 8)
Tuning exp_ijk_split
        Best time: 0.030022688 s -- speedup: 1.117247995915622 -- workgroupsize: (32, 12)
Tuning exp_ijk_nosym
        Best time: 0.03002459 s -- speedup: 1.1171772204050079 -- workgroupsize: (32, 8)
Tuning exp_ijk_nosym_split
        Best time: 0.036404205 s -- speedup: 0.9213987230321331 -- workgroupsize: (32, 12)
Tuning exp_ijk_nosym_fusedloop
        Best time: 0.027793449 s -- speedup: 1.2068595013162993 -- workgroupsize: (32, 8)
Tuning exp_ijk_nosym_fusedloop_inter
        Best time: 0.029036533 s -- speedup: 1.1551925982347824 -- workgroupsize: (32, 8)
```

## Timing 2026-01-23

### reference - H100 (snellius)

```
─────────────────────────────────────────────────────────────────────────────────
           Trixi.jl                     Time                    Allocations
                               ───────────────────────   ────────────────────────
       Tot / % measured:            3.46s /  74.7%           47.7MiB /  34.6%

Section                ncalls     time    %tot     avg     alloc    %tot      avg
─────────────────────────────────────────────────────────────────────────────────
rhs!                       51    1.90s   73.7%  37.3ms    579KiB    3.4%  11.3KiB
  ~rhs!~                   51    1.90s   73.5%  37.2ms   23.9KiB    0.1%     480B
  volume integral          51   1.33ms    0.1%  26.2μs    109KiB    0.6%  2.13KiB
  prolong2interfaces       51   1.21ms    0.0%  23.8μs   97.6KiB    0.6%  1.91KiB
  interface flux           51   1.17ms    0.0%  23.0μs    114KiB    0.7%  2.23KiB
  reset ∂u/∂t              51   1.05ms    0.0%  20.7μs   64.2KiB    0.4%  1.26KiB
  surface integral         51   1.00ms    0.0%  19.7μs   96.0KiB    0.6%  1.88KiB
  Jacobian                 51    990μs    0.0%  19.4μs   74.5KiB    0.4%  1.46KiB
  prolong2boundaries       51   38.0μs    0.0%   746ns     0.00B    0.0%    0.00B
  boundary flux            51   27.6μs    0.0%   541ns     0.00B    0.0%    0.00B
  prolong2mortars          51   20.4μs    0.0%   400ns     0.00B    0.0%    0.00B
  source terms             51   7.11μs    0.0%   139ns     0.00B    0.0%    0.00B
  mortar flux              51   6.62μs    0.0%   130ns     0.00B    0.0%    0.00B
calculate dt               11    678ms   26.3%  61.6ms   15.9MiB   96.5%  1.45MiB
performance data            3    651μs    0.0%   217μs   16.1KiB    0.1%  5.37KiB
─────────────────────────────────────────────────────────────────────────────────
```

### ijk_fusedloop - H100 (snellius)

```
─────────────────────────────────────────────────────────────────────────────────
           Trixi.jl                     Time                    Allocations
                               ───────────────────────   ────────────────────────
       Tot / % measured:            3.81s /  77.7%           47.7MiB /  34.6%

Section                ncalls     time    %tot     avg     alloc    %tot      avg
─────────────────────────────────────────────────────────────────────────────────
rhs!                       51    2.28s   77.2%  44.8ms    591KiB    3.5%  11.6KiB
  ~rhs!~                   51    2.28s   77.0%  44.7ms   23.9KiB    0.1%     480B
  volume integral          51   1.54ms    0.1%  30.3μs    121KiB    0.7%  2.37KiB
  prolong2interfaces       51   1.32ms    0.0%  25.8μs   97.6KiB    0.6%  1.91KiB
  interface flux           51   1.19ms    0.0%  23.3μs    114KiB    0.7%  2.23KiB
  reset ∂u/∂t              51   1.14ms    0.0%  22.4μs   64.2KiB    0.4%  1.26KiB
  surface integral         51   1.08ms    0.0%  21.2μs   96.0KiB    0.6%  1.88KiB
  Jacobian                 51    989μs    0.0%  19.4μs   74.5KiB    0.4%  1.46KiB
  prolong2boundaries       51   35.7μs    0.0%   700ns     0.00B    0.0%    0.00B
  boundary flux            51   31.9μs    0.0%   626ns     0.00B    0.0%    0.00B
  prolong2mortars          51   24.7μs    0.0%   484ns     0.00B    0.0%    0.00B
  source terms             51   8.98μs    0.0%   176ns     0.00B    0.0%    0.00B
  mortar flux              51   6.45μs    0.0%   126ns     0.00B    0.0%    0.00B
calculate dt               11    674ms   22.8%  61.3ms   15.9MiB   96.4%  1.44MiB
performance data            3    628μs    0.0%   209μs   16.1KiB    0.1%  5.37KiB
─────────────────────────────────────────────────────────────────────────────────
```

## Tuning 2026-01-22

### Summary tables

#### Execution time (ms)

| **variant** | **A4000** | **A100** | **H100** |
| ----------- | --------- | -------- | -------- |
| reference | 8.86 | 6.15 | 3.98 |
| split | 8.5 | 6.27 | 3.88 |
| index | 8.9 | 6.3 | 4.0 |
| ijk | 1.06 | 0.19 | 0.17 |
| ijk_fusedloop | 1.03 | 0.19 | **0.17** |
| ijk_split | 1.42 | 0.25 | 0.2 |
| ijk_nosym | 1.95 | 0.55 | 0.2 |
| ijk_nosym_split | 2.26 | 0.63 | 0.23 |
| ijk_nosym_fusedloop | 1.89 | 0.55 | 0.2 |
| ijk_nosym_fusedloop_inter | 1.92 | 0.54 | 0.19 |

#### Speedup

| **variant** | **A4000** | **A100** | **H100** |
| ----------- | --------- | -------- | -------- |
| split | 1.04 | 0.98 | 1.02 |
| index | 0.99 | 0.97 | 0.99 |
| ijk | 8.35 | 31.27 | 23.17 |
| ijk_fusedloop | 8.59 | **31.53** | 23.42 |
| ijk_split | 6.22 | 23.86 | 19.82 |
| ijk_nosym | 4.52 | 11.10 | 19.50 |
| ijk_nosym_split | 3.92 | 9.65 | 17.29 |
| ijk_nosym_fusedloop | 4.68 | 11.02 | 19.54 |
| ijk_nosym_fusedloop_inter | 4.61 | 11.28 | 20.09 |

### Raw data

#### A4000 (DAS6-VU)

```
Tuning reference
        Best time: 0.008864381 s -- workgroupsize: 32
Tuning exp_split
        Best time: 0.008503194 s -- speedup: 1.0424766270180357 -- workgroupsize: 32
Tuning exp_index
        Best time: 0.008904896 s -- speedup: 0.995450255679572 -- workgroupsize: 32
Tuning exp_ijk
        Best time: 0.001060705 s -- speedup: 8.357065348046818 -- workgroupsize: (32, 3)
Tuning exp_ijk_fusedloop
        Best time: 0.00103095 s -- speedup: 8.598264707308791 -- workgroupsize: (32, 3)
Tuning exp_ijk_split
        Best time: 0.001423275 s -- speedup: 6.228157594280795 -- workgroupsize: (32, 9)
Tuning exp_ijk_nosym
        Best time: 0.001958846 s -- speedup: 4.525307757730826 -- workgroupsize: (32, 8)
Tuning exp_ijk_nosym_split
        Best time: 0.002260932 s -- speedup: 3.9206756328805996 -- workgroupsize: (32, 9)
Tuning exp_ijk_nosym_fusedloop
        Best time: 0.001892081 s -- speedup: 4.68499023033369 -- workgroupsize: (32, 4)
Tuning exp_ijk_nosym_fusedloop_inter
        Best time: 0.001922008 s -- speedup: 4.61204167724588 -- workgroupsize: (32, 2)
```

#### A100 (snellius)

```
Tuning reference
	Best time: 0.006151405 s -- workgroupsize: 96
Tuning exp_split
	Best time: 0.006276812 s -- speedup: 0.980020590070246 -- workgroupsize: 32
Tuning exp_index
	Best time: 0.006309166 s -- speedup: 0.9749949517891906 -- workgroupsize: 32
Tuning exp_ijk
	Best time: 0.000196716 s -- speedup: 31.27048638646577 -- workgroupsize: (32, 2)
Tuning exp_ijk_fusedloop
	Best time: 0.000195079 s -- speedup: 31.532891802808095 -- workgroupsize: (32, 2)
Tuning exp_ijk_split
	Best time: 0.000257741 s -- speedup: 23.866614159175295 -- workgroupsize: (32, 6)
Tuning exp_ijk_nosym
	Best time: 0.000553694 s -- speedup: 11.109755568960473 -- workgroupsize: (32, 8)
Tuning exp_ijk_nosym_split
	Best time: 0.00063689 s -- speedup: 9.65850460833111 -- workgroupsize: (32, 11)
Tuning exp_ijk_nosym_fusedloop
	Best time: 0.000558099 s -- speedup: 11.022067769338415 -- workgroupsize: (32, 8)
Tuning exp_ijk_nosym_fusedloop_inter
	Best time: 0.000545078 s -- speedup: 11.285366498005791 -- workgroupsize: (32, 8)
```

#### H100 (snellius)

```
Tuning reference
	Best time: 0.003984991 s -- workgroupsize: 32
Tuning exp_split
	Best time: 0.003885372 s -- speedup: 1.0256395011854722 -- workgroupsize: 32
Tuning exp_index
	Best time: 0.004002771 s -- speedup: 0.99555807714206 -- workgroupsize: 32
Tuning exp_ijk
	Best time: 0.000171939 s -- speedup: 23.176771994719058 -- workgroupsize: (128, 1)
Tuning exp_ijk_fusedloop
	Best time: 0.000170088 s -- speedup: 23.428995578759235 -- workgroupsize: (128, 1)
Tuning exp_ijk_split
	Best time: 0.000201039 s -- speedup: 19.821979814861795 -- workgroupsize: (128, 1)
Tuning exp_ijk_nosym
	Best time: 0.000204278 s -- speedup: 19.507685604910954 -- workgroupsize: (32, 7)
Tuning exp_ijk_nosym_split
	Best time: 0.000230439 s -- speedup: 17.2930406745386 -- workgroupsize: (32, 9)
Tuning exp_ijk_nosym_fusedloop
	Best time: 0.000203899 s -- speedup: 19.543945777075905 -- workgroupsize: (32, 7)
Tuning exp_ijk_nosym_fusedloop_inter
	Best time: 0.000198338 s -- speedup: 20.091918845606994 -- workgroupsize: (32, 7)
  ```

## Profiling 2025-12-03

### Original code

```
Device-side activity: GPU was busy for 2.44 s (73.73% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name                                                                                           ⋯
├──────────┼────────────┼───────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────
│   50.76% │     1.68 s │    50 │  33.56 ms ± 4.26   ( 32.02 ‥ 52.18)  │ gpu__flux_differencing_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndi ⋯
│    9.43% │  311.69 ms │    50 │   6.23 ms ± 0.17   (  6.16 ‥ 7.09)   │ gpu_surface_integral_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndice ⋯
│    5.49% │  181.45 ms │    50 │   3.63 ms ± 0.21   (  3.55 ‥ 4.56)   │ gpu_prolong2interfaces_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndi ⋯
│    2.64% │   87.28 ms │    50 │   1.75 ms ± 0.16   (  1.69 ‥ 2.44)   │ gpu_interface_flux_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices< ⋯
│    2.27% │    75.0 ms │    50 │    1.5 ms ± 0.25   (  1.42 ‥ 2.59)   │ gpu__apply_jacobian_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices ⋯
│    1.69% │   55.94 ms │    90 │ 621.54 µs ± 3.19   (610.59 ‥ 626.56) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices ⋯
│    0.63% │   20.72 ms │    50 │ 414.34 µs ± 6.0    (410.56 ‥ 439.41) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices ⋯
│    0.34% │   11.08 ms │    10 │   1.11 ms ± 0.0    (   1.1 ‥ 1.11)   │ gpu_max_scaled_speed_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndice ⋯
│    0.28% │    9.27 ms │    50 │ 185.35 µs ± 6.37   ( 178.1 ‥ 213.38) │ gpu_fill_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<O ⋯
│    0.14% │    4.52 ms │    11 │ 410.84 µs ± 3.0    ( 409.6 ‥ 419.86) │ [copy device to device memory]                                                                 ⋯
│    0.07% │    2.38 ms │     9 │ 263.96 µs ± 1.1    (261.78 ‥ 265.36) │ partial_mapreduce_grid(INFINITE_OR_GIANT, _, Bool, CartesianIndices<1, Tuple<OneTo<Int64>>>, C ⋯
│    0.00% │   49.11 µs │    10 │   4.91 µs ± 0.17   (  4.77 ‥ 5.25)   │ partial_mapreduce_grid(identity, max, Float64, CartesianIndices<1, Tuple<OneTo<Int64>>>, Carte ⋯
│    0.00% │   40.29 µs │    10 │   4.03 µs ± 0.18   (  3.81 ‥ 4.29)   │ partial_mapreduce_grid(identity, max, Float64, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<I ⋯
│    0.00% │   36.72 µs │    19 │   1.93 µs ± 0.18   (  1.67 ‥ 2.15)   │ [copy device to pageable memory]                                                               ⋯
│    0.00% │   36.72 µs │     9 │   4.08 µs ± 0.14   (  3.81 ‥ 4.29)   │ partial_mapreduce_grid(identity, _, Bool, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64> ⋯
└──────────┴────────────┴───────┴──────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                                                                         1 column omitted
```

### exp_ijk

#### workgroup = (32, 1)

```
Device-side activity: GPU was busy for 1.96 s (69.43% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name                                                                                           ⋯
├──────────┼────────────┼───────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────
│   42.39% │     1.19 s │    50 │   23.9 ms ± 0.92   ( 23.61 ‥ 27.01)  │ gpu__exp_ijk_flux_differencing_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, Carte ⋯
│   11.07% │  312.01 ms │    50 │   6.24 ms ± 0.23   (  6.15 ‥ 7.17)   │ gpu_surface_integral_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndice ⋯
│    6.50% │  183.12 ms │    50 │   3.66 ms ± 0.28   (  3.55 ‥ 4.67)   │ gpu_prolong2interfaces_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndi ⋯
│    3.12% │   88.01 ms │    50 │   1.76 ms ± 0.2    (  1.69 ‥ 2.45)   │ gpu_interface_flux_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices< ⋯
│    2.67% │   75.18 ms │    50 │    1.5 ms ± 0.28   (  1.42 ‥ 2.59)   │ gpu__apply_jacobian_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices ⋯
│    1.99% │   55.96 ms │    90 │ 621.83 µs ± 1.61   (617.27 ‥ 625.13) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices ⋯
│    0.74% │   20.73 ms │    50 │ 414.67 µs ± 7.16   (410.56 ‥ 440.36) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices ⋯
│    0.39% │   11.08 ms │    10 │   1.11 ms ± 0.0    (   1.1 ‥ 1.12)   │ gpu_max_scaled_speed_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndice ⋯
│    0.33% │    9.31 ms │    50 │ 186.15 µs ± 8.49   (178.34 ‥ 214.1)  │ gpu_fill_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<O ⋯
│    0.16% │    4.52 ms │    11 │ 410.86 µs ± 3.0    ( 409.6 ‥ 419.86) │ [copy device to device memory]                                                                 ⋯
│    0.08% │    2.38 ms │     9 │ 264.22 µs ± 0.87   (263.21 ‥ 266.08) │ partial_mapreduce_grid(INFINITE_OR_GIANT, _, Bool, CartesianIndices<1, Tuple<OneTo<Int64>>>, C ⋯
│    0.00% │   49.59 µs │    10 │   4.96 µs ± 0.15   (  4.77 ‥ 5.25)   │ partial_mapreduce_grid(identity, max, Float64, CartesianIndices<1, Tuple<OneTo<Int64>>>, Carte ⋯
│    0.00% │   39.82 µs │    10 │   3.98 µs ± 0.16   (  3.81 ‥ 4.29)   │ partial_mapreduce_grid(identity, max, Float64, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<I ⋯
│    0.00% │   37.91 µs │     9 │   4.21 µs ± 0.29   (  3.81 ‥ 4.77)   │ partial_mapreduce_grid(identity, _, Bool, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64> ⋯
│    0.00% │   37.67 µs │    19 │   1.98 µs ± 0.24   (  1.67 ‥ 2.38)   │ [copy device to pageable memory]                                                               ⋯
└──────────┴────────────┴───────┴──────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                                                                         1 column omitted
```

#### workgroup = (32, 8)

```
Device-side activity: GPU was busy for 1.61 s (64.96% of the trace)
┌──────────┬────────────┬───────┬──────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────
│ Time (%) │ Total time │ Calls │ Time distribution                    │ Name                                                                                           ⋯
├──────────┼────────────┼───────┼──────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────
│   34.01% │  843.89 ms │    50 │  16.88 ms ± 1.4    ( 16.39 ‥ 21.74)  │ gpu__exp_ijk_flux_differencing_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, Carte ⋯
│   12.62% │  313.22 ms │    50 │   6.26 ms ± 0.25   (  6.15 ‥ 7.13)   │ gpu_surface_integral_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndice ⋯
│    7.44% │  184.54 ms │    50 │   3.69 ms ± 0.28   (  3.55 ‥ 4.7)    │ gpu_prolong2interfaces_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndi ⋯
│    3.57% │   88.54 ms │    50 │   1.77 ms ± 0.2    (   1.7 ‥ 2.45)   │ gpu_interface_flux_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices< ⋯
│    3.11% │   77.25 ms │    50 │   1.54 ms ± 0.31   (  1.42 ‥ 2.59)   │ gpu__apply_jacobian_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices ⋯
│    2.25% │   55.92 ms │    90 │ 621.32 µs ± 1.73   ( 617.5 ‥ 626.33) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices ⋯
│    0.84% │   20.76 ms │    50 │ 415.15 µs ± 7.92   (411.03 ‥ 439.64) │ gpu_broadcast_kernel_linear(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices ⋯
│    0.45% │   11.11 ms │    10 │   1.11 ms ± 0.0    (  1.11 ‥ 1.12)   │ gpu_max_scaled_speed_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndice ⋯
│    0.38% │    9.35 ms │    50 │ 187.04 µs ± 9.07   ( 178.1 ‥ 213.62) │ gpu_fill_kernel_(CompilerMetadata<DynamicSize, DynamicCheck, void, CartesianIndices<1, Tuple<O ⋯
│    0.18% │    4.52 ms │    11 │ 411.32 µs ± 3.1    (410.08 ‥ 420.57) │ [copy device to device memory]                                                                 ⋯
│    0.10% │    2.39 ms │     9 │ 265.36 µs ± 1.19   (263.45 ‥ 266.55) │ partial_mapreduce_grid(INFINITE_OR_GIANT, _, Bool, CartesianIndices<1, Tuple<OneTo<Int64>>>, C ⋯
│    0.00% │   50.78 µs │    10 │   5.08 µs ± 0.16   (  4.77 ‥ 5.25)   │ partial_mapreduce_grid(identity, max, Float64, CartesianIndices<1, Tuple<OneTo<Int64>>>, Carte ⋯
│    0.00% │   40.29 µs │    10 │   4.03 µs ± 0.21   (  3.81 ‥ 4.29)   │ partial_mapreduce_grid(identity, max, Float64, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<I ⋯
│    0.00% │   36.24 µs │    19 │   1.91 µs ± 0.18   (  1.67 ‥ 2.38)   │ [copy device to pageable memory]                                                               ⋯
│    0.00% │   36.24 µs │     9 │   4.03 µs ± 0.22   (  3.81 ‥ 4.53)   │ partial_mapreduce_grid(identity, _, Bool, CartesianIndices<2, Tuple<OneTo<Int64>, OneTo<Int64> ⋯
└──────────┴────────────┴───────┴──────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────
                                                                                                                                                         1 column omitted
```

## Timing 2025-12-03

### Original code

```
─────────────────────────────────────────────────────────────────────────────────
           Trixi.jl                     Time                    Allocations
                               ───────────────────────   ────────────────────────
       Tot / % measured:            3.92s /  77.6%           49.0MiB /  33.2%

Section                ncalls     time    %tot     avg     alloc    %tot      avg
─────────────────────────────────────────────────────────────────────────────────
rhs!                       51    2.40s   79.1%  47.2ms    561KiB    3.4%  11.0KiB
  ~rhs!~                   51    2.40s   78.9%  47.0ms   23.9KiB    0.1%     480B
  interface flux           51   1.19ms    0.0%  23.3μs    114KiB    0.7%  2.23KiB
  reset ∂u/∂t              51   1.16ms    0.0%  22.8μs   64.2KiB    0.4%  1.26KiB
  prolong2interfaces       51   1.15ms    0.0%  22.5μs   97.6KiB    0.6%  1.91KiB
  volume integral          51   1.06ms    0.0%  20.7μs   91.3KiB    0.5%  1.79KiB
  surface integral         51   1.03ms    0.0%  20.2μs   96.0KiB    0.6%  1.88KiB
  Jacobian                 51    999μs    0.0%  19.6μs   74.5KiB    0.4%  1.46KiB
  boundary flux            51   21.6μs    0.0%   424ns     0.00B    0.0%    0.00B
  prolong2boundaries       51   14.9μs    0.0%   291ns     0.00B    0.0%    0.00B
  prolong2mortars          51   14.2μs    0.0%   278ns     0.00B    0.0%    0.00B
  source terms             51   4.23μs    0.0%  82.9ns     0.00B    0.0%    0.00B
  mortar flux              51   4.14μs    0.0%  81.2ns     0.00B    0.0%    0.00B
calculate dt               11    636ms   20.9%  57.8ms   15.7MiB   96.5%  1.43MiB
performance data            3    564μs    0.0%   188μs   16.4KiB    0.1%  5.46KiB
─────────────────────────────────────────────────────────────────────────────────
```

### exp_ijk

#### workgroup = (32, 1)

```
─────────────────────────────────────────────────────────────────────────────────
           Trixi.jl                     Time                    Allocations
                               ───────────────────────   ────────────────────────
       Tot / % measured:            3.48s /  74.9%           49.0MiB /  33.4%

Section                ncalls     time    %tot     avg     alloc    %tot      avg
─────────────────────────────────────────────────────────────────────────────────
rhs!                       51    1.97s   75.5%  38.6ms    571KiB    3.4%  11.2KiB
  ~rhs!~                   51    1.96s   75.2%  38.5ms   23.9KiB    0.1%     480B
  prolong2interfaces       51   1.20ms    0.0%  23.6μs   97.6KiB    0.6%  1.91KiB
  reset ∂u/∂t              51   1.20ms    0.0%  23.6μs   64.2KiB    0.4%  1.26KiB
  interface flux           51   1.16ms    0.0%  22.7μs    114KiB    0.7%  2.23KiB
  surface integral         51   1.07ms    0.0%  21.0μs   96.0KiB    0.6%  1.88KiB
  volume integral          51   1.07ms    0.0%  20.9μs    101KiB    0.6%  1.98KiB
  Jacobian                 51    945μs    0.0%  18.5μs   74.5KiB    0.4%  1.46KiB
  prolong2mortars          51   24.6μs    0.0%   482ns     0.00B    0.0%    0.00B
  prolong2boundaries       51   16.3μs    0.0%   320ns     0.00B    0.0%    0.00B
  boundary flux            51   10.9μs    0.0%   214ns     0.00B    0.0%    0.00B
  mortar flux              51   8.76μs    0.0%   172ns     0.00B    0.0%    0.00B
  source terms             51   5.21μs    0.0%   102ns     0.00B    0.0%    0.00B
calculate dt               11    640ms   24.5%  58.2ms   15.8MiB   96.5%  1.43MiB
performance data            3    578μs    0.0%   193μs   16.4KiB    0.1%  5.46KiB
─────────────────────────────────────────────────────────────────────────────────
```

#### workgroup = (32, 8)

```
─────────────────────────────────────────────────────────────────────────────────
           Trixi.jl                     Time                    Allocations
                               ───────────────────────   ────────────────────────
       Tot / % measured:            3.15s /  72.1%           49.1MiB /  33.7%

Section                ncalls     time    %tot     avg     alloc    %tot      avg
─────────────────────────────────────────────────────────────────────────────────
rhs!                       51    1.58s   69.6%  31.1ms    571KiB    3.4%  11.2KiB
  ~rhs!~                   51    1.58s   69.3%  30.9ms   23.9KiB    0.1%     480B
  reset ∂u/∂t              51   1.21ms    0.1%  23.7μs   64.2KiB    0.4%  1.26KiB
  prolong2interfaces       51   1.20ms    0.1%  23.6μs   97.6KiB    0.6%  1.91KiB
  interface flux           51   1.19ms    0.1%  23.4μs    114KiB    0.7%  2.23KiB
  surface integral         51   1.04ms    0.0%  20.4μs   96.0KiB    0.6%  1.88KiB
  volume integral          51   1.01ms    0.0%  19.8μs    101KiB    0.6%  1.98KiB
  Jacobian                 51    987μs    0.0%  19.4μs   74.5KiB    0.4%  1.46KiB
  prolong2boundaries       51   13.8μs    0.0%   270ns     0.00B    0.0%    0.00B
  boundary flux            51   10.4μs    0.0%   204ns     0.00B    0.0%    0.00B
  prolong2mortars          51   9.66μs    0.0%   189ns     0.00B    0.0%    0.00B
  source terms             51   8.24μs    0.0%   162ns     0.00B    0.0%    0.00B
  mortar flux              51   4.21μs    0.0%  82.6ns     0.00B    0.0%    0.00B
calculate dt               11    691ms   30.4%  62.8ms   16.0MiB   96.5%  1.45MiB
performance data            3    593μs    0.0%   198μs   16.4KiB    0.1%  5.46KiB
─────────────────────────────────────────────────────────────────────────────────
```