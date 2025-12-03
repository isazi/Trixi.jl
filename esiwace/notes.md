
# Notes on performance

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

### exp_ijk tuned code

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