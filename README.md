# Parallel computing using BLAS Libraries with OpenMP 

parallel version of math_lib_benchmark

## Benchmark on multithreading performance

'AVG' is the averaged actual time takes for the whole binary to run for a single replication. Other four categories are the averaged time elapsed for a single replication for each thread.

![Imgur](https://imgur.com/SDHJCzD.jpeg)

## Potential bugs

When using OpenBLAS, `USE_OpenMP=1` should be set. However, using `cmake` to compile the library will cause problems. The `LAPACKE_dgesv` function will give very inaccurate result when dimension is high (n > 33). The project is test on different platforms with similar errors. I haven't figure out what cause this particular problem. Build the `libopenblas.a` with `Makefile` and use USE_OPENMP=1 can avoid such error.
