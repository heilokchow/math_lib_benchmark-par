# Parallel computing using BLAS Libraries with OpenMP 

parallel version of math_lib_benchmark

## Benchmark

![Imgur](https://i.imgur.com/WGSUEG5.png?2)

![Imgur](https://i.imgur.com/PE9cALk.png?1)

`llt()` function solves positve definite matrix is considered here which is faster. 

## Potential bugs

When using OpenBLAS, `USE_OPENMP=1` should be set. However, using `cmake` to compile the library will cause problems. The `LAPACKE_dgesv` function will give very inaccurate result when dimension is high (n > 33). The project is test on different platforms with similar errors. I haven't figure out what cause this particular problem. Build the `libopenblas.a` with `Makefile` and use `USE_OPENMP=1` can avoid such error. (Update: cmake with no flag also cause this issue)
