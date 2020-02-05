cmake_minimum_required(VERSION 3.8)

project("math_bench")

option(MODEL "Use Eigen or not" ON)

configure_file(benchConfig.h.in benchConfig.h)

set(BLA_VENDOR OpenBLAS)

find_package(Eigen3 REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenMP REQUIRED)

if(BLAS_FOUND)
	message("${LAPACK_LIBRARIES}")
endif(BLAS_FOUND)

add_executable(benchmark "bench.cpp")
target_compile_options(benchmark
	PUBLIC -O2
	PUBLIC -mfma
)
target_link_libraries(benchmark
	PUBLIC Eigen3::Eigen
	PUBLIC "${BLAS_LIBRARIES}"
	PUBLIC OpenMP::OpenMP_CXX
)
