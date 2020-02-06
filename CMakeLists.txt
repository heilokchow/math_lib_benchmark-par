cmake_minimum_required(VERSION 3.8)

project("math_bench")

option(MODEL "Use Eigen or not" ON)

configure_file(benchConfig.h.in benchConfig.h)

find_package(Eigen3 REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(benchmark "bench.cpp")

target_compile_options(benchmark
	PUBLIC -O2
	PUBLIC -mfma
)

target_link_libraries(benchmark
	PUBLIC Eigen3::Eigen
	PUBLIC OpenBLAS::OpenBLAS
	PUBLIC OpenMP::OpenMP_CXX
)
