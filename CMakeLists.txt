cmake_minimum_required(VERSION 3.8)

project("math_bench")

option(MODEL "Use Eigen or not" ON)

configure_file(benchConfig.h.in benchConfig.h)

# set(BLA_VENDOR OpenBLAS)

find_package(Eigen3 REQUIRED)
#find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)
# find_package(BLAS)

# message("BLAS Library: ${BLAS_LIBRARIES}")

add_executable(benchmark "bench.cpp")

target_compile_options(benchmark
	PUBLIC -O2
	PUBLIC -mfma
)

target_include_directories(benchmark
	PUBLIC "${CMAKE_CURRENT_BINARY_DIR}"
	PUBLIC "/home/heilokchow/Documents/Course/OpenBLAS/include"
)

target_link_libraries(benchmark
	PUBLIC Eigen3::Eigen
#	PUBLIC OpenBLAS::OpenBLAS
#	PUBLIC "/home/heilokchow/Documents/Course/OpenBLAS/lib/libopenblas.a"
	PUBLIC "/usr/local/lib/libcblas.a"
	PUBLIC "/usr/local/lib/librefblas.a"
	PUBLIC "/usr/local/lib/liblapacke.a"
	PUBLIC "/usr/local/lib/liblapack.a"
	PUBLIC -gfortran
#	PUBLIC -pthread
	PUBLIC OpenMP::OpenMP_CXX
)
