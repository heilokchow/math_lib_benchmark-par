cmake_minimum_required(VERSION 3.8)

project("math_bench")

configure_file(benchConfig.h.in benchConfig.h)

find_package(Eigen3 REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "/home/heilokchow/intel")
message("CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

set(BLA_VENDOR Intel10_64lp)
find_package(LAPACK)

message("LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")

add_executable(benchmark bench.cpp)

target_compile_options(benchmark
	PUBLIC -O2
	PUBLIC -mfma
)

target_include_directories(benchmark
	PUBLIC "${CMAKE_CURRENT_BINARY_DIR}"
)

target_link_libraries(benchmark
	PUBLIC Eigen3::Eigen
	PUBLIC OpenBLAS::OpenBLAS
	PUBLIC OpenMP::OpenMP_CXX
)

enable_testing()

function(do_test target dim nrep thd)
	add_test(NAME "${target}_${thd}" COMMAND "./${target}" ${dim} ${nrep} ${thd})
endfunction(do_test)

# Multithread performance testing
do_test(benchmark 2000 100 1)
do_test(benchmark 2000 100 2)
do_test(benchmark 2000 100 3)
do_test(benchmark 2000 100 4)
do_test(benchmark 2000 100 5)
do_test(benchmark 2000 100 6)
do_test(benchmark 2000 100 7)
do_test(benchmark 2000 100 8)
do_test(benchmark 2000 100 9)
do_test(benchmark 2000 100 10)
do_test(benchmark 2000 100 11)
do_test(benchmark 2000 100 12)
do_test(benchmark 2000 100 13)
do_test(benchmark 2000 100 14)
do_test(benchmark 2000 100 15)
do_test(benchmark 2000 100 16)
