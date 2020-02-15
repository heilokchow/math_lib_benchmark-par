cmake_minimum_required(VERSION 3.8)

project("math_bench")

find_package(Eigen3 REQUIRED)
find_package(OpenBLAS REQUIRED)
find_package(OpenMP REQUIRED)

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
	add_test(NAME "${target}_${dim}" COMMAND "./${target}" ${dim} ${nrep} ${thd})
endfunction(do_test)

# Multithread performance testing
# set(THREAD 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
# foreach(X IN LISTS THREAD)
# 	do_test(benchmark 2000 100 "${X}")
# endforeach()

# Singlethread performance testing
set(DIMENSION 50 75 100 150 250 400 600 850 1300 2000 3000 4500 6500 10000)
foreach(X IN LISTS DIMENSION)
	do_test(benchmark "${X}" 10 1)
endforeach()

