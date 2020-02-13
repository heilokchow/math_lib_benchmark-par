MKL_ROOT	= $(HOME)/intel/mkl
OPENBLAS_ROOT	= $(HOME)/OpenBLAS
LAPACK_ROOT 	= $(HOME)/.local/lapack
EIGEN_DIR	= $(HOME)/Eigen
CCFLAG = -O2 -mfma

INCLUDE_DIR = -I$(EIGEN_DIR)

ifeq ($(MKL), 1)
INCLUDE_DIR += -I$(MKL_ROOT)/include
else ifeq ($(OPENBLAS), 1)
INCLUDE_DIR += -I$(OPENBLAS_ROOT)/include 
else
INCLUDE_DIR += 
endif

ifeq ($(MKL), 1)
LIB_DIR = -DUSE_MKL -DUSE_NESTED -Wl,--start-group $(MKL_ROOT)/lib/intel64/libmkl_intel_lp64.a $(MKL_ROOT)/lib/intel64/libmkl_gnu_thread.a $(MKL_ROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -fopenmp -lm -ldl
else ifeq ($(OPENBLAS), 1)
LIB_DIR = -L$(OPENBLAS_ROOT)/lib -l:libopenblas.a -fopenmp
else
LIB_DIR = -L$(LAPACK_ROOT) -l:liblapacke.a -l:liblapack.a -l:libcblas.a -l:libblas.a -lgfortran -fopenmp
endif

all : bench.cpp
	g++ bench.cpp $(CCFLAG) $(INCLUDE_DIR) $(LIB_DIR)

clean :
	-rm a.out

.PHONY : clean
