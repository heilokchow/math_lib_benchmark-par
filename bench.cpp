#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE

#include <iostream>
#include <chrono>
#include <stdio.h>
#include <random>
#include <string>
#include <fstream>
#include <memory>
#include <Eigen/Dense>
#include <omp.h>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

void check(const int&);

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cout << "No. of input: " << argc << std::endl;
        puts("./a.out <dim> <nrep> [threads]");
        exit(0);
    }
  
    int n = std::stoi(argv[1]);
    int nrep = std::stoi(argv[2]);
    int num_threads;

    if (argc == 4) {
        num_threads = std::stoi(argv[3]);
    }

    double sum = 0.0;
    double t_eigen = 0.0, t_openblas = 0.0;
    double start_eigen, end_eigen, start_openblas, end_openblas;

    std::ofstream out;
    out.open("result.txt", std::ios_base::app);

    omp_set_num_threads(num_threads);

#ifdef USE_MKL
#ifdef USE_NESTED
    omp_set_nested(1);
    printf("Whether nested is allowed: %d\n", omp_get_nested());
    int k = omp_get_num_procs() / num_threads;
#endif
#endif

    // Perform Martix Multiplication
    
    #pragma omp parallel for schedule(dynamic) shared(t_eigen, t_openblas) \
            firstprivate(sum, start_eigen, end_eigen, start_openblas, end_openblas)
    for (int i = 0; i < nrep; i++) {

#ifdef USE_MKL
#ifdef USE_NESTED
        mkl_set_num_threads_local(k);
#endif
#endif

        double* x1 = new double[n*n];
        double* x2 = new double[n*n];
        double* y = new double[n*n];
        double* z = new double[n*n];
        
        std::random_device device;
        std::mt19937 generator(device());
        std::normal_distribution<double> normal(0.0, 1.0);

        for (int i = 0; i < n*n; i++)
            y[i] = normal(generator);

        for (int i = 0; i < n*n; i++)
            z[i] = normal(generator);

        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            X1(&x1[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            Y(&y[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            Z(&z[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));

        start_eigen = omp_get_wtime();
        X1.noalias() = Y * Z;
        end_eigen = omp_get_wtime();

        start_openblas = omp_get_wtime();
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, y, n, z, n, 0.0, x2, n);
        end_openblas = omp_get_wtime();

        for (int i = 0; i < n*n; i++)
            sum += abs(x1[i] - x2[i]);

        int id = omp_get_thread_num();
        printf("ID: %d, Error: %f #Matrix Multiplication\n", id, sum);

        #pragma omp critical
        {
            t_eigen += end_eigen - start_eigen;
            t_openblas += end_openblas - start_openblas;
        }

        delete[] x1;
        delete[] x2;
        delete[] y;
        delete[] z;
    }

    out << "DGEMM, Eigen, " << n << ", " << t_eigen / nrep << "\n";
    out << "DGEMM, OBLAS, " << n << ", " << t_openblas / nrep << "\n";

    // Perform Matrix Inversion
    
    sum = 0.0;
    t_eigen = 0.0;
    t_openblas = 0.0;

    #pragma omp parallel for schedule(dynamic) shared(t_eigen, t_openblas) \
            firstprivate(sum, start_eigen, end_eigen, start_openblas, end_openblas)
    for (int i = 0; i < nrep; i++) {

#ifdef USE_MKL
#ifdef USE_NESTED
        mkl_set_num_threads_local(k);
#endif
#endif

        double* yy = new double[n*n];
        double* y = new double[n*n];
        double* b = new double[n];
        double* x = new double[n];
        int* ipiv = new int[n];

        std::random_device device;
        std::mt19937 generator(device());
        std::normal_distribution<double> normal(0.0, 1.0);

        for (int i = 0; i < n*n; i++)
            y[i] = normal(generator);

        for (int i = 0; i < n; i++)
            b[i] = normal(generator);

        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            Y(&y[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            YY(&yy[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<1>> B(&b[0], n);
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<1>> X(&x[0], n);
        
        YY.noalias() = Y * Y.transpose();
        Eigen::MatrixXd YY_copy(YY);
        X.noalias() = YY * B;

        start_eigen = omp_get_wtime();
        B.noalias() = YY_copy.llt().solve(X);
        end_eigen = omp_get_wtime();

        start_openblas = omp_get_wtime();
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, yy, n, ipiv, x, 1);
        end_openblas = omp_get_wtime();

        for (int i = 0; i < n; i++)
            sum += abs(x[i] - b[i]);

        int id = omp_get_thread_num();
        printf("ID: %d, Error: %f #Linear Solver\n", id, sum);

        #pragma omp critical
        {
            t_eigen += end_eigen - start_eigen;
            t_openblas += end_openblas - start_openblas;
        }

        delete[] yy;
        delete[] y;
        delete[] b;
        delete[] x;
        delete[] ipiv;
    }

    out << "DGESV, Eigen, " << n << ", " << t_eigen / nrep << "\n";
    out << "DGESV, OBLAS, " << n << ", " << t_openblas / nrep << "\n";
    out.close();

    return 0;
}
