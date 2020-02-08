#include <iostream>
#include <chrono>
#include <stdio.h>
#include <random>
#include <string>
#include <fstream>
#include <memory>
#include <Eigen/Dense>
#include <cblas.h>
#include <lapacke.h>
#include <omp.h>
#include "benchConfig.h"

#define EIGEN_NO_DEBUG
#define EIGEN_DONT_PARALLELIZE

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cout << "No. of input: " << argc << std::endl;
        puts("./a.out <dim> <nrep>");
        exit(0);
    }
  
    int n = std::stoi(argv[1]);
    int nrep = std::stoi(argv[2]);

    double t_eigen = 0;
    double t_openblas = 0; 
    double start_eigen, end_eigen;
    double start_openblas, end_openblas;

    // Perform Martix Multiplication
    
    #pragma omp parallel for schedule(dynamic) firstprivate(start_eigen, end_eigen)
    for (int i = 0; i < nrep; i++) {

        double* x = new double[n*n];
        double* y = new double[n*n];
        double* z = new double[n*n];
        int p = 0;
        
        int id = omp_get_thread_num();
        printf("ID: %d, #Eigen Matrix Multiplication\n", id);

        std::random_device device;
        std::mt19937 generator(device());
        std::normal_distribution<double> normal(0.0, 1.0);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                p = i * n + j;
                y[p] = normal(generator);
                z[p] = normal(generator);
            }
        }

        Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n, n);
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            X(&y[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            Y(&y[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));

        start_eigen = omp_get_wtime();
        C.noalias() = X * Y;
        end_eigen = omp_get_wtime();

        #pragma omp critical
        t_eigen += end_eigen - start_eigen;

        delete[] x;
        delete[] y;
        delete[] z;
    }

    #pragma omp parallel for schedule(dynamic) firstprivate(start_openblas, end_openblas)
    for (int i = 0; i < nrep; i++) {
        double* x = new double[n*n];
        double* y = new double[n*n];
        double* z = new double[n*n];
        int p = 0;

        std::random_device device;
        std::mt19937 generator(device());
        std::normal_distribution<double> normal(0.0, 1.0);

        int id = omp_get_thread_num();
        printf("ID: %d, #OpenBLAS Matrix Multiplication\n", id);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                p = i * n + j;
                y[p] = normal(generator);
                z[p] = normal(generator);
            }
        }

        start_openblas = omp_get_wtime();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, y, n, z, n, 0.0, x, n);
        end_openblas = omp_get_wtime();

        #pragma omp critical
        t_openblas += end_openblas - start_openblas;

        delete[] x;
        delete[] y;
        delete[] z;
    }

    printf("Eigen DGEMM average time: %f\n", t_eigen/nrep);
    printf("OpenBLAS DGEMM average time: %f\n", t_openblas/nrep);

    // Perform Matrix Inversion
    
    t_eigen = 0;
    t_openblas = 0;

    #pragma omp parallel for schedule(dynamic) firstprivate(start_eigen, end_eigen)
    for (int i = 0; i < nrep; i++) {

        double* yy = new double[n*n];
        double* y = new double[n*n];
        double* b = new double[n];
        double* x = new double[n];

        int p = 0;
        
        int id = omp_get_thread_num();
        printf("ID: %d, #Eigen Linear Solver\n", id);

        std::random_device device;
        std::mt19937 generator(device());
        std::normal_distribution<double> normal(0.0, 1.0);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                p = i * n + j;
                y[p] = normal(generator);
                yy[p] = 0.0;
            }
            b[i] = normal(generator);
            x[i] = 0.0;
        }

        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            Y(&y[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            YY(&yy[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<1>> B(&b[0], n);
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<1>> X(&x[0], n);
        
        YY.noalias() = Y * Y.transpose();
        X.noalias() = YY * B;
        
        start_eigen = omp_get_wtime();
        B.noalias() = YY.llt().solve(X);
        end_eigen = omp_get_wtime();

        #pragma omp critical
        t_eigen += end_eigen - start_eigen;

        delete[] yy;
        delete[] y;
        delete[] b;
        delete[] x;
    }

    #pragma omp parallel for schedule(dynamic) firstprivate(start_openblas, end_openblas)
    for (int i = 0; i < nrep; i++) {

        double* yy = new double[n*n];
        double* y = new double[n*n];
        double* b = new double[n];
        double* x = new double[n];
        int* ipiv = new int[n];

        int p = 0;
        
        int id = omp_get_thread_num();
        printf("ID: %d, #OpenBLAS Linear Solver\n", id);

        std::random_device device;
        std::mt19937 generator(device());
        std::normal_distribution<double> normal(0.0, 1.0);

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                p = i * n + j;
                y[p] = normal(generator);
                yy[p] = 0.0;
            }
            b[i] = normal(generator);
            x[i] = 0.0;
        }

        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            Y(&y[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));
        Eigen::Map<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, 1>>
            YY(&yy[0], n, n, Eigen::Stride<Eigen::Dynamic, 1>(n, 1));
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<1>> B(&b[0], n);
        Eigen::Map<Eigen::VectorXd, 0, Eigen::InnerStride<1>> X(&x[0], n);
        
        YY.noalias() = Y * Y.transpose();
        X.noalias() = YY * B;
        
        start_openblas = omp_get_wtime();
        LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, 1, yy, n, ipiv, x, 1);
        end_openblas = omp_get_wtime();

        #pragma omp critical
        t_openblas += end_openblas - start_openblas;

        delete[] yy;
        delete[] y;
        delete[] b;
        delete[] x;
        delete[] ipiv;
    }

    printf("Eigen DGESV average time: %f\n", t_eigen/nrep);
    printf("OpenBLAS DEGSV average time: %f\n", t_openblas/nrep);

    return 0;
}
