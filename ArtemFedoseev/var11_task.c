#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define  Max(a, b) ((a)>(b)?(a):(b))

#define N   ((1 << 12) + 2)
const double maxeps = 0.1e-7;
const int itmax = 100;
double eps;
double A[N][N], B[N][N];
double eps_errors[N];

void relax();

void resid();

void init();

void verify();

int main(int an, char **as) {
    double start_time = omp_get_wtime();
    init();
#pragma omp parallel default(none) shared(itmax, eps, maxeps)
    {
#pragma omp master
        {
            for (int it = 1; it <= itmax; it++) {
                eps = 0.;
                relax();
                resid();
//                printf("it=%4i   eps=%f\n", it, eps);
                if (eps < maxeps) break;
            }
        }
    }
    verify();
    printf("%lf\n", omp_get_wtime() - start_time);
    return 0;
}

void init() {
    for (int i = 0; i <= N - 1; i++) {
        eps_errors[i] = 0;
        for (int j = 0; j <= N - 1; j++) {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) A[i][j] = 0.;
            else A[i][j] = (1. + i + j);
        }
    }
}

void relax() {
#pragma omp taskloop default(none) shared(A, B)
    for (int j = 2; j <= N - 3; j++) {
        for (int i = 2; i <= N - 3; i++) {
            B[i][j] = (A[i - 2][j] + A[i - 1][j] + A[i + 2][j] + A[i + 1][j] + A[i][j - 2] + A[i][j - 1] +
                       A[i][j + 2] +
                       A[i][j + 1]) / 8.;
        }
    }
}

void resid() {
#pragma omp taskgroup
    {
        for (int i = 1; i <= N - 2; i++) {
#pragma omp task default(none) shared(eps, A, B, eps_errors) firstprivate(i)
            {
                double l_eps = 0.;
                for (int j = 1; j <= N - 2; j++) {
                    double e;
                    e = fabs(A[i][j] - B[i][j]);
                    A[i][j] = B[i][j];
                    l_eps = Max(l_eps, e);
                }
                eps_errors[i] = l_eps;
            }
        }
    }
    for (int i = 1; i <= N - 2; i++) {
        eps = Max(eps, eps_errors[i]);
    }
}

void verify() {
    double s;
    s = 0.;
    for (int j = 0; j <= N - 1; j++) {
        for (int i = 0; i <= N - 1; i++) {
            s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
        }
    }
    printf("  S = %f\n", s);
}
