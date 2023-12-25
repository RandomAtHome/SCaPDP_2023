#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define  Max(a, b) ((a)>(b)?(a):(b))
#define  Min(a, b) ((a)<(b)?(a):(b))

#define N   ((1 << 12) + 2)
const double maxeps = 0.1e-7;
const int itmax = 100;
double eps;
double A[N][N], B[N][N];

void relax();

void resid();

void init();

void verify();

int main(int an, char **as) {
    double start_time = omp_get_wtime();
#pragma omp parallel default(none) shared(itmax, eps, maxeps)
    {
#pragma omp master
        {
            init();
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
#pragma omp taskloop default(none) shared(A)
    for (int i = 0; i <= N - 1; i++) {
        for (int j = 0; j <= N - 1; j++) {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) A[i][j] = 0.;
            else A[i][j] = (1. + i + j);
        }
    }
}

void relax() {
#pragma omp taskloop default(none) shared(A, B)
    for (int i = 2; i <= N - 3; i++) {
        for (int j = 2; j <= N - 3; j++) {
            B[i][j] = (A[i - 2][j] + A[i - 1][j] + A[i + 2][j] + A[i + 1][j] + A[i][j - 2] + A[i][j - 1] +
                       A[i][j + 2] +
                       A[i][j + 1]) / 8.;
        }
    }
}

void resid() {
    int TASK_CNT = omp_get_max_threads();
#pragma omp taskgroup
    {
        for (int task_id = 0; task_id < TASK_CNT; task_id++) {
            int step = ceil(((N - 1) - 1) / TASK_CNT);
#pragma omp task default(none) shared(step, TASK_CNT, A, eps, B) firstprivate(task_id)
            {
                int start = 1 + step * task_id;
                int end = Min((start + step), N - 1);
                double l_eps = 0;
                for (int i = start; i < end; i++) {
                    for (int j = 1; j <= N - 2; j++) {
                        double e;
                        e = fabs(A[i][j] - B[i][j]);
                        A[i][j] = B[i][j];
                        l_eps = Max(l_eps, e);
                    }
                }
#pragma omp critical
                {
                    eps = Max(eps, l_eps);
                }
            }
        }
    }
}

void verify() {
    double s;
    s = 0.;
    for (int i = 0; i <= N - 1; i++) {
        for (int j = 0; j <= N - 1; j++) {
            s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
        }
    }
//    printf("  S = %f\n", s);
}
