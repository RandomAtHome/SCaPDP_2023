/* Include benchmark-specific header. */
#include <upc_relaxed.h>
#include "heat-3d.h"

double bench_t_start, bench_t_end;
shared [N * N] double A[N][N][N];
shared [N * N] double B[N][N][N];

static
double rtclock() {
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, NULL);
    if (stat != 0)
        printf("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start() {
    bench_t_start = rtclock();
}

void bench_timer_stop() {
    bench_t_end = rtclock();
}

void bench_timer_print() {
    printf("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


static
void init_array(int n) {
    upc_forall(int i = 0; i < n; i++; &(A[i][0][0])) {
        double (*loc_a)[n] = (double (*)[n]) &(A[i][0][0]);
        double (*loc_b)[n] = (double (*)[n]) &(B[i][0][0]);
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                loc_a[j][k] = loc_b[j][k] = (double) (i + j + (n - k)) * 10 / (n);
        }
}

static
void kernel_heat_3d(int tsteps,
                    int n) {
    for (int t = 1; t <= TSTEPS; t++) {
        upc_forall(int i = 1; i < n - 1; i++; &(B[i][0][0])) {
            double (*loc_b)[n] = (double (*)[n]) &(B[i][0][0]);
            double (*loc_a)[n] = (double (*)[n]) &(A[i][0][0]);
            for (int j = 1; j < n - 1; j++) {
                for (int k = 1; k < n - 1; k++) {
                    loc_b[j][k] = 0.125 * (A[i + 1][j][k] - 2.0 * loc_a[j][k] + A[i - 1][j][k])
                                 + 0.125 * (loc_a[j + 1][k] - 2.0 * loc_a[j][k] + loc_a[j - 1][k])
                                 + 0.125 * (loc_a[j][k + 1] - 2.0 * loc_a[j][k] + loc_a[j][k - 1])
                                 + loc_a[j][k];
                }
            }
        }
        upc_barrier;
        upc_forall(int i = 1; i < n - 1; i++; &(A[i][0][0])) {
            double (*loc_b)[n] = (double (*)[n]) &(B[i][0][0]);
            double (*loc_a)[n] = (double (*)[n]) &(A[i][0][0]);
            for (int j = 1; j < n - 1; j++) {
                for (int k = 1; k < n - 1; k++) {
                    loc_a[j][k] = 0.125 * (B[i + 1][j][k] - 2.0 * loc_b[j][k] + B[i - 1][j][k])
                                 + 0.125 * (loc_b[j + 1][k] - 2.0 * loc_b[j][k] + loc_b[j - 1][k])
                                 + 0.125 * (loc_b[j][k + 1] - 2.0 * loc_b[j][k] + loc_b[j][k - 1])
                                 + loc_b[j][k];
                }
            }
        }
        upc_barrier;
    }
}

static
void validate(int n) {
    static shared double hash[THREADS];
    hash[MYTHREAD] = 0;
    upc_forall(int i = 0; i < n; i++; &(A[i][0][0])) {
        double (*loc_a)[n] = (double (*)[n]) &(A[i][0][0]);
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                hash[MYTHREAD] += loc_a[j][k] / (n * n);
            }
        }
    }
    upc_barrier;
    if (MYTHREAD == 0) {
        double local_hash = 0;
        for (int i = 0; i < THREADS; i++) {
            local_hash += hash[i];
        }
        printf("%lf\n", local_hash);
    }
}


int main(int argc, char** argv) {
    const int n = N;
    int tsteps = TSTEPS;

    init_array(n);

    if (MYTHREAD == 0) {
        bench_timer_start();
    }

    kernel_heat_3d(tsteps, n);

    if (MYTHREAD == 0) {
        bench_timer_stop();
        bench_timer_print();
    }
    validate(n);

    return 0;
}
