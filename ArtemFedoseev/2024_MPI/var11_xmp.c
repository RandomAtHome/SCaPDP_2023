#include <math.h>
#include <stdio.h>
#include "omp.h" // чисто для библиотеки времени подключили, вообще говоря не нужна библиотека. хотя есть interoperability
/*
 * Ссылка на документацию, благо она маленькая.
 * https://xcalablemp.org/handbook/overview.html
 */

#define  Max(a, b) ((a)>(b)?(a):(b))
#define  N ((1 << 13) + 2)

double A[N][N], B[N][N];
#pragma xmp nodes p[*][2] // описали структуру наших "узлов", пользуясь тем что у нас по условию всегда чётное число процессов
#pragma xmp template t[N][N] // описали шаблон распределения данных, благо у нас он один всегда...
#pragma xmp distribute t[block][block] onto p // ...и описали, как это раскидывается по узлам.
#pragma xmp align A[i][j] with t[i][j] // увязали реальные данные с шаблоном
#pragma xmp shadow A[2][2] // !!! killer-feature XMP! описали, сколько у нас "граничных" данных, которые нужны с соседних узлоа
#pragma xmp align B[i][j] with t[i][j] // увязали реальные данные с шаблоном

double eps;
void relax();

void resid();

void init();

void verify();

int main(int an, char** as) {
    double maxeps = 0.1e-7;
    int itmax = 100;
    double time = omp_get_wtime();
    init();
    for (int it = 1; it <= itmax; it++) {
        eps = 0.;
        relax();
        resid();
#pragma xmp task on p[0][0] // чтобы не дублировать печать итераций, сказали какой узел печатает
        {
            printf("it=%4i   eps=%f\n", it, eps);
        }
        if (eps < maxeps) {
            break;
        }
    }
    verify();
#pragma xmp task on p[0][0]// чтобы не дублировать печать итераций, сказали какой узел печатает
    {
        printf("%lf\n", omp_get_wtime() - time);
    }
    return 0;
}

void init() {
#pragma xmp loop (i,j) on t[i][j]// увязали как индексы цикла соотносятся с шаблоном (который соотносится с распределением по узлам)
    for (int i = 0; i <= N - 1; i++) {
        for (int j = 0; j <= N - 1; j++) {
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                A[i][j] = 0.;
            } else {
                A[i][j] = (1. + i + j);
            }
        }
    }
}

void relax() {
#pragma xmp reflect (A) // прежде чем вычислять цикл, синхронизировали гало (shadow директива наверху) между узлами
#pragma xmp loop (i,j) on t[i][j]
    for (int i = 2; i <= N - 3; i++) {
        for (int j = 2; j <= N - 3; j++) {
            B[i][j] = (A[i - 2][j] + A[i - 1][j] + A[i + 2][j] + A[i + 1][j] + A[i][j - 2] + A[i][j - 1] + A[i][j + 2] +
                       A[i][j + 1]) / 8.;
        }
    }
}

void resid() {
#pragma xmp loop (i,j) on t[i][j] reduction(max:eps)
    for (int i = 1; i <= N - 2; i++) {
        for (int j = 1; j <= N - 2; j++) {
            double e = fabs(A[i][j] - B[i][j]);
            A[i][j] = B[i][j];
            eps = Max(eps, e);
        }
    }
}

void verify() {
    double s = 0.;
#pragma xmp loop (i, j) on t[i][j] reduction(+:s)
    for (int i = 0; i <= N - 1; i++) {
        for (int j = 0; j <= N - 1; j++) {
            s = s + A[i][j] * (i + 1) * (j + 1) / (N * N);
        }
    }
#pragma xmp task on p[0][0]
    {
        printf("  S = %f\n", s);
    }
}
