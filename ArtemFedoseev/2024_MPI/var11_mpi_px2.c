#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <malloc.h>

const int N = ((1 << 10) + 2);
const double maxeps = 0.1e-7;
const int itmax = 100;

////
int NET_SIZE, RANK;
double eps;
double (*A)[N / 2];
double (*B)[N / 2];
double (*side_buffer_out)[2];
double (*side_buffer_in)[2];
double (*up_buffer_in)[2];
double (*down_buffer_in)[2];
int height, width = N / 2;
int global_i_offset = 0, global_j_offset = 0;
int cell_up = MPI_PROC_NULL;
int cell_down = MPI_PROC_NULL;
int cell_side = MPI_PROC_NULL;

////
void relax();

void resid();

void init();

void verify();

int main(int an, char **as) {
    MPI_Init(&an, &as);
    MPI_Comm_size(MPI_COMM_WORLD, &NET_SIZE);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    if (NET_SIZE % 2 && NET_SIZE != 1) {
        printf("Proc. number should be multiple of 2!\nGot: %d\n", NET_SIZE);
        MPI_Finalize();
        return 1;
    }
    double time = MPI_Wtime();
    init();
    for (int it = 1; it <= itmax; it++) {
        eps = 0.;
        relax();
        resid();
        if (RANK == 0) {
            printf("it=%4i   eps=%f\n", it, eps);
        }
        if (eps < maxeps) break;
    }
    verify();
    if (RANK == 0) {
        printf("%lf\n", MPI_Wtime() - time);
    }
    MPI_Finalize();
    return 0;
}

void init() {
    int base_height;
    int left_over;
    int row_num = RANK / 2;
    int col_num = RANK % 2;
    if (NET_SIZE == 1) {
        base_height = N;
        left_over = 0;
        width = N;
        height = N;
    } else {
        int vert_blocks = (NET_SIZE / 2);
        base_height = N / vert_blocks;
        left_over = N % vert_blocks;
        height = base_height + (row_num < left_over); //
    }
    // determine neighbours
    if (RANK >= 2) {
        cell_up = RANK - 2;
    }
    if (RANK < NET_SIZE - 2) {
        cell_down = RANK + 2;
    }
    if (NET_SIZE != 1) {
        cell_side = col_num ? RANK - 1 : RANK + 1;
    }
    // allocate border buffers
    side_buffer_out = (cell_side != MPI_PROC_NULL) ? malloc(2 * height * sizeof(double)) : NULL;
    side_buffer_in = (cell_side != MPI_PROC_NULL) ? malloc(2 * height * sizeof(double)) : NULL;
    up_buffer_in = (cell_up != MPI_PROC_NULL) ? malloc(2 * width * sizeof(double)) : NULL;
    down_buffer_in = (cell_down != MPI_PROC_NULL) ? malloc(2 * width * sizeof(double)) : NULL;
    // get local to global consts
    global_i_offset = base_height * row_num + (int) fmin(row_num, left_over);
    global_j_offset = width * col_num;
    // init our data
    for (int i = 0; i < height; i++) {
        for (int j = 0; j <= width; j++) {
            if ((global_i_offset + i) == 0 ||
                (global_i_offset + i) == N - 1 ||
                (global_j_offset + j) == 0 ||
                (global_j_offset + j) == N - 1) {
                A[i][j] = 0.;
            } else {
                A[i][j] = (1. + global_j_offset + i + global_j_offset + j);
            }
        }
    }
}

void relax() {
    MPI_Request exchanges[6];
    MPI_Status statuses[6];
    for (int i = 0; i < height; i++) {
        side_buffer_out[i][0] = A[i][width - 2];
        side_buffer_out[i][1] = A[i][width - 1];
    }
    MPI_Isend(A, 2 * width, MPI_DOUBLE, cell_up, 0, MPI_COMM_WORLD, &exchanges[0]);
    MPI_Irecv(up_buffer_in, 2 * width, MPI_DOUBLE, cell_up, 0, MPI_COMM_WORLD, &exchanges[1]);
    MPI_Isend(&(A[height - 2][0]), 2 * width, MPI_DOUBLE, cell_down, 0, MPI_COMM_WORLD, &exchanges[2]);
    MPI_Irecv(down_buffer_in, 2 * width, MPI_DOUBLE, cell_down, 0, MPI_COMM_WORLD, &exchanges[3]);
    MPI_Isend(side_buffer_out, 2 * height, MPI_DOUBLE, cell_side, 0, MPI_COMM_WORLD, &exchanges[4]);
    MPI_Irecv(side_buffer_in, 2 * height, MPI_DOUBLE, cell_side, 0, MPI_COMM_WORLD, &exchanges[5]);
    // setup sends and receives
    // do non-border stuff
    for (int i = 2; i < height - 2; i++) {
        for (int j = 2; j < width - 2; j++) {
            B[i][j] = (A[i + 2][j]
                       + A[i + 1][j]
                       + A[i][j - 2] + A[i][j - 1] + A[i][j + 2] + A[i][j + 1]
                       + A[i - 1][j]
                       + A[i - 2][j]
                      ) / 8.;
        }
    }
    if (cell_up != MPI_PROC_NULL) {

    }
    // wait for border syncs
    // do border stuff
    MPI_Waitall(6, exchanges, statuses);
    MPI_Barrier(MPI_COMM_WORLD); // вообще говоря необязательно нужно, потом упремся в Allreduce
}

void resid() {
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            double e = fabs(A[i][j] - B[i][j]);
            A[i][j] = B[i][j];
            eps = fmax(eps, e);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void verify() {
    double s = 0.;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            s += A[i][j] * (global_i_offset + i + 1) * (global_j_offset + j + 1) / (N * N);
        }
    }
    double total_s = 0;
    MPI_Reduce(&s, &total_s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (RANK == 0) {
        printf("  S = %f\n", total_s);
    }
}
