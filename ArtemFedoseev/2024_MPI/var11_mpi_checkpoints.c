#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <malloc.h>

const int N = ((1 << 10) + 2);
const double maxeps = 0.1e-7;
const int itmax = 100;

////
const char ckpt_name[] = "checkpoint.dat";
const int ckpt_period = 5;
int NET_SIZE, RANK;
double eps;
int height, full_height;
int global_i_offset = 0;
int border_size = 2;
int cell_up = MPI_PROC_NULL;
int cell_down = MPI_PROC_NULL;

////
void relax(double (* A)[N], double (* B)[N]);

void resid(double (* A)[N], double (* B)[N]);

void allocate(double (** A)[N], double (** B)[N]);

void init(double (* A)[N]);

void verify(double (* A)[N]);

void create_checkpoint(double (* A)[N], int it) {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, ckpt_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_size(fh, sizeof(int) + N * N * sizeof(double));
    if (RANK == 0) {
        MPI_File_write(fh, &it, 1, MPI_INT, MPI_STATUS_IGNORE);
    }
    MPI_Offset offset = sizeof(int) + global_i_offset * N * sizeof(double);
    MPI_File_write_at_all(fh, offset, &A[border_size][0], N * height, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

void read_checkpoint(double (* A)[N], int* pit) {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, ckpt_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_File_read_all(fh, pit, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_Offset offset = sizeof(int) + global_i_offset * N * sizeof(double);
    MPI_File_read_at_all(fh, offset, &A[border_size][0], N * height, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

int main(int an, char** as) {
    MPI_Init(&an, &as);
    MPI_Comm_size(MPI_COMM_WORLD, &NET_SIZE);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    if (NET_SIZE % 2 && NET_SIZE != 1) {
        printf("Proc. number should be multiple of 2!\nGot: %d\n", NET_SIZE);
        MPI_Finalize();
        return 1;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double time = MPI_Wtime();
    double (* A)[N] = NULL;
    double (* B)[N] = NULL;
    allocate(&A, &B);
    init(A);
    for (int it = 1; it <= itmax; it++) {
        eps = 0.;
        if (it % ckpt_period == 0) {
            create_checkpoint(A, it);
            read_checkpoint(A, &it);
        }
        relax(A, B);
        resid(A, B);
        if (RANK == 0) {
            printf("it=%4i   eps=%f\n", it, eps);
        }
        if (eps < maxeps) break;
    }
    verify(A);
    double end_time = MPI_Wtime(); // в verify групповая операция и неявный барьер
    if (RANK == 0) {
        printf("%lf\n", end_time - time);
    }
    free(A);
    free(B);
    MPI_Finalize();
    return 0;
}

void allocate(double (** A)[N], double (** B)[N]) {
    int base_height = N / NET_SIZE;
    int left_over = N % NET_SIZE;
    int row_num = RANK;
    height = base_height + (row_num < left_over); //
    full_height = height + 2 * border_size;
    // determine neighbours
    cell_up = (RANK > 0) ? RANK - 1 : MPI_PROC_NULL;
    cell_down = (RANK < NET_SIZE - 1) ? RANK + 1 : MPI_PROC_NULL;
    // get local to global consts
    global_i_offset = base_height * row_num + (int) fmin(row_num, left_over);
    if (*A) {
        free(*A);
    }
    if (*B) {
        free(*B);
    }
    *A = malloc((height + 2 * border_size) * N * sizeof(double));
    *B = malloc(height * N * sizeof(double));
}

void init(double (* A)[N]) {
    // init our data
    for (int i = 0; i < full_height; i++) {
        for (int j = 0; j < N; j++) {
            if ((global_i_offset + i - border_size) == 0 ||
                (global_i_offset + i - border_size) == N - 1 ||
                j == 0 ||
                j == N - 1 ||
                i < border_size ||
                i > full_height - border_size) {
                A[i][j] = 0.;
            } else {
                A[i][j] = (1. + global_i_offset + i - border_size + j);
            }
        }
    }

}

void relax(double (* A)[N], double (* B)[N]) {
    MPI_Request out[2], in[2];
    // setup sends and receives
    MPI_Isend(&A[border_size][0], border_size * N, MPI_DOUBLE, cell_up, 0, MPI_COMM_WORLD, &out[0]);
    MPI_Irecv(&A[0][0], border_size * N, MPI_DOUBLE, cell_up, 0, MPI_COMM_WORLD, &in[0]);
    MPI_Isend(&(A[full_height - 2 * border_size][0]), border_size * N, MPI_DOUBLE, cell_down, 0, MPI_COMM_WORLD,
              &out[1]);
    MPI_Irecv(&A[full_height - border_size][0], border_size * N, MPI_DOUBLE, cell_down, 0, MPI_COMM_WORLD, &in[1]);
    // do non-border stuff
    for (int i = 2 * border_size; i < full_height - (2 * border_size); i++) {
        for (int j = 2; j < N - 2; j++) {
            B[i - border_size][j] = (A[i + 2][j]
                                     + A[i + 1][j]
                                     + A[i][j - 2] + A[i][j - 1] + A[i][j + 2] + A[i][j + 1]
                                     + A[i - 1][j]
                                     + A[i - 2][j]
                                    ) / 8.;
        }
    }
    // wait for border syncs
    MPI_Waitall(2, in, MPI_STATUSES_IGNORE);
    // do border stuff
    if (cell_up != MPI_PROC_NULL) {
        for (int i = border_size; i < 2 * border_size; i++) {
            for (int j = 2; j < N - 2; j++) {
                B[i - border_size][j] = (A[i + 2][j]
                                         + A[i + 1][j]
                                         + A[i][j - 2] + A[i][j - 1] + A[i][j + 2] + A[i][j + 1]
                                         + A[i - 1][j]
                                         + A[i - 2][j]
                                        ) / 8.;
            }
        }
    }
    if (cell_down != MPI_PROC_NULL) {
        for (int i = full_height - (2 * border_size); i < full_height - border_size; i++) {
            for (int j = 2; j < N - 2; j++) {
                B[i - border_size][j] = (A[i + 2][j]
                                         + A[i + 1][j]
                                         + A[i][j - 2] + A[i][j - 1] + A[i][j + 2] + A[i][j + 1]
                                         + A[i - 1][j]
                                         + A[i - 2][j]
                                        ) / 8.;
            }
        }
    }
    MPI_Waitall(2, out, MPI_STATUSES_IGNORE);
}

void resid(double (* A)[N], double (* B)[N]) {
    for (int i = (cell_up == MPI_PROC_NULL) + border_size;
         i < full_height - ((cell_down == MPI_PROC_NULL) + border_size); i++) {
        for (int j = 1; j < N - 1; j++) {
            double e = fabs(A[i][j] - B[i - border_size][j]);
            A[i][j] = B[i - border_size][j];
            eps = fmax(eps, e);
        }
    }
    MPI_Allreduce(MPI_IN_PLACE, &eps, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
}

void verify(double (* A)[N]) {
    double s = 0.;
    for (int i = border_size; i < full_height - border_size; i++) {
        for (int j = 0; j < N; j++) {
            s += A[i][j] * (global_i_offset + i - border_size + 1) * (j + 1) / (N * N);
        }
    }
    double total_s = 0;
//    printf("  lS = %f\n", s);
    MPI_Reduce(&s, &total_s, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (RANK == 0) {
        printf("  S = %f\n", total_s);
    }
}
