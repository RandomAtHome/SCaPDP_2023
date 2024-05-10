/* Include benchmark-specific header. */
#include "heat-3d.h"
#include <mpi.h>

double bench_t_start, bench_t_end;

const char filename[] = "save.dat";
MPI_Comm COMM = MPI_COMM_NULL;

int RANK, NET_SIZE;
int halo_height;
int i_offset = 0;
int cell_up = MPI_PROC_NULL;
int cell_down = MPI_PROC_NULL;


static
void init_array(int n,
                double A[halo_height][n][n]) {
    for (int i = 1; i < halo_height - 1; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                A[i][j][k] = (double) (i_offset + (i - 1) + j + (n - k)) * 10 / (n);
}

void create_checkpoint(double A[halo_height][N][N], int t) {
    MPI_File fh;
    MPI_File_open(COMM, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_size(fh, sizeof(int) + N * N * N * sizeof(double));
    if (RANK == 0) {
        MPI_File_write(fh, &t, 1, MPI_INT, MPI_STATUS_IGNORE);
    }
    MPI_Offset offset = sizeof(int) + i_offset * N * N * sizeof(double);
    MPI_File_write_at_all(fh, offset, &A[1][0][0], N * N * (halo_height - 2), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

void recover(double A[halo_height][N][N], int* t) {
    MPI_File fh;
    MPI_File_open(COMM, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (fh == MPI_FILE_NULL) {
        *t = 1;
        init_array(N, A);
        return;
    }
    MPI_File_read_all(fh, t, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_Offset offset = sizeof(int) + i_offset * N * N * sizeof(double);
    MPI_File_read_at_all(fh, offset, &A[1][0][0], N * N * (halo_height - 2), MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}


static
void print_array(int n,
                 double A[halo_height][n][n]) {

    int flag = 0;
    if (cell_up == MPI_PROC_NULL) {
        MPI_Send(&flag, 1, MPI_INT, RANK + 1 == NET_SIZE ? MPI_PROC_NULL : RANK + 1, 0, COMM);
    }
    MPI_Recv(&flag, 1, MPI_INT, cell_up, 0, COMM, MPI_STATUS_IGNORE);
//    printf("I am rank %d\n", RANK);
    for (int i = 1; i < halo_height - 1; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                if ((i_offset + (i - 1) * n * n + j * n + k) % 20 == 0) {
                    fprintf(stderr, "\n");
                }
                fprintf(stderr, "%0.2lf ", A[i][j][k]);
            }
        }
    }
//    printf("I am rank %d and done\n", RANK);
    MPI_Send(&flag, 1, MPI_INT, cell_down, 0, COMM);
}

static
void validate(int n,
              double A[halo_height][n][n]) {

    double hash = 0, total_hash;
    for (int i = 1; i < halo_height - 1; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                hash += A[i][j][k] / (n * n);
            }
        }
    }
    MPI_Reduce(&hash, &total_hash, 1, MPI_DOUBLE, MPI_SUM, 0, COMM);
    if (RANK == 0) {
        printf("Hash is %lf\n", total_hash);
    }
}

static
void do_computation(int n,
                    double src[halo_height][n][n],
                    double tgt[halo_height][n][n]) {
    MPI_Request out[2], in[2];
    MPI_Isend(&src[1][0][0], n * n, MPI_DOUBLE, cell_up, 0, COMM, &out[0]);
    MPI_Irecv(&src[0][0][0], n * n, MPI_DOUBLE, cell_up, 0, COMM, &in[0]);
    MPI_Isend(&src[halo_height - 2][0][0], n * n, MPI_DOUBLE, cell_down, 0, COMM, &out[1]);
    MPI_Irecv(&src[halo_height - 1][0][0], n * n, MPI_DOUBLE, cell_down, 0, COMM, &in[1]);
    int is_upmost = (cell_up == MPI_PROC_NULL ? 1 : 0);
    int is_lowest = (cell_down == MPI_PROC_NULL ? 1 : 0);
    ///
    /// non-border
    for (int i = 1 + 1; i < halo_height - (1 + 1); i++) {
        for (int j = 1; j < n - 1; j++) {
            for (int k = 1; k < n - 1; k++) {
                tgt[i][j][k] = 0.125 * (src[i + 1][j][k] - 2.0 * src[i][j][k] + src[i - 1][j][k])
                               + 0.125 * (src[i][j + 1][k] - 2.0 * src[i][j][k] + src[i][j - 1][k])
                               + 0.125 * (src[i][j][k + 1] - 2.0 * src[i][j][k] + src[i][j][k - 1])
                               + src[i][j][k];
            }
        }
    }
    MPI_Waitall(2, in, MPI_STATUSES_IGNORE);
    if (!is_upmost) {
        int i = 1;
        for (int j = 1; j < n - 1; j++) {
            for (int k = 1; k < n - 1; k++) {
                tgt[i][j][k] = 0.125 * (src[i + 1][j][k] - 2.0 * src[i][j][k] + src[i - 1][j][k])
                               + 0.125 * (src[i][j + 1][k] - 2.0 * src[i][j][k] + src[i][j - 1][k])
                               + 0.125 * (src[i][j][k + 1] - 2.0 * src[i][j][k] + src[i][j][k - 1])
                               + src[i][j][k];
            }
        }
    }
    if (!is_lowest) {
        int i = halo_height - 2;
        for (int j = 1; j < n - 1; j++) {
            for (int k = 1; k < n - 1; k++) {
                tgt[i][j][k] = 0.125 * (src[i + 1][j][k] - 2.0 * src[i][j][k] + src[i - 1][j][k])
                               + 0.125 * (src[i][j + 1][k] - 2.0 * src[i][j][k] + src[i][j - 1][k])
                               + 0.125 * (src[i][j][k + 1] - 2.0 * src[i][j][k] + src[i][j][k - 1])
                               + src[i][j][k];
            }
        }
    }
    MPI_Waitall(2, out, MPI_STATUSES_IGNORE);
    ///
}

static
void kernel_heat_3d(int cur_step,
                    int n,
                    double A[halo_height][n][n],
                    double B[halo_height][n][n]) {

    for (int t = cur_step; t <= TSTEPS; t++) {
        if (t % 10 == 0) {
            create_checkpoint(A, t);
            recover(A, &t);
        }
        do_computation(n, A, B);
        do_computation(n, B, A);
    }
}


int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    if (NET_SIZE % 2 && NET_SIZE != 1) {
        MPI_Finalize();
        return 1;
    }
    MPI_Comm_dup(MPI_COMM_WORLD, &COMM);
    MPI_Comm_size(COMM, &NET_SIZE);
    MPI_Comm_rank(COMM, &RANK);
    MPI_Barrier(COMM);
    bench_t_start = MPI_Wtime();
    int n = N;

    int base_height = N / NET_SIZE;
    int left_over = N % NET_SIZE;
    halo_height = base_height + (RANK < left_over) + 2;
    if (RANK > 0) {
        cell_up = RANK - 1;
    }
    if (RANK < NET_SIZE - 1) {
        cell_down = RANK + 1;
    }

    i_offset = base_height * RANK + (int) fmin(RANK, left_over);

    double (*A)[halo_height][n][n] = (double (*)[halo_height][n][n]) malloc((halo_height) * (n) * (n) * sizeof(double));
    double (*B)[halo_height][n][n] = (double (*)[halo_height][n][n]) malloc((halo_height) * (n) * (n) * sizeof(double));

    init_array(n, *A);
    init_array(n, *B);

    int curr_step = 1;
    kernel_heat_3d(curr_step, n, *A, *B);

    MPI_Barrier(COMM);
    bench_t_end = MPI_Wtime();

    validate(n, *A);
    if (RANK == 0) {
        printf("%0.6lf\n", bench_t_end - bench_t_start);
    }
    free((void *) A);
    free((void *) B);
    MPI_Finalize();
    return 0;
}
