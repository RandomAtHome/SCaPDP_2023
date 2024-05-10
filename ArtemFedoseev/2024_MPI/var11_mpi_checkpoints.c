#include <math.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <malloc.h>
#include <signal.h>
#include <setjmp.h>

const int N = ((1 << 10) + 2);
const double maxeps = 0.1e-7;
const int itmax = 100;
const int border_size = 2;

//
const char ckpt_name[] = "checkpoint.dat";
const int ckpt_period = 5;
int have_failed = 0;
jmp_buf recv_point;
//
MPI_Comm GLOBAL_COMM;
int NET_SIZE, RANK;
int cell_up = MPI_PROC_NULL;
int cell_down = MPI_PROC_NULL;
//
double eps;
int height, full_height;
int global_i_offset = 0;

static void verbose_errhandler(MPI_Comm* pcomm, int* perr, ...) {
    // взято с https://github.com/ICLDisco/ulfm-testing/blob/master/tutorial/02.err_handler.c
    MPI_Comm comm = *pcomm;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int rank, size, nf, len, eclass;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Group group_c, group_f;
    int *ranks_gc, *ranks_gf;

    MPI_Comm newcomm = NULL;
    MPI_Error_class(err, &eclass);
    MPI_Error_string(err, errstr, &len);
    switch (eclass) {
        case MPIX_ERR_PROC_FAILED:
            /* We use a combination of 'ack/get_acked' to obtain the list of
             * failed processes (as seen by the local rank).
             */
            MPIX_Comm_failure_ack(comm);
            MPIX_Comm_failure_get_acked(comm, &group_f);
            MPI_Group_size(group_f, &nf);
            printf("Rank %d / %d: Notified of error %s. %d found dead: { ",
                   rank, size, errstr, nf);

            /* We use 'translate_ranks' to obtain the ranks of failed procs
             * in the input communicator 'comm'.
             */
            ranks_gf = (int*)malloc(nf * sizeof(int));
            ranks_gc = (int*)malloc(nf * sizeof(int));
            MPI_Comm_group(comm, &group_c);
            for(int i = 0; i < nf; i++) {
                ranks_gf[i] = i;
            }
            MPI_Group_translate_ranks(group_f, nf, ranks_gf,
                                      group_c, ranks_gc);
            for(int i = 0; i < nf; i++) {
                printf("%d ", ranks_gc[i]);
            }
            printf("}\n");
            free(ranks_gf); free(ranks_gc);
            MPIX_Comm_revoke(comm);
            // we don't reraise ERR_REVOKED on ourselves, do we?
        case MPIX_ERR_REVOKED:
            // shrink communicator
            printf("I (rank %d of %d) shrinked\n", rank, size);
            MPIX_Comm_shrink(comm, &newcomm);
            break;
        default:
            printf("I (rank %d of %d) don't understand what happened.\n%s\n", rank, size, errstr);
            MPI_Abort(comm, err);
    }
    // setup global variables
    GLOBAL_COMM = newcomm;
    MPI_Comm_rank(GLOBAL_COMM, &RANK);
    MPI_Comm_size(GLOBAL_COMM, &NET_SIZE);
    have_failed = 1;
    longjmp(recv_point, 1);
}
////
void relax(double (* A)[N], double (* B)[N]);

void resid(double (* A)[N], double (* B)[N]);

void allocate(double (** A)[N], double (** B)[N]);

void init(double (* A)[N]);

void verify(double (* A)[N]);

void create_checkpoint(double (* A)[N], int it) {
    MPI_File fh;
    MPI_File_open(GLOBAL_COMM, ckpt_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    MPI_File_set_size(fh, sizeof(int) + N * N * sizeof(double));
    if (RANK == 0) {
        MPI_File_write(fh, &it, 1, MPI_INT, MPI_STATUS_IGNORE);
    }
    MPI_Offset offset = sizeof(int) + global_i_offset * N * sizeof(double);
    MPI_File_write_at_all(fh, offset, &A[border_size][0], N * height, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

/**
 *
 * */
void recover(double (* A)[N], int* pit) {
    MPI_File fh;
    MPI_File_open(GLOBAL_COMM, ckpt_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if (fh == MPI_FILE_NULL) {
        if (RANK == 0) {
            printf("No checkpoint!\n");
        }
        *pit = 1;
        init(A);
        return;
    }
    MPI_File_read_all(fh, pit, 1, MPI_INT, MPI_STATUS_IGNORE);
    MPI_Offset offset = sizeof(int) + global_i_offset * N * sizeof(double);
    MPI_File_read_at_all(fh, offset, &A[border_size][0], N * height, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

int main(int an, char** as) {
    MPI_Init(&an, &as);
    MPI_Errhandler errhandler;
    MPI_Comm_size(MPI_COMM_WORLD, &NET_SIZE);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    if (NET_SIZE % 2 && NET_SIZE != 1) {
        printf("Proc. number should be multiple of 2!\nGot: %d\n", NET_SIZE);
        MPI_Finalize();
        return 1;
    }
    MPI_Comm_create_errhandler(verbose_errhandler, &errhandler);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, errhandler);
    MPI_Comm_dup(MPI_COMM_WORLD, &GLOBAL_COMM);
    //
    MPI_Barrier(GLOBAL_COMM);
    double time = MPI_Wtime();
    double (* A)[N] = NULL;
    double (* B)[N] = NULL;
    allocate(&A, &B);
    init(A);
    setjmp(recv_point);
    for (int it = 1; it <= itmax; it++) {
        if (have_failed) {
            allocate(&A, &B);
            recover(A, &it);
            have_failed = 0;
        }
        eps = 0.;
        if (it % ckpt_period == 0) {
            create_checkpoint(A, it);
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
    MPI_Isend(&A[border_size][0], border_size * N, MPI_DOUBLE, cell_up, 0, GLOBAL_COMM, &out[0]);
    MPI_Irecv(&A[0][0], border_size * N, MPI_DOUBLE, cell_up, 0, GLOBAL_COMM, &in[0]);
    if (RANK == 2) {
        printf("Off I go!\n");
        raise(SIGKILL);
    }
    MPI_Isend(&(A[full_height - 2 * border_size][0]), border_size * N, MPI_DOUBLE, cell_down, 0, GLOBAL_COMM,
              &out[1]);
    MPI_Irecv(&A[full_height - border_size][0], border_size * N, MPI_DOUBLE, cell_down, 0, GLOBAL_COMM, &in[1]);
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
    MPI_Allreduce(MPI_IN_PLACE, &eps, 1, MPI_DOUBLE, MPI_MAX, GLOBAL_COMM);
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
    MPI_Reduce(&s, &total_s, 1, MPI_DOUBLE, MPI_SUM, 0, GLOBAL_COMM);
    if (RANK == 0) {
        printf("  S = %f\n", total_s);
    }
}
