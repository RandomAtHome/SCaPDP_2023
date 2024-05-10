/* Include benchmark-specific header. */
#include "heat-3d.h"
#include <mpi.h>
#include <mpi-ext.h>
#include <setjmp.h>
#include <signal.h>

double bench_t_start = 0, bench_t_end = 0;

const char filename[] = "save.dat";
MPI_Comm COMM = MPI_COMM_NULL;

int RANK, NET_SIZE;
int halo_height;
int i_offset = 0;
int cell_up = MPI_PROC_NULL;
int cell_down = MPI_PROC_NULL;
int need_restore = 0;
jmp_buf restart;
int FIRST_INIT = 1;
int IS_VICTIM = 0;

char** gargv;

/**
 * if comm == MPI_PROC_NULL - gets "correct" parent world
 * else spawns missing amount of processes
 */
int MPIX_Comm_replace(MPI_Comm comm, MPI_Comm* newcomm) {
    // https://github.com/ICLDisco/ulfm-testing/blob/master/tutorial/10.respawn.c
    MPI_Comm icomm, /* the intercomm between the spawnees and the old (shrinked) world */
    scomm, /* the local comm for each sides of icomm */
    mcomm; /* the intracomm, merged from icomm */
    int rc, flag, rflag, nc, ns, nd;

    redo:
    if (comm == MPI_COMM_NULL) { /* am I a new process? */
        /* I am a new spawnee, waiting for my new rank assignment
         * it will be sent by rank 0 in the old world */
        MPI_Comm_get_parent(&icomm);
        scomm = MPI_COMM_WORLD;
    } else {
        /* I am a survivor: Spawn the appropriate number
         * of replacement processes (we check that this operation worked
         * before we procees further) */
        /* First: remove dead processes */

        MPIX_Comm_shrink(comm, &scomm);
        MPI_Comm_size(scomm, &ns);
        MPI_Comm_size(comm, &nc);
        nd = nc - ns; /* number of deads */
        if (0 == nd) {
            /* Nobody was dead to start with. We are done here */
            MPI_Comm_free(&scomm);
            *newcomm = comm;
            return MPI_SUCCESS;
        }
        /* We handle failures during this function ourselves... */
        MPI_Comm_set_errhandler(scomm, MPI_ERRORS_RETURN);

        rc = MPI_Comm_spawn(gargv[0], &gargv[1], nd, MPI_INFO_NULL,
                            0, scomm, &icomm, MPI_ERRCODES_IGNORE);
        flag = (MPI_SUCCESS == rc);
        MPIX_Comm_agree(scomm, &flag);
        if (!flag) {
            if (MPI_SUCCESS == rc) {
                MPIX_Comm_revoke(icomm);
                MPI_Comm_free(&icomm);
            }
            MPI_Comm_free(&scomm);
            // if we failed - retry
            goto redo;
        }
    }

    rc = MPI_Intercomm_merge(icomm, 1, &mcomm);
    rflag = flag = (MPI_SUCCESS == rc);
    MPIX_Comm_agree(scomm, &flag);
    if (MPI_COMM_WORLD != scomm) MPI_Comm_free(&scomm);
    MPIX_Comm_agree(icomm, &rflag);
    MPI_Comm_free(&icomm);
    if (!(flag && rflag)) {
        if (MPI_SUCCESS == rc) {
            MPI_Comm_free(&mcomm);
        }
        // if we failed - retry
        goto redo;
    }

    /* restore the error handler */
    if (MPI_COMM_NULL != comm) {
        MPI_Errhandler errh;
        MPI_Comm_get_errhandler(comm, &errh);
        MPI_Comm_set_errhandler(mcomm, errh);
    }
    *newcomm = mcomm;

    MPI_Comm_size(mcomm, &NET_SIZE);
    MPI_Comm_rank(mcomm, &RANK);
    return MPI_SUCCESS;
}

/**
 * Универсальный обработчик ошибок, который ужимает коммуникатор, исключая упавшие процессы.
 */
static void verbose_errhandler(MPI_Comm* pcomm, int* perr, ...) {
    // взято с https://github.com/ICLDisco/ulfm-testing/blob/master/tutorial/02.err_handler.c
    MPI_Comm comm = *pcomm;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int rank, size, nf, len, eclass;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    MPI_Group group_c, group_f;
    int* ranks_gc, * ranks_gf;

    MPI_Comm newcomm = NULL;
    MPI_Error_class(err, &eclass);
    MPI_Error_string(err, errstr, &len);
    switch (eclass) {
        case MPIX_ERR_PROC_FAILED:
            MPIX_Comm_get_failed(comm, &group_f);
            MPIX_Comm_ack_failed(comm, NET_SIZE, &nf);
            printf("Rank %d / %d: Notified of error %s. %d found dead: { ",
                   rank, size, errstr, nf);

            ranks_gf = (int*) malloc(nf * sizeof(int));
            ranks_gc = (int*) malloc(nf * sizeof(int));
            MPI_Comm_group(comm, &group_c);
            for (int i = 0; i < nf; i++) {
                ranks_gf[i] = i;
            }
            MPI_Group_translate_ranks(group_f, nf, ranks_gf,
                                      group_c, ranks_gc);
            for (int i = 0; i < nf; i++) {
                printf("%d ", ranks_gc[i]);
            }
            printf("}\n");
            free(ranks_gf);
            free(ranks_gc);
            MPIX_Comm_revoke(comm);
            printf("I (rank %d of %d) reinited comm\n", rank, size);
            MPIX_Comm_replace(comm, &newcomm);
            break;
        case MPIX_ERR_REVOKED:
            printf("I (rank %d of %d) was told to reinit comm\n", rank, size);
            MPIX_Comm_replace(comm, &newcomm);
            break;
        default:
            printf("Unknown error at %d / %d\n%s\n", rank, size, errstr);
            MPI_Abort(comm, err);
    }
    // setup global variables
    COMM = newcomm;
    MPI_Comm_rank(COMM, &RANK);
    MPI_Comm_size(COMM, &NET_SIZE);
    need_restore = 1;
    longjmp(restart, 1);
}


static
void init_array(int n,
                double A[halo_height][n][n]) {
    for (int i = 1; i < halo_height - 1; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                A[i][j][k] = (double) (i_offset + (i - 1) + j + (n - k)) * 10 / (n);
}

/*
 * ULFM не очень работает с MPI IO, на текущий момент
 * https://docs.open-mpi.org/en/v5.0.x/features/ulfm.html#known-limitations-in-ulfm
 *
 * работает, если отключить shared file pointers, согласно
 * https://github.com/open-mpi/ompi/issues/12197
 * https://github.com/open-mpi/ompi/issues/9656
 * т.е. добавить
 * --mca sharedfp ^sm
 * */
void create_checkpoint(double A[halo_height][N][N], int t) {
    if (RANK == 0) {
        printf("Made a checkpoint!\n");
    }
    MPI_File fh;
    MPI_Barrier(COMM);
    MPI_File_open(COMM, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if (fh == MPI_FILE_NULL) {
        if (RANK == 0) {
            printf("MPI IO is broken, leaving\n");
        }
        MPI_Abort(COMM, 1);
        exit(1);
    }
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
    if (IS_VICTIM) {
        printf("Fare thee well!\n");
        raise(SIGKILL);
    }
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
        if (t % 100 == 0) {
            if (RANK == 0) {
                printf("I am at %d\n", t);
            }
            create_checkpoint(A, t);
            validate(n, A);
        }
        do_computation(n, A, B);
        do_computation(n, B, A);
    }
}


int main(int argc, char** argv) {
    MPI_Comm parent; /* a parent comm for the work, w/o the spares */
    gargv = argv;
    MPI_Init(&argc, &argv);
    MPI_Comm_dup(MPI_COMM_WORLD, &COMM);
    /* Am I a spare ? */
    MPI_Comm_get_parent(&parent);
    if (MPI_COMM_NULL == parent) {
        /* First run: Let's create an initial parent, a copy of MPI_COMM_WORLD */
        MPI_Comm_dup(MPI_COMM_WORLD, &COMM);
        MPI_Barrier(COMM);
        bench_t_start = MPI_Wtime();
    } else {
        /* I am a spare, lets get the repaired parent */
        printf("Spare was born!\n");
        MPIX_Comm_replace(MPI_COMM_NULL, &COMM);
        need_restore = 2;
    }

    MPI_Comm_size(COMM, &NET_SIZE);
    MPI_Comm_rank(COMM, &RANK);
    if (need_restore == 0 && RANK == NET_SIZE - 1 && RANK) {
        IS_VICTIM = 1;
    }
    MPI_Errhandler errhandler;
    MPI_Comm_create_errhandler(verbose_errhandler, &errhandler);
    MPI_Comm_set_errhandler(COMM, errhandler);
    if (NET_SIZE % 2 && NET_SIZE != 1) {
        MPI_Finalize();
        return 1;
    }

    setjmp(restart);
    int n = N;

    int base_height = N / NET_SIZE;
    int left_over = N % NET_SIZE;
    halo_height = base_height + (RANK < left_over) + 2;
    cell_up = (RANK > 0) ? RANK - 1 : MPI_PROC_NULL;
    cell_down = (RANK < NET_SIZE - 1) ? RANK + 1 : MPI_PROC_NULL;

    i_offset = base_height * RANK + (int) fmin(RANK, left_over);

    double (* A)[halo_height][n][n];
    double (* B)[halo_height][n][n];
    if (FIRST_INIT) {
        A = (double (*)[halo_height][n][n]) malloc(
                (halo_height) * (n) * (n) * sizeof(double));
        B = (double (*)[halo_height][n][n]) malloc(
                (halo_height) * (n) * (n) * sizeof(double));
    }

    init_array(n, *A);
    init_array(n, *B);
    FIRST_INIT = 0;
    int curr_step = 1;
    if (need_restore) {
//        printf("[%d] I am at restore\n", RANK);
//        printf("[%d] I have offset: %d, halo_height: %d, N: %d, \n", RANK, i_offset, halo_height, N);
        recover(*A, &curr_step);
        need_restore = 0;
    }
    kernel_heat_3d(curr_step, n, *A, *B);

    MPI_Barrier(COMM);
    bench_t_end = MPI_Wtime();

    validate(n, *A);
    if (RANK == 0) {
        printf("%0.6lf\n", bench_t_end - bench_t_start);
    }
    free((void*) A);
    free((void*) B);
    MPI_Finalize();
    return 0;
}
