#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    const int N = 4096;

    int base        = N / P;
    int extra       = N % P;
    int local_n     = base + (rank < extra ? 1 : 0);
    int local_start = rank * base + std::min(rank, extra);

    double t_start = MPI_Wtime();

    // Initialize local chunks
    std::vector<double> u(local_n), v(local_n);
    for (int i = 0; i < local_n; ++i) {
        double k = local_start + i + 1.0;
        u[i] = k;
        v[i] = 1.0 / k;
    }

    // Local dot product
    double local_dot = 0.0;
    for (int i = 0; i < local_n; ++i)
        local_dot += u[i] * v[i];

    // Binary tree reduction
    for (int step = 1; step < P; step *= 2) {
        if (rank % (2 * step) == 0) {
            int partner = rank + step;
            if (partner < P) {
                double recv_val;
                MPI_Recv(&recv_val, 1, MPI_DOUBLE, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                local_dot += recv_val;
            }
        } else {
            MPI_Send(&local_dot, 1, MPI_DOUBLE, rank - step, 0, MPI_COMM_WORLD);
            break;
        }
    }

    double t_elapsed = MPI_Wtime() - t_start;

    // Collect worst-case time
    double t_max;
    MPI_Reduce(&t_elapsed, &t_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "N=" << N << "  P=" << P << "\n";
        std::cout << "Dot product = " << local_dot << "\n";
        std::cout << "Expected    = " << N << "\n";
        std::cout << "Max time    = " << t_max << " s\n";
    }

    MPI_Finalize();
    return 0;
}
