#include <mpi.h>
#include <Kokkos_Core.hpp>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {
        constexpr int P = 4;
        constexpr int Q = 2;
        constexpr int M = 16;

        int world_rank, world_size;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        if (world_size != P * Q) {
            if (world_rank == 0)
                std::cerr << "[P6A] Need exactly " << (P * Q) << " MPI ranks (got "
                          << world_size << ").\n";
            Kokkos::finalize();
            MPI_Finalize();
            return 1;
        }

        const int row_id = world_rank / Q;
        const int col_id = world_rank % Q;

        MPI_Comm row_comm, col_comm;
        MPI_Comm_split(MPI_COMM_WORLD, row_id, col_id, &row_comm);
        MPI_Comm_split(MPI_COMM_WORLD, col_id, row_id, &col_comm);

        const int rows_per_proc = M / P;
        const int scatter_vals_per_rank = rows_per_proc / Q;

        std::vector<double> x_global(M, 0.0);
        if (world_rank == 0)
            for (int J = 0; J < M; ++J) x_global[J] = static_cast<double>(J);

        std::vector<double> x_linear_chunk(rows_per_proc, 0.0);
        if (col_id == 0) {
            MPI_Scatter(world_rank == 0 ? x_global.data() : nullptr,
                        rows_per_proc, MPI_DOUBLE,
                        x_linear_chunk.data(), rows_per_proc, MPI_DOUBLE,
                        0, col_comm);
        }
        MPI_Bcast(x_linear_chunk.data(), rows_per_proc, MPI_DOUBLE, 0, row_comm);

        Kokkos::View<double*> x("x", rows_per_proc);
        {
            auto xh = Kokkos::create_mirror_view(x);
            for (int i = 0; i < rows_per_proc; ++i) xh(i) = x_linear_chunk[i];
            Kokkos::deep_copy(x, xh);
        }

        Kokkos::View<double*> y_linear("y_linear", rows_per_proc);
        Kokkos::parallel_for(
            "scale_x_to_y", rows_per_proc,
            KOKKOS_LAMBDA(int i) { y_linear(i) = 2.0 * x(i); });
        Kokkos::fence();

        auto y_host = Kokkos::create_mirror_view(y_linear);
        Kokkos::deep_copy(y_host, y_linear);

        std::vector<double> y_scatter_block(scatter_vals_per_rank);
        for (int j = 0; j < scatter_vals_per_rank; ++j) {
            const int linear_idx = col_id + j * Q;
            y_scatter_block[j] = y_host(linear_idx);
        }

        std::vector<double> gathered_row(rows_per_proc, 0.0);
        MPI_Gather(y_scatter_block.data(), scatter_vals_per_rank, MPI_DOUBLE,
                   gathered_row.data(), scatter_vals_per_rank, MPI_DOUBLE,
                   0, row_comm);

        if (col_id == 0) {
            std::vector<double> row_major_chunk(rows_per_proc);
            for (int k = 0; k < rows_per_proc; ++k)
                row_major_chunk[k] =
                    gathered_row[(k % Q) * scatter_vals_per_rank + k / Q];

            std::vector<double> y_global(M, 0.0);
            MPI_Gather(row_major_chunk.data(), rows_per_proc, MPI_DOUBLE,
                       world_rank == 0 ? y_global.data() : nullptr,
                       rows_per_proc, MPI_DOUBLE, 0, col_comm);

            if (world_rank == 0) {
                bool ok = true;
                for (int J = 0; J < M && ok; ++J) {
                    if (y_global[J] != 2.0 * static_cast<double>(J)) ok = false;
                }
                std::cout << (ok ? "[P6A] PASSED: y[J] == 2*x[J] for all J\n"
                                 : "[P6A] FAILED verification\n");
                for (int J = 0; J < M; ++J)
                    std::cout << "[P6A]   y[" << J << "] = " << y_global[J] << "\n";
            }
        }

        MPI_Comm_free(&row_comm);
        MPI_Comm_free(&col_comm);
    }
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
