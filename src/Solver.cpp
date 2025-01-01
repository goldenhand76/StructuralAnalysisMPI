#include "SolverMPI.h"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <mpi.h>

std::vector<double> Solver::solveSystem(int rank, int size, const std::vector<std::vector<double>>& K_global, const std::vector<double>& F) {
    int n = K_global.size();
    std::vector<std::vector<double>> A = K_global;  // Copy of K_global for manipulation
    std::vector<double> b = F;                     // Copy of F (right-hand side vector)
    std::vector<double> u(n, 0.0);                // Solution vector

    // Perform Gaussian elimination
    for (int i = 0; i < n; ++i) {
        // Step 1: Pivoting (done by rank 0 and broadcasted)
        int maxRow = i;
        if (rank == 0) {
            double maxEl = std::abs(A[i][i]);
            for (int k = i + 1; k < n; ++k) {
                if (std::abs(A[k][i]) > maxEl) {
                    maxEl = std::abs(A[k][i]);
                    maxRow = k;
                }
            }
            // Swap rows in the matrix and the vector
            if (maxRow != i) {
                std::swap(A[i], A[maxRow]);
                std::swap(b[i], b[maxRow]);
            }
        }

        // Broadcast the pivot row to all processes
        MPI_Bcast(&A[i][0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&b[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Step 2: Make rows below the pivot zero in the current column
        for (int k = i + 1 + rank; k < n; k += size) {
            double c = -A[k][i] / A[i][i];
            for (int j = i; j < n; ++j) {
                if (i == j) {
                    A[k][j] = 0;
                }
                else {
                    A[k][j] += c * A[i][j];
                }
            }
            b[k] += c * b[i];
        }

        // Gather the updated rows back to the root process
        for (int k = i + 1; k < n; ++k) {
            MPI_Allreduce(MPI_IN_PLACE, &A[k][0], n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(MPI_IN_PLACE, &b[k], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
    }

    // Step 3: Back-substitution (start from the last row and move upwards)
    for (int i = n - 1; i >= 0; --i) {
        if (rank == 0) {
            u[i] = b[i] / A[i][i];
        }

        // Broadcast the solved value to all processes
        MPI_Bcast(&u[i], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Update the right-hand side vector b for rows above the current row
        for (int k = i - 1 - rank; k >= 0; k -= size) {
            b[k] -= A[k][i] * u[i];
        }

        // Gather the updated rows back to the root process
        for (int k = i - 1; k >= 0; --k) {
            MPI_Allreduce(MPI_IN_PLACE, &b[k], 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        }
    }

    return u;  // Return the solution vector (displacement/flow field)
}