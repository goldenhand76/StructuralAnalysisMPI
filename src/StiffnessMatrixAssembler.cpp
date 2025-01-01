#include "StiffnessMatrixAssemblerMPI.h"
#include <iostream>
#include <mpi.h>

// Helper function to compute the stiffness matrix for an individual element
void StiffnessMatrixAssembler::computeElementStiffness(const Triangle& element, double E, double nu, double Ke[3][3]) {
    // Placeholder: Just for demonstration, not real stiffness calculation
    Ke[0][0] = E / (1 - nu * nu);
    Ke[0][1] = E * nu / (1 - nu * nu);
    Ke[0][2] = 0;

    Ke[1][0] = E * nu / (1 - nu * nu);
    Ke[1][1] = E / (1 - nu * nu);
    Ke[1][2] = 0;

    Ke[2][0] = 0;
    Ke[2][1] = 0;
    Ke[2][2] = E / (2 * (1 + nu)); // Shear modulus for the 2D case
}

void StiffnessMatrixAssembler::assembleGlobalStiffness(int rank, int size, const Graph& mesh, double E, double nu, std::vector<std::vector<double>>& K_global) {
    size_t totalNodes = mesh.points.size();

    // Initialize local stiffness matrix
    std::vector<std::vector<double>> K_local(totalNodes, std::vector<double>(totalNodes, 0.0));

    // Divide triangles among processes
    size_t totalTriangles = mesh.triangles.size();
    size_t trianglesPerProcess = totalTriangles / size;
    size_t remainder = totalTriangles % size;
    size_t startIdx = rank * trianglesPerProcess;
    size_t endIdx = startIdx + trianglesPerProcess + (rank < remainder ? 1 : 0);

    printf("Rank %d: Handling elements from %lu to %lu\n", rank, startIdx, endIdx);

    for (size_t elemIdx = startIdx; elemIdx < endIdx; ++elemIdx) {
        const auto& element = mesh.triangles[elemIdx];

        // Compute local stiffness matrix for the current triangle
        double Ke[3][3];
        computeElementStiffness(element, E, nu, Ke);

        // Get global indices of triangle's nodes
        int globalIndices[3] = {
            mesh.getNodeIndex(element.a),
            mesh.getNodeIndex(element.b),
            mesh.getNodeIndex(element.c)
        };

        // Assemble local contributions into local stiffness matrix
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                int global_i = globalIndices[i];
                int global_j = globalIndices[j];
                K_local[global_i][global_j] += Ke[i][j];
            }
        }
    }

    // Flatten local stiffness matrix for MPI communication
    std::vector<double> K_local_flat(totalNodes * totalNodes, 0.0);
    for (size_t i = 0; i < totalNodes; ++i) {
        for (size_t j = 0; j < totalNodes; ++j) {
            K_local_flat[i * totalNodes + j] = K_local[i][j];
        }
    }

    // Flattened global stiffness matrix for root
    std::vector<double> K_global_flat;
    if (rank == 0) {
        K_global_flat.resize(totalNodes * totalNodes, 0.0);
    }

    // Reduce local matrices to the global matrix on root
    MPI_Reduce(K_local_flat.data(), K_global_flat.data(), totalNodes * totalNodes, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Unflatten the global stiffness matrix at root
    if (rank == 0) {
        K_global.resize(totalNodes, std::vector<double>(totalNodes, 0.0));
        for (size_t i = 0; i < totalNodes; ++i) {
            for (size_t j = 0; j < totalNodes; ++j) {
                K_global[i][j] = K_global_flat[i * totalNodes + j];
            }
        }
    }

    // Ensure all ranks finish
    MPI_Barrier(MPI_COMM_WORLD);
}

