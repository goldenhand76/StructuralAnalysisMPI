#include "BoundaryConditionsMPI.h"
#include <mpi.h>
#include <stdexcept>
#include <iostream>

void BoundaryConditions::applyBoundaryConditions(
    int rank, int size, std::vector<std::vector<double>>& K_global,
    std::vector<double>& F, const std::vector<int>& fixedNodes,
    const std::vector<Load>& loads) {
    size_t totalNodes = F.size();

    // Validate dimensions
    if (K_global.size() != totalNodes || K_global[0].size() != totalNodes) {
        throw std::runtime_error("Mismatch in K_global dimensions.");
    }

    // Step 1: Apply boundary conditions for fixed nodes
    size_t numFixedNodes = fixedNodes.size();
    size_t fixedNodesPerRank = (size > 0) ? (numFixedNodes / size) : numFixedNodes;
    size_t startIdx = rank * fixedNodesPerRank;
    size_t endIdx = (rank == size - 1) ? numFixedNodes : startIdx + fixedNodesPerRank;

    if (startIdx >= numFixedNodes) return;  // No fixed nodes to process

    for (size_t i = startIdx; i < endIdx; ++i) {
        int node = fixedNodes[i];
        if (node >= totalNodes) continue;

        for (size_t j = 0; j < totalNodes; ++j) {
            K_global[node][j] = 0.0;
        }
        K_global[node][node] = 1.0;
        F[node] = 0.0;
    }

    // Step 2: Apply loads to the force vector
    // Convert std::vector<Load> to std::vector<int> for node indices
    std::vector<int> loadNodes;
    for (const auto& load : loads) {
        loadNodes.push_back(load.node);
    }

    size_t numLoads = loadNodes.size();
    size_t loadsPerRank = (size > 0) ? (numLoads / size) : numLoads;
    startIdx = rank * loadsPerRank;
    endIdx = (rank == size - 1) ? numLoads : startIdx + loadsPerRank;

    if (startIdx >= numLoads) return;  // No loads to process

    for (size_t i = startIdx; i < endIdx; ++i) {
        int node = loadNodes[i];
        if (node < totalNodes) {
            F[node] = loads[i].value;
        }
    }

    // Step 3: Reduce and broadcast
    std::vector<double> K_flat(totalNodes * totalNodes, 0.0);
    for (size_t i = 0; i < totalNodes; ++i) {
        for (size_t j = 0; j < totalNodes; ++j) {
            K_flat[i * totalNodes + j] = (i < K_global.size() && j < K_global[i].size()) ? K_global[i][j] : 0.0;
        }
    }

    std::vector<double> F_local = F;
    std::vector<double> K_global_flat(totalNodes * totalNodes, 0.0);
    std::vector<double> F_global(totalNodes, 0.0);

    MPI_Reduce(K_flat.data(), K_global_flat.data(), totalNodes * totalNodes, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(F_local.data(), F_global.data(), totalNodes, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (size_t i = 0; i < totalNodes; ++i) {
            for (size_t j = 0; j < totalNodes; ++j) {
                size_t index = i * totalNodes + j;
                K_global[i][j] = (index < K_global_flat.size()) ? K_global_flat[index] : 0.0;
            }
        }
        F = F_global;
    }

    MPI_Bcast(F.data(), totalNodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<double> K_global_flat_bcast(totalNodes * totalNodes, 0.0);
    if (rank == 0) {
        K_global_flat_bcast = K_global_flat;
    }
    MPI_Bcast(K_global_flat_bcast.data(), totalNodes * totalNodes, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        K_global.resize(totalNodes, std::vector<double>(totalNodes, 0.0));
        for (size_t i = 0; i < totalNodes; ++i) {
            for (size_t j = 0; j < totalNodes; ++j) {
                size_t index = i * totalNodes + j;
                K_global[i][j] = (index < K_global_flat_bcast.size()) ? K_global_flat_bcast[index] : 0.0;
            }
        }
    }
}
