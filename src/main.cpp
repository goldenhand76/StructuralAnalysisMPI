#include "DelaunayTriangulationMPI.h"
#include "StiffnessMatrixAssemblerMPI.h"
#include "BoundaryConditionsMPI.h"
#include "SolverMPI.h"
#include "PostProcessorMPI.h"
#include "chrono"
#include <random>
#include <omp.h>
#include <mpi.h>


void generatePointsWithHole(Graph& graph, float squareSize, float holeSize) {
    
    // Define the boundaries of the hole
    float halfSquare = squareSize / 2;
    float halfHole = holeSize / 2;
    float holeMinX = halfSquare - halfHole;
    float holeMaxX = halfSquare + halfHole;
    float holeMinY = halfSquare - halfHole;
    float holeMaxY = halfSquare + halfHole;

    //int numberPoints = squareSize * squareSize;
    //std::default_random_engine eng(std::random_device{}());
    //std::uniform_int_distribution<int> dist_w(0, squareSize);
    //std::uniform_int_distribution<int> dist_h(0, squareSize);

    //for (int i = 0; i < numberPoints; ++i) {
    //    Point point = Point(dist_w(eng), dist_h(eng));
    //    if (!(point.x >= holeMinX && point.x <= holeMaxX && point.y >= holeMinY && point.y <= holeMaxY)) {
    //        graph.addPoint(point);
    //    }
    //}

    for (float i = 0; i <= squareSize; ++i) {
        for (float j = 0; j <= squareSize; ++j) {
            // Check if the point is outside the hole boundaries
            if (!(i >= holeMinX && i <= holeMaxX && j >= holeMinY && j <= holeMaxY)) {
                graph.addPoint(Point(i, j));
            }
        }
    }
}

int main(int argc, char** argv) {
    // Convert the input argument to an integer
    int num_threads = std::atoi(argv[1]);

    // Set the number of threads
    omp_set_num_threads(num_threads);

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto start = std::chrono::high_resolution_clock::now(); // Start timing

    Graph graph;
    std::vector<Point> points;

    generatePointsWithHole(graph, 20, 5);

    int numDomains = 16;

    graph.triangulation(rank, size);

    //graph.triangulationWithLocalStorage(rank, size);
     
    //graph.triangulationWithTaskPool();
    
    //graph.triangulationWithDomainDecomposition(numDomains);


    // Print triangles
    //graph.printTriangles();

    MPI_Barrier(MPI_COMM_WORLD);
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto phase1_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> phase1_duration = phase1_end - start;
    std::cout << "Time taken for Phase 1 (Generating triangles): " << phase1_duration.count() << " seconds." << std::endl;
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    StiffnessMatrixAssembler assembler;
    BoundaryConditions bc;
    Solver solver;
    PostProcessor postProcessor;


    // Define material properties
    double E = 210e9; // Young's modulus in Pascals
    double nu = 0.3;  // Poisson's ratio

    // Create a global stiffness matrix (initialize as an empty 2D vector)
    std::vector<std::vector<double>> K_global;
    std::vector<double> F(graph.points.size(), 0.0); // Initialize force vector with zeros

    // Assemble the global stiffness matrix
    if (rank == 0) {
        assembler.assembleGlobalStiffness(rank, size, graph, E, nu, K_global);
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto phase2_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> phase2_duration = phase2_end - phase1_end;
    std::cout << "Time taken for Phase 2 (Assembling global stiffness matrix): " << phase2_duration.count() << " seconds." << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Print the global stiffness matrix
    //std::cout << "\nGlobal Stiffness Matrix:" << std::endl;
    //for (const auto& row : K_global) {
    //    for (const auto& value : row) {
    //        std::cout << value << "\t";
    //    }
    //    std::cout << std::endl;
    //}

    // Define fixed nodes (for example: fix node 0 and node 1)
    std::vector<int> fixedNodes = { 0, 1 };

    // Define loads (for example: apply a load of 1000 N at node 2)
    std::vector<Load> loads = {
        { 2, 1000.0 } // Load of 1000 N applied at node 2
    };

    // Apply boundary conditions
    if (rank == 0) {
        bc.applyBoundaryConditions(rank, size, K_global, F, fixedNodes, loads);
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto phase3_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> phase3_duration = phase3_end - phase2_end;
    std::cout << "Time taken for Phase 3 (Applying boundary conditions): " << phase3_duration.count() << " seconds." << std::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Solve the system for displacements
    std::vector<double> displacements = solver.solveSystem(rank, size, K_global, F);
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto phase4_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> phase4_duration = phase4_end - phase3_end;
    std::cout << "Time taken for Phase 4 (Solving system for displacements): " << phase4_duration.count() << " seconds." << std::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Print the results
    //std::cout << "\nDisplacement Vector:" << std::endl;
    //for (const auto& value : displacements) {
    //    std::cout << value << "\t";
    //}
    //std::cout << std::endl;

    // Post-process to compute the stress field
    std::vector<std::vector<double>> stress_field = postProcessor.computeStressField(rank, size, displacements, graph.triangles, graph.points, E, nu);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    auto phase5_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> phase5_duration = phase5_end - phase4_end;
    std::cout << "Time taken for Phase 5 (Post-processing to compute stress field): " << phase5_duration.count() << " seconds." << std::endl;
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Print the stress results
    //std::cout << "\nStress Field:" << std::endl;
    //for (const auto& stress : stress_field) {
    //    std::cout << "Stress: ";
    //    for (const auto& s : stress) {
    //        std::cout << s << "\t";
    //    }
    //    std::cout << std::endl;
    //}
    MPI_Finalize();

    return 0;
}


// 100 Points : 7 Sec
// 400 Points : 35 Minutes
// CPU Usage  : 80-90%

