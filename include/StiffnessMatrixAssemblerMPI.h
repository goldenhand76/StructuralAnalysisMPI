#ifndef STIFFNESSMATRIXASSEMBLER_H
#define STIFFNESSMATRIXASSEMBLER_H

#include <vector>
#include "DelaunayTriangulationMPI.h"

class StiffnessMatrixAssembler {
public:
    void assembleGlobalStiffness(int rank, int size, const Graph& mesh, double E, double nu, std::vector<std::vector<double>>& K_global);

private:
    void computeElementStiffness(const Triangle& element, double E, double nu, double Ke[3][3]);
};

#endif // STIFFNESSMATRIXASSEMBLER_H
