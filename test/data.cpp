/**
 * @file  : data.cpp
 * @author: Jon Woolfrey
 * @date  : October 2025
 * @brief : Executable for generating performance data.
 *
 * This software is publicly available under the GNU General Public License V3.0. You are free to
 * use it and modify it as you see fit. If you find it useful, please acknowledge it.
 *
 * @see https://github.com/Woolfrey/SimpleQPSolver
 */

#include <cstdlib>                                                                                  // std::atoi, srand
#include <ctime>                                                                                    // time
#include <Eigen/Dense>
#include <fstream>                                                                                  // std::ofstream
#include <iostream>                                                                                 // std::cout, std::cerr
#include <QPSolver.h>
#include <string>                                                                                   // std::string

int main(int argc, char *argv[])
{
    using namespace Eigen;

    using Solver = QPSolver<float>;

    // --- Parse command line arguments ---
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " m n sampleSize method\n"
                  << "       method is either activeSet or interiorPoint\n";
        return -1;
    }

    const int m          = std::atoi(argv[1]);
    const int n          = std::atoi(argv[2]);
    const int sampleSize = std::atoi(argv[3]);

    const std::string methodName = argv[4];

    // --- Solver options ---
    Solver::Options options;
    options.maximumIterations    = 100;
    options.minimumStepSize      = 0.001;
    options.initialBarrierScalar = 100;
    options.barrierReductionRate = 0.001;

    if      (methodName == "activeSet")     options.method = Solver::Options::Method::activeSet;
    else if (methodName == "interiorPoint") options.method = Solver::Options::Method::interiorPoint;
    else
    {
        std::cerr << "[ERROR] Unknown method \"" << methodName << "\". "
                  << "Expected \"activeSet\" or \"interiorPoint\".\n";
        return -1;
    }

    Solver solver(options);

    std::cout << "Generating " << sampleSize << " random " << m << "x" << n << " systems.\n"
              << "Here are the solver options:\n"
              << "   - Maximum no. of iterations : " << options.maximumIterations << "\n"
              << "   - Minimum step size         : " << options.minimumStepSize << "\n";

    if (options.method == Solver::Options::Method::interiorPoint)
    {
        std::cout << "   - Initial barrier scalar    : " << options.initialBarrierScalar << "\n"
                  << "   - Barrier reduction rate    : " << options.barrierReductionRate << "\n";
    }

    srand((unsigned int) time(NULL));

    VectorXf xMin = -5 * VectorXf::Ones(n);
    VectorXf xMax =  5 * VectorXf::Ones(n);

    // --- Open CSV file ---
    std::ofstream file("data.csv");
    if (not file.is_open())
    {
        std::cerr << "[ERROR] Could not open file for writing.\n";
        return -1;
    }

    // Write header
    file << "error,time,iterations\n";

    // --- Run experiments ---
    for (int i = 0; i < sampleSize; ++i)
    {
        MatrixXf A     = MatrixXf::Random(m, n);
        VectorXf xTrue = VectorXf::Random(n);
        VectorXf y     = A * xTrue;
        VectorXf xd    = 10 * VectorXf::Random(n);
        VectorXf x0    =  5 * VectorXf::Random(n);

        Solver::Result result;

        if (m < n) result = solver.constrained_least_squares(xd, MatrixXf::Identity(n, n), A, y, xMin, xMax, x0);
        else       result = solver.constrained_least_squares(y, A, MatrixXf::Identity(m, m), xMin, xMax, x0);

        const float error = (y - A * result.solution).norm();

        // Write data to CSV
        file << error << "," << result.solutionTime << "," << result.numberOfIterations << "\n";
    }

    file.close();
    std::cout << "Data saved to data.csv\n";

    return 0;
}
