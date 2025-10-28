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

#include <iostream>   // std::cout, std::cerr
#include <fstream>    // std::ofstream
#include <cstdlib>    // std::atoi
#include <ctime>      // clock_t, clock, CLOCKS_PER_SEC
#include <Eigen/Dense>
#include <QPSolver.h>

int main(int argc, char *argv[])
{
    using namespace Eigen;

    // --- Parse command line arguments ---
    if (argc != 5)
    {
        std::cerr << "Usage: " << argv[0] << " m n sampleSize method\n";
        return -1;
    }

    int m = std::atoi(argv[1]);
    int n = std::atoi(argv[2]);
    int sampleSize = std::atoi(argv[3]);

    // --- Solver options ---
    SolverOptions<float> options;
    options.maxSteps             = 100;
    options.stepSizeTolerance    = 0.001;
    options.initialBarrierScalar = 100;
    options.barrierReductionRate = 0.001;
    options.method               = argv[4];
    
    QPSolver<float> solver(options);
    
    std::cout << "Generating " << sampleSize << " random " << m << "x" << n << " systems.\n"
              << "Here are the solver options:\n"
              << "   - Maximum no. of steps   : " << options.maxSteps << "\n"
              << "   - Step size tolerance    : " << options.stepSizeTolerance << "\n";
  
    if (options.method == "interior point")
    {
        std::cout << "   - Initial barrier scalar : " << options.initialBarrierScalar << "\n"
                  << "   - Barrier reduction rate : " << options.barrierReductionRate << "\n";
    }
           

    srand((unsigned int) time(NULL));

    VectorXf xMin = -5 * VectorXf::Ones(n);
    VectorXf xMax =  5 * VectorXf::Ones(n);

    // --- Open CSV file ---
    std::ofstream file("data.csv");
    if (!file.is_open())
    {
        std::cerr << "[ERROR] Could not open file for writing.\n";
        return -1;
    }

    // Write header
    file << "error,time,steps\n";

    // --- Run experiments ---
    for (int i = 0; i < sampleSize; ++i)
    {
        MatrixXf A     = MatrixXf::Random(m, n);
        VectorXf xTrue = VectorXf::Random(n);
        VectorXf y     = A * xTrue;
        VectorXf xd    = 10 * VectorXf::Random(n);
        VectorXf x0    =  5 * VectorXf::Random(n);

        clock_t timer = clock();
        VectorXf x;
        if (m < n) x = solver.constrained_least_squares(xd, MatrixXf::Identity(n, n), A, y, xMin, xMax, x0);
        else       x = solver.constrained_least_squares(y, A, MatrixXf::Identity(m, m), xMin, xMax, x0);
        
        timer = clock() - timer;

        float t = static_cast<float>(timer) / CLOCKS_PER_SEC;
        float error = (y - A * x).norm();
        int steps = solver.results().numberOfSteps;

        // Write data to CSV
        file << error << "," << t << "," << steps << "\n";
    }

    file.close();
    std::cout << "Data saved to data.csv\n";

    return 0;
}
