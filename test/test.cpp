/**
 * @file  : test.cpp
 * @author: Jon Woolfrey
 * @date  : August 2023
 * @brief : Executable for demonstrating and testing the QPSolver class.
 *
 * This software is publicly available under the GNU General Public License V3.0. You are free to
 * use it and modify it as you see fit. If you find it useful, please acknowledge it.
 *
 * @see https://github.com/Woolfrey/SimpleQPSolver
 */

#include <cstdlib>                                                                                  // srand
#include <ctime>                                                                                    // time
#include <iostream>                                                                                 // std::cout
#include <QPSolver.h>                                                                               // Custom class
#include <string>                                                                                   // std::string

int main(int argc, char *argv[])
{
    using Solver = QPSolver<float>;

    // Parse command line argument
    if (argc > 2)
    {
        std::cerr << "Usage: " << argv[0] << " [activeSet | interiorPoint]\n";
        return -1;
    }

    const std::string methodName = (argc == 2) ? argv[1] : "interiorPoint";                        // Default to interior point

    // Options for the solver
    Solver::Options options;
    options.maximumIterations    = 10;                                                              // Terminates after this many iterations
    options.minimumStepSize      = 0.001;                                                           // Terminates if step size is smaller than this
    options.initialBarrierScalar = 100;                                                             // This is for the interior point method
    options.barrierReductionRate = 0.005;                                                           // This is for the interior point method

    if      (methodName == "activeSet")     options.method = Solver::Options::Method::activeSet;
    else if (methodName == "interiorPoint") options.method = Solver::Options::Method::interiorPoint;
    else
    {
        std::cerr << "[ERROR] Unknown method \"" << methodName << "\". "
                  << "Expected \"activeSet\" or \"interiorPoint\".\n";
        return -1;
    }

    Solver solver(options);                                                                         // Create an instance of the class

    // Convert the solver status to text for printing
    auto status_to_string = [](const Solver::Result::Status status) -> std::string
    {
        switch (status)
        {
            case Solver::Result::Status::converged:            return "converged";
            case Solver::Result::Status::maxIterationsReached: return "reached maximum iterations";
            case Solver::Result::Status::timeout:              return "timed out";
        }
        return "unknown";                                                                           // Unreachable, silences warning
    };

    // Variables used in this scope
    unsigned int m, n;                                                                              // Dimensions for the various problems.
    Eigen::MatrixXf H, A, comparison;                                                               // Matrices used in various problems
    Eigen::VectorXf f, x, xMin, xMax, y;                                                            // Vectors used in various problems
    Solver::Result result;                                                                          // Returned by the solver

    srand((unsigned int) time(NULL));                                                               // Seed the random number generator

    std::cout << "\n**********************************************************************\n"
              <<   "*                        A GENERIC QP PROBLEM                        *\n"
              <<   "**********************************************************************\n" << std::endl;

    m = 5;
    n = 5;

    Eigen::MatrixXf temp = Eigen::MatrixXf::Random(n,n);
    H = temp * temp.transpose() + 0.1 * Eigen::MatrixXf::Identity(n,n);                             // Guaranteed positive definite
    f = Eigen::VectorXf::Random(n);

    std::cout << "\nThe SimpleQPSolver can be used to find solutions to generic quadratic programming "
              << "problems of the form:\n"
              << "\n     min 0.5*x'*H*x + x'*f \n"
              << "\nwhere:\n"
              << "   - x (nx1) is the decision variable,\n"
              << "   - H (nxn) is a positive definite matrix such that H = H', and\n"
              << "   - f (nx1) is a vector for the linear component of the quadratic equation.\n"
              << "\nFor example, here is the H matrix (Hessian) for a " << m << "x" << n << " system:\n";

    std::cout << "\n" << H << std::endl;

    std::cout << "\nAnd here is the f vector (transposed):\n";

    std::cout << "\n" << f.transpose() << std::endl;

    std::cout << "\nThen we can call `QPSolver<float>::Result result = solver.solve(H,f);' to get:\n";

    result = solver.solve(H,f);

    std::cout << "\n" << result.solution.transpose() << std::endl;

    std::cout << "\nIt took " << result.solutionTime * 1000 << " ms to solve (" << 1 / result.solutionTime << " Hz).\n"
              << "\nYou could also use doubles here with QPSolver<double>.\n";

    std::cout << "\n**********************************************************************\n"
              <<   "*                      OVERDETERMINED SYSTEMS                        *\n"
              <<   "**********************************************************************\n" << std::endl;

    m = 10;
    n = 6;

    A = Eigen::MatrixXf::Random(m,n);
    y = A*Eigen::VectorXf::Random(n);

    std::cout << "\nThe QP solver can be used to minimize classic least squares / regression "
              << "problems of the form:\n"
              << "\n     min 0.5*|| y - A*x ||^2 = 0.5*(y - A*x)'*W*(y - A*x)\n"
              << "\nwhere:\n"
              << " - y (mx1) is given,\n"
              << " - A (mxn) is also given,\n"
              << " - W (mxm) is a weighting matrix, and\n"
              << " - x (nx1) is the decision variable.\n"
              << "\nIt is assumed that m > n.\n";

    std::cout << "\nFor example, here is a " << m << "x" << n << " system where A is:\n";

    std::cout << "\n" << A << std::endl;

    std::cout << "\nAnd y (transposed) is:\n";

    std::cout << "\n" << y.transpose() << std::endl;

    std::cout << "\nWe can find an approximate solution by calling "
              << "'solver.least_squares(y,A,W)':\n";

    result = solver.least_squares(y,A,Eigen::MatrixXf::Identity(m,m));
    x      = result.solution;

    std::cout << "\n" << x.transpose() << std::endl;

    std::cout << "\nThe error norm ||y - A*x|| is: " << (y - A*x).norm() << ", "
              <<   "and it took " << result.solutionTime * 1000 << " ms to solve (" << 1 / result.solutionTime << " Hz).\n";

    std::cout << "\n**********************************************************************\n"
              <<   "*                      UNDERDETERMINED SYSTEMS                       *\n"
              <<   "**********************************************************************\n" << std::endl;

    m = 6;
    n = 7;

    A = Eigen::MatrixXf::Random(m,n);
    y = A*Eigen::VectorXf::Random(n);

    std::cout << "\nWe can also solve systems where the solution is underdetermined. "
              <<   "For example, the matrix A here is " << m << "x" << n << ":\n";

    std::cout << "\n" << A << std::endl;

    std::cout << "\nAnd the y vector (transposed) is:\n";

    std::cout << "\n" << y.transpose() << std::endl;

    std::cout << "\nSince m < n there are infinite possible solutions. We can give the QP solver a "
              <<   "desired value xd for the solution.\n This problem takes the form:\n";

    std::cout << "\n      min 0.5*(xd - x)'*W*(xd - x)\n"
              <<   "      subject to: A*x = y\n";

    std::cout << "\nWhere W (nxn) is a weighting matrix. We can then call "
              <<   "'solver.redundant_least_squares(xd,W,A,y);' to get:\n";

    result = solver.redundant_least_squares(Eigen::VectorXf::Random(n),
                                            Eigen::MatrixXf::Identity(n,n),
                                            A, y);
    x      = result.solution;

    std::cout << "\n" << x.transpose() << std::endl;

    std::cout << "\nThe error norm ||y - A*x|| is: " << (y - A*x).norm() << ", "
              <<   "and it took " << result.solutionTime * 1000 << " ms to solve (" << 1 / result.solutionTime << " Hz).\n";

    std::cout << "\n**********************************************************************\n"
              <<   "*                      CONSTRAINED SYSTEMS                           *\n"
              <<   "**********************************************************************\n" << std::endl;

    std::cout << "\nUsing the " << methodName << " method for the constrained problems.\n";

    m = 9;
    n = 6;

    A = Eigen::MatrixXf::Random(m,n);
    x.resize(n); x << -3, 5, -1, 6, 0, -2;
    y = A*x;

    xMin = -5*Eigen::VectorXf::Ones(n);
    xMax =  5*Eigen::VectorXf::Ones(n); xMax(1) = 2.7; xMax(3) = 3.6;                               // Manually override the limits

    std::cout << "\nOften there are constraints on the solution. A generic form of this problem is:\n"
              << "\n     min 0.5*x'*H*x + x'*f\n"
              <<   "     subject to: B*x <= z\n"
              << "\nwhere:\n"
              << "   - B (cxn) is a constraint matrix, and\n"
              << "   - z (cx1) is a constraint vector.\n"
              << "\nSince this uses an iterative method for determining a solution, the solver also requires an initial guess x0 (nx1).\n"
              << "We would call: `solver.solve(H,f,B,z,x0);'\n";

    std::cout << "\nA more specific example is for a constrained least squares problem of the form:\n"
              << "\n     min 0.5*(y - A*x)'*W*(y - A*x)\n"
              << "\n     subject to: xMin <= x <= xMax\n"
              << "\nwhere xMin (nx1) and xMax(nx1) are lower and upper bounds on the solution.\n"
              << "\nThe equivalent constraints here are:\n"
              << "\n    B = [  I ]  z = [  xMax ]\n"
              <<   "        [ -I ]      [ -xMin ]\n"
              << "\n(I wrote a special function for this case because I am lazy.)\n";

    std::cout << "\nFor the following system of A:\n";

    std::cout << "\n" << A << std::endl;

    std::cout << "\n and y (transposed):\n";

    std::cout << "\n" << y.transpose() << std::endl;

    std::cout << "\nWe can call 'solver.constrained_least_squares(y,A,W,xMin,xMax,x0);'\n";

    result = solver.constrained_least_squares(y,A,Eigen::MatrixXf::Identity(m,m),xMin,xMax,0.5*(xMin+xMax));
    x      = result.solution;

    std::cout << "\nHere is xMin, the solution x, and xMax side-by-side:\n";
    comparison.resize(n,3);
    comparison.col(0) = xMin;
    comparison.col(1) = x;
    comparison.col(2) = xMax;
    std::cout << "\n" << comparison << std::endl;

    for(int i = 0; i < x.size(); i++)
    {
        if(x(i) < xMin(i) - 1e-06 or x(i) > xMax(i) + 1e-06)
        {
            std::cerr << "\n[FLAGRANT SYSTEM ERROR] CONSTRAINT VIOLATED!"
                      << "  How did that happen? (ー_ーゞ\n";
            return -1;
        }
    }

    std::cout << "\nThe error norm ||y - A*x|| is: " << (y - A*x).norm() << "\n\n"
              <<   "It took " << result.solutionTime * 1000 << " ms to solve (" << 1 / result.solutionTime << " Hz) "
              <<   "in " << result.numberOfIterations << " iterations.\n"
              <<   "Solver status: " << status_to_string(result.status) << ".\n";

    std::cout << "\n**********************************************************************\n"
              <<   "*                CONSTRAINED SYSTEMS (REDUNDANT CASE)                *\n"
              <<   "**********************************************************************\n" << std::endl;

    m = 12;
    n = 17;

    xMin = -3*Eigen::VectorXf::Ones(n);
    xMax =  3*Eigen::VectorXf::Ones(n);

    A                     = Eigen::MatrixXf::Random(m,n);
    Eigen::VectorXf xTrue = Eigen::VectorXf::Random(n);
    y                     = A*xTrue;

    Eigen::VectorXf xd = 5*Eigen::VectorXf::Random(n);

    Eigen::VectorXf x0 =  2*Eigen::VectorXf::Random(n);

    std::cout << "\nWe can even solve redundant systems subject to constraint:\n"
              << "\n      min 0.5*(xd - x)'*W*(xd - x)\n"
              << "     subject to: A*x  = y\n"
              << "                 B*x <= z\n";

    std::cout << "\nAgain, there is a function for least squares problems with lower and upper "
              << "bounds on the solution:\n"
              << "\n      B = [  I ] * x <= z = [  xMax ]"
              << "\n          [ -I ]            [ -xMin ]\n";

    std::cout << "\nWe would call: 'solver.constrained_least_squares(xd,W,A,y,xMin,xMax,x0)'\n";

    std::cout << "\nHere is the solution for a " << m << "x" << n << " system:\n";

    result = solver.constrained_least_squares(xd,Eigen::MatrixXf::Identity(n,n),A,y,xMin,xMax,x0);
    x      = result.solution;

    comparison.resize(n,3);
    comparison.col(0) = xMin;
    comparison.col(1) = x;
    comparison.col(2) = xMax;
    std::cout << "\n" << comparison << std::endl;

    for(int i = 0; i < x.size(); i++)
    {
        if(x(i) < xMin(i) - 1e-04 or x(i) > xMax(i) + 1e-04)
        {
            std::cerr << "\n[FLAGRANT SYSTEM ERROR] CONSTRAINT VIOLATED! How did that happen? (ー_ーゞ\n";
            return -1;
        }
    }

    std::cout << "\nThe error norm is ||y - A*x|| is: " << (y - A*x).norm() << ".\n\n"
              <<   "It took " << result.solutionTime * 1000 << " ms to solve (" << 1.0 / result.solutionTime << " Hz) "
              <<   "in " << result.numberOfIterations << " iterations.\n"
              <<   "Solver status: " << status_to_string(result.status) << ".\n";

    return 0;
}
