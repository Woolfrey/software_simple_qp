/**
 * @file    QPSolver.h
 * @author  Jon Woolfrey
 * @date    August 2023
 * @version 3.0.0
 * @brief   A class for solving convex optimisation (quadratic programming; QP) problems.
 *
 * @details This class provides methods for solving convex optimisation problems through 2 avenues:
 *          1. Unconstrained and equality constrained problems, and
 *          2. Inequality constrained problems, with 2 possible algorithms:
 *               i. Interior point, and
 *              ii. Active set.
 *
 * @copyright This software is publicly available under the GNU General Public License V3.0.
 *            You are free to use it and modify it as you see fit. If you find it useful,
 *            please acknowledge it.
 *
 * @see https://github.com/Woolfrey/software_simple_qp
 */

#ifndef QP_SOLVER_H
#define QP_SOLVER_H

#include <algorithm>                                                                                // std::max, std::min
#include <chrono>                                                                                   // std::chrono::steady_clock
#include <Eigen/Dense>                                                                              // Linear algebra and matrix decomposition
#include <iostream>                                                                                 // std::cout
#include <limits>                                                                                   // std::numeric_limits
#include <stdexcept>                                                                                // std::invalid_argument, std::runtime_error
#include <string>                                                                                   // std::string, std::to_string
#include <type_traits>                                                                              // std::is_same_v
#include <vector>                                                                                   // std::vector

/**
 * @brief A class for solving convex optimisation problems.
 */
template <class DataType = float>
class QPSolver
{
    public:

        static_assert(std::is_same_v<DataType, float> or std::is_same_v<DataType, double>,
                      "[FLAGRANT SYSTEM ERROR] [QP SOLVER] DataType must be float or double.");

        struct Options
        {
            // General
            enum class Method { activeSet, interiorPoint };
            Method method                  = Method::interiorPoint;                                 ///< Determines which algorithm is used.
            DataType minimumStepSize       = 1e-03;                                                 ///< Solver will terminate when norm of step change in decision variable is less than this amount
            DataType maximumSolutionTime   = 1e-02;                                                 ///< Solver will terminate if it takes longer than this.
            unsigned int maximumIterations = 5;                                                     ///< Solver will terminate if this many iterations are reached.

            // Interior point method
            DataType initialBarrierScalar = 100;                                                    ///< Determines how large the slope of the log barrier function is
            DataType barrierReductionRate = 1e-02;                                                  ///< Determines how fast slope of log barrier function shrinks
        };

        struct Result
        {
            enum class Status {converged, maxIterationsReached, timeout};

            Status status                   = Status::converged;                                    ///< Why the solver terminated
            unsigned int numberOfIterations = 1;                                                    ///< How many times the optimisation process repeated to converge on a solution.
            double solutionTime             = 0.0;                                                  ///< How long it took to solve the problem (s)
            DataType finalStepSize          = 0.0;                                                  ///< Magnitude of the last step before the solver terminated
            Eigen::Vector<DataType,Eigen::Dynamic> solution;                                        ///< Result for the decision variable which minimises the objective function
        };

        /**
         * @brief Constructor with default options.
         */
        QPSolver();

        /**
         * @brief Constructor.
         * @param options Parameters for the solver.
         */
        explicit QPSolver(const Options &options);

        /**
         * @brief Minimize 0.5*x'*H*x + x'*f, where x is the decision variable.
         * @param H The Hessian matrix. It is assumed to be positive semi-definite.
         * @param f A vector.
         * @return The optimal solution for x.
         */
        Result
        solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
              const Eigen::Vector<DataType, Eigen::Dynamic> &f);

        /**
         * @brief Linear least squares of a problem y - A*x.
         * @param y The vector of outputs or observations
         * @param A The matrix defining the linear relationship between y and x
         * @param W A positive-definite weighting on the y values.
         * @return The vector x which returns the minimum norm || y - A*x ||
         */
        Result
        least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                      const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                      const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W);

        /**
         * @brief Solve a redundant least squares problem.
         * @details The problem is of the form:
         *          min 0.5*(xd - x)'*W*(xd - x)
         *          subject to: A*x = y
         * @param xd Desired value for the solution.
         * @param W Positive-definite weighting on the desired value / solution.
         * @param A Linear equality constraint matrix.
         * @param y Linear equality constraint vector.
         */
        Result
        redundant_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                const Eigen::Vector<DataType, Eigen::Dynamic> &y);
        /**
         * @brief Solve an under-determined least squares poblem with lower and upper bounds on the solution.
         * @details The problem is of the form:
         *          min 0.5*(y - A*x)^T*W*(y - A*x)
         *          subject to: xMin <= x <= xMax
         *          It uses the method set in the options, and therefore requires an initial guess.
         * @param y The image of x, or output of the system of equations.
         * @param A The linear map from the input x to the output y.
         * @param W A positive-definite weighting matrix on the error.
         * @param xMin A lower bound for values for the solution.
         * @param xMax An upper bound for values for the solution.
         * @param x0 An initial guess for the feasible solution.
         */
        Result
        constrained_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &xMin,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &xMax,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &x0);

        /**
         * @brief Solve a redundant least squares problem with upper and lower bounds on the solution.
         * @details The problem is of the form:
         *          min 0.5*(xd - x)'*W*(xd - x)
         *          subject to: A*x = y
         *                  xMin <= x <= xMax
         *          It uses the method set in the options, and thus requires a start point as an argument.
         * @param xd Desired value for the solution.
         * @param W Weighting on the desired value / solution.
         * @param A Linear equality constraint matrix.
         * @param y Linear equality constraint vector.
         * @param xMin Lower bound on the solution.
         * @param xMax upper bound on the solution.
         * @param x0 Starting point for the algorithm.
         */
        Result
        constrained_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &xMin,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &xMax,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &x0);

        /**
         * @brief Solve a redundant least squares problem with inequality constraints on the solution.
         * @details The problem is of the form:
         *          min 0.5*(xd - x)'*W*(xd - x)
         *          subject to: A*x  = y
         *                      B*x <= z
         *          It uses the method set in the options, and thus requires a start point as an argument.
         * @param xd Desired value for the solution.
         * @param W Weighting on the desired value / solution.
         * @param A Equality constraint matrix.
         * @param y Equality constraint vector.
         * @param B Inequality constraint matrix.
         * @param z Inequality constraint vector.
         * @param x0 Starting point for the algorithm.
         */
        Result
        constrained_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &x0);

        /**
         * @brief Solve a generic quadratic programming problem with inequality constraints.
         * @details The problem is of the form:
         *          min 0.5*x'*H*x + x'*f
         *          subject to: B*x <= z
         *          It uses the method set in the options, and thus requires a start point as an argument.
         * @param H A positive semi-definite matrix such that H = H'.
         * @param f A vector for the linear component of the problem.
         * @param B Inequality constraint matrix.
         * @param z Inequality constraint vector.
         * @param x0 Start point for the algorithm.
         * @return x: A solution that minimizes the problem whilst obeying inequality constraints.
         */
        Result
        solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
              const Eigen::Vector<DataType, Eigen::Dynamic> &f,
              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
              const Eigen::Vector<DataType, Eigen::Dynamic> &z,
              const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
        {
            return solve(H, f,
                         Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>(0, f.size()),
                         Eigen::Vector<DataType, Eigen::Dynamic>(0),
                         B, z, x0);
        }

        /**
         * @brief Solve a generic quadratic programming problem with inequality constraints.
         * @details The problem is of the form:
         *          min 0.5*x'*H*x + x'*f
         *          subject to: A*x  = y
         *                      B*x <= z
         *          It uses the method set in the options, and thus requires a start point as an argument.
         * @param H A positive semi-definite matrix such that H = H'.
         * @param f A vector for the linear component of the problem.
         * @param A Equality constraint matrix.
         * @param y Equality constraint vector.
         * @param B Inequality constraint matrix.
         * @param z Inequality constraint vector.
         * @param x0 Start point for the algorithm.
         * @return x: A solution that minimizes the problem whilst obeying inequality constraints.
         */
        Result
        solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
              const Eigen::Vector<DataType, Eigen::Dynamic> &f,
              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
              const Eigen::Vector<DataType, Eigen::Dynamic> &y,
              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
              const Eigen::Vector<DataType, Eigen::Dynamic> &z,
              const Eigen::Vector<DataType, Eigen::Dynamic> &x0);

    private:

        Eigen::LDLT<Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>> _primalLDLT;            ///< Decomposition of the primal Hessian
        Eigen::LDLT<Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>> _dualLDLT;              ///< Decomposition of the dual Hessian, A * W^-1 * A^T

        // General (declared in the same order as the constructor's initialiser list)
        typename Options::Method _method;                                                           ///< Determines which algorithm is applied
        DataType     _minimumStepSize;                                                              ///< Solver terminates if magnitude of change in decision variable is less than this.
        DataType     _maximumSolutionTime;                                                          ///< Solver terminates if elapsed time is greater than this.
        unsigned int _maximumIterations;                                                            ///< Solver terminates if number of iterations exceeds this.

        // Interior point method
        DataType _initialBarrierScalar;                                                             ///< Determines steepness of slope for log barrier function
        DataType _barrierReductionRate;                                                             ///< Determines how fast log barrier slope shrinks

        /**
         * @brief Convert box constraints xMin <= x <= xMax to the form B * x <= z.
         * @param xMin Lower bound on the solution.
         * @param xMax Upper bound on the solution.
         * @param B Inequality constraint matrix (output).
         * @param z Inequality constraint vector (output).
         */
        static
        void
        box_to_inequality(const Eigen::Vector<DataType, Eigen::Dynamic> &xMin,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &xMax,
                          Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                          Eigen::Vector<DataType, Eigen::Dynamic> &z);

        /**
         * @brief The active set algorithm.
         * @param H The Hessian matrix
         * @param f The linear component of the QP problem
         * @param A The equality constraint matrix
         * @param y The equality constraint vector
         * @param B The inequality constraint matrix
         * @param z The inequality constraint vector
         * @param x0 Start point, or initial guess for the solution.
         */
        Result
        active_set(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                   const Eigen::Vector<DataType, Eigen::Dynamic> &f,
                   const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                   const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                   const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                   const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                   const Eigen::Vector<DataType, Eigen::Dynamic> &x0);

        /**
         * @brief The interior point algorithm.
         * @param H The Hessian matrix
         * @param f The linear component of the QP problem
         * @param A The equality constraint matrix
         * @param y The equality constraint vector
         * @param B The inequality constraint matrix
         * @param z The inequality constraint vector
         * @param x0 Start point, or initial guess for the solution.
         */
        Result
        interior_point(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                       const Eigen::Vector<DataType, Eigen::Dynamic> &f,
                       const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                       const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                       const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                       const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                       const Eigen::Vector<DataType, Eigen::Dynamic> &x0);
};                                                                                                  // Required after class declaration


  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                            Constructors                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
QPSolver<DataType>::QPSolver()
: QPSolver(Options())                                                                               // Delegate with default options
{
}

template <class DataType>
QPSolver<DataType>::QPSolver(const Options &options)
: _method(options.method),
  _minimumStepSize(options.minimumStepSize),
  _maximumSolutionTime(options.maximumSolutionTime),
  _maximumIterations(options.maximumIterations),
  _initialBarrierScalar(options.initialBarrierScalar),
  _barrierReductionRate(options.barrierReductionRate)
{
    // Ensure options are sound
    if (_barrierReductionRate  > 1.0
    or  _barrierReductionRate <= 0.0)
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] Constructor: "
                                    "Barrier reduction rate for interior point method must be between 0 and 1, "
                                    "but received " + std::to_string(_barrierReductionRate) + ".");
    }

    if (_initialBarrierScalar <= 0.0)
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] Constructor: "
                                    "Initial barrier scalar for interior point method must be positive, "
                                    "but received " + std::to_string(_initialBarrierScalar) + ".");
    }

    if (_maximumSolutionTime <= 0.0)
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] Constructor: "
                                    "Maximum solution time for solver must be positive, "
                                    "but received " + std::to_string(_maximumSolutionTime) + ".");
    }

    if (_minimumStepSize <= 0.0)
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] Constructor: "
                                    "Minimum step size for solver must be positive, "
                                    "but received " + std::to_string(_minimumStepSize) + ".");
    }

    if (_maximumIterations == 0)
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] Constructor: "
                                    "Maximum number of iterations must be at least 1.");
    }

    const std::string method = (_method == Options::Method::activeSet)
                             ? "Active Set"
                             : "Interior Point";

    std::cout << "[INFO] [QP SOLVER] Using the " + method + " method." << std::endl;
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                     min 1/2 x' * H * x + x' * f                                //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
typename QPSolver<DataType>::Result
QPSolver<DataType>::solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &f)
{
     if(H.rows() != H.cols())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                      "Expected a square matrix for the Hessian H but it was "
                                      + std::to_string(H.rows()) + "x" + std::to_string(H.cols()) + ".");
     }
     else if(H.rows() != f.size())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                      "Dimensions of arguments do not match. "
                                      "The Hessian H was " + std::to_string(H.rows()) + "x" + std::to_string(H.cols()) +
                                      " and the f vector was " + std::to_string(f.size()) + "x1.");
     }
     else
     {
        Result result;                                                                              // We want to return this

        const auto startTime = std::chrono::steady_clock::now();

        _primalLDLT.compute(H);

        if (_primalLDLT.info() != Eigen::Success or not _primalLDLT.isPositive())
        {
            throw std::runtime_error("[ERROR] [QP SOLVER] solve(): "
                                     "Hessian matrix H could not be decomposed.");
        }

        result.solution = _primalLDLT.solve(-f);                                                    // Too easy lol ᕙ(▀̿̿ĺ̯̿̿▀̿ ̿) ᕗ
        result.solutionTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

        return result;
     }
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                               min 1/2 (y -A * x)' *W *(y - A * x)                              //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
typename QPSolver<DataType>::Result
QPSolver<DataType>::least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W)
{
    if(A.rows() < A.cols())                                                                         // Underdetermined system, use other function
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
                                    "The A matrix has more columns than rows (" + std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + "). "
                                    "Did you mean to call redundant_least_squares()?");
    }
    else if(W.rows() != W.cols())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
                                    "Expected a square weighting matrix W but it was "
                                    + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
    }
    else if(y.size() != A.rows() or W.rows() != A.rows())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
                                    "Dimensions of input arguments do not match. "
                                    "The y vector was " + std::to_string(y.size()) + "x1, "
                                    "the A matrix had " + std::to_string(A.rows()) + " rows, and "
                                    "the weighting matrix W was " + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
    }
    else
    {
        Result result;                                                                              // We want to return this

        const auto startTime = std::chrono::steady_clock::now();

        const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> AtW = A.transpose() * W;      // Compute once, use twice

        _primalLDLT.compute(AtW * A);

        if (_primalLDLT.info() != Eigen::Success or not _primalLDLT.isPositive())
        {
            throw std::runtime_error("[ERROR] [QP SOLVER] least_squares(): "
                                     "Matrix A^T * W * A could not be decomposed. "
                                     "Is W positive definite and A full column rank?");
        }

        result.solution     = _primalLDLT.solve(AtW * y);
        result.solutionTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

        return result;
    }
}

   ////////////////////////////////////////////////////////////////////////////////////////////////////
  //                               min  1/2 (x_d - x)' * W * (x_d - x)                              //
 //                                      subject to: A * x = y                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
typename QPSolver<DataType>::Result
QPSolver<DataType>::redundant_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                            const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                            const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                            const Eigen::Vector<DataType, Eigen::Dynamic> &y)
{
    if(A.rows() >= A.cols())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
                                    "The equality constraint matrix has at least as many rows as columns ("
                                    + std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + "). "
                                    "Did you mean to call least_squares()?");
    }
    else if(W.rows() != W.cols())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
                                    "Expected the weighting matrix to be square but it was "
                                    + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
    }
    else if(xd.size() != W.rows() or W.cols() != A.cols())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
                                    "Dimensions for the decision variable do not match. "
                                    "The desired vector had " + std::to_string(xd.size()) + " elements, "
                                    "the weighting matrix was " + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ", and "
                                    "the constraint matrix had " + std::to_string(A.cols()) + " columns.");
    }
    else if(y.size() != A.rows())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
                                    "Dimensions for the equality constraint do not match. "
                                    "The constraint vector had " + std::to_string(y.size()) + " elements, and "
                                    "the constraint matrix had " + std::to_string(A.rows()) + " rows.");
    }
    else
    {
        Result result;                                                                              // We want to return this

        const auto startTime = std::chrono::steady_clock::now();

        _primalLDLT.compute(W);

        if (_primalLDLT.info() != Eigen::Success or not _primalLDLT.isPositive())
        {
            throw std::runtime_error("[ERROR] [QP SOLVER] redundant_least_squares(): "
                                     "Weighting matrix W could not be decomposed. "
                                     "Is it positive definite?");
        }

        const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> invWAt = _primalLDLT.solve(A.transpose()); // W^-1 * A^T

        _dualLDLT.compute(A * invWAt);                                                              // A * W^-1 * A^T

        if (_dualLDLT.info() != Eigen::Success or not _dualLDLT.isPositive())
        {
            throw std::runtime_error("[ERROR] [QP SOLVER] redundant_least_squares(): "
                                     "Matrix A * W^-1 * A^T could not be decomposed. "
                                     "Is A full row rank?");
        }

        result.solution     = xd + invWAt * _dualLDLT.solve(y - A * xd);
        result.solutionTime = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

        return result;
    }
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //              Private helper, used by both box-constrained least squares functions              //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
void
QPSolver<DataType>::box_to_inequality(const Eigen::Vector<DataType, Eigen::Dynamic> &xMin,
                                      const Eigen::Vector<DataType, Eigen::Dynamic> &xMax,
                                      Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                                      Eigen::Vector<DataType, Eigen::Dynamic> &z)
{
    if (xMin.size() != xMax.size())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] box_to_inequality(): "
                                    "Dimensions of inequality constraints do not match. "
                                    "The xMin argument had " + std::to_string(xMin.size()) + " elements, and "
                                    "the xMax argument had " + std::to_string(xMax.size()) + " elements.");
    }
    else if ((xMin.array() > xMax.array()).any())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] box_to_inequality(): "
                                    "Infeasible bounds. At least one element of xMin is greater than xMax.");
    }

    const auto n = xMin.size();

    // B = [  I ] * x <= z = [  xMax ]
    //     [ -I ]            [ -xMin ]

    B.resize(2*n, n);
    B << Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n),
        -Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);

    z.resize(2*n);
    z << xMax, -xMin;
}

   ////////////////////////////////////////////////////////////////////////////////////////////////////
  //                               min 1/2 (y - A * x)' * W * (y - A * x)                           //
 //                                  subject to:  x_min  <= x <= x_max                             //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
typename QPSolver<DataType>::Result
QPSolver<DataType>::constrained_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &xMin,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &xMax,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
    // Ensure that the input arguments are sound
    if (W.rows() != W.cols())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                    "Expected the weighting matrix W to be square, but it was "
                                    + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
    }
    else if (y.size() != A.rows()
         or  A.rows() != W.rows())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                    "Dimensions of the linear equation do not match. "
                                    "The y vector had " + std::to_string(y.size()) + " elements, "
                                    "the A matrix had " + std::to_string(A.rows()) + " rows, and "
                                    "the weighting matrix W had " + std::to_string(W.rows()) + " rows.");
    }
    else if (A.cols() != xMin.size()
         or  A.cols() != x0.size())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                    "Dimensions for decision variable do not match. "
                                    "The A matrix had " + std::to_string(A.cols()) + " columns, "
                                    "the xMin argument had " + std::to_string(xMin.size()) + " elements, and "
                                    "the start point x0 had " + std::to_string(x0.size()) + " elements.");
    }

    Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> B;
    Eigen::Vector<DataType, Eigen::Dynamic> z;

    box_to_inequality(xMin, xMax, B, z);                                                            // Checks xMin vs. xMax size and ordering

    const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> AtW = A.transpose() * W;          // Compute once, use twice

    return solve(AtW * A, -AtW * y, B, z, x0);
}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
   //                              min 1/2 (x_d - x)' * W * (x_d - x)                                //
  //                                subject to: A * x = y                                           //
 //                                       x_min <= x <= x_max                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
typename QPSolver<DataType>::Result
QPSolver<DataType>::constrained_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &xMin,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &xMax,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
    Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> B;
    Eigen::Vector<DataType, Eigen::Dynamic> z;

    box_to_inequality(xMin, xMax, B, z);                                                            // Throws if bounds are invalid

    return constrained_least_squares(xd, W, A, y, B, z, x0);                                        // Remaining dimensions checked there
}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
   //                                   min 1/2 (xd - x)' * W * (xd - x)                             //
  //                                    subject to: A * x  = y                                      //
 //                                                 B * x <= z                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
typename QPSolver<DataType>::Result
QPSolver<DataType>::constrained_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
    if(xd.size() != W.rows())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                    "Dimensions do not match. "
                                    "The desired value xd had " + std::to_string(xd.size()) + " elements, but "
                                    "the weighting matrix W had " + std::to_string(W.rows()) + " rows.");
    }

    return solve(W, -W * xd, A, y, B, z, x0);                                                       // H = W, f = -W * xd. Remaining checks done in solve()
}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
   //                               min 1/2 x^T * H * x + x^T * f                                    //
  //                                    subject to: A * x  = y                                      //
 //                                                 B * x <= z                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
typename QPSolver<DataType>::Result
QPSolver<DataType>::solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &f,
                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
    // Check inputs are sound
    if (H.rows() != H.cols())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                    "Expected Hessian matrix H to be square, but it was "
                                    + std::to_string(H.rows()) + "x" + std::to_string(H.cols()) + ".");
    }
    else if (H.rows() != f.size()
         or  H.rows() != A.cols()
         or  A.cols() != B.cols()
         or  B.cols() != x0.size())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                    "Dimensions of arguments do not match. "
                                    "The Hessian matrix H had " + std::to_string(H.rows()) + " rows, "
                                    "the f vector had " + std::to_string(f.size()) + " elements, "
                                    "the equality constraint matrix A had " + std::to_string(A.cols()) + " columns, "
                                    "the inequality constraint matrix B had " + std::to_string(B.cols()) + " columns, and "
                                    "the initial guess vector x0 had " + std::to_string(x0.size()) + " elements.");
    }
    else if (A.rows() != y.size())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                    "Dimensions of equality constraint(s) do not match. "
                                    "The matrix A had " + std::to_string(A.rows()) + " rows, and "
                                    "the vector y had " + std::to_string(y.size()) + " elements.");
    }
    else if (B.rows() != z.size())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                    "Dimensions of inequality constraint(s) do not match. "
                                    "The matrix B had " + std::to_string(B.rows()) + " rows, and "
                                    "the vector z had " + std::to_string(z.size()) + " elements.");
    }

    // Pass on to the appropriate algorithm
    switch (_method)
    {
        case Options::Method::activeSet:
            return active_set(H, f, A, y, B, z, x0);

        case Options::Method::interiorPoint:
            return interior_point(H, f, A, y, B, z, x0);
    }

    throw std::logic_error("[FLAGRANT SYSTEM ERROR] [QP SOLVER] solve(): "
                           "Unknown solver method.");                                               // Unreachable, silences warning
}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
   //                               min 1/2 x^T * H * x + x^T * f                                    //
  //                                    subject to: A * x  = y                                      //
 //                                                 B * x <= z                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
typename QPSolver<DataType>::Result
QPSolver<DataType>::active_set(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                               const Eigen::Vector<DataType, Eigen::Dynamic> &f,
                               const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                               const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                               const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                               const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                               const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
    using Matrix = Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Vector<DataType, Eigen::Dynamic>;

    constexpr DataType multiplierTolerance  = 1e-06;                                                // Accounts for floating point error in Lagrange multipliers
    constexpr DataType feasibilityTolerance = 1e-06;                                                // Allowable constraint violation at the start point

    Result result;                                                                                  // We want to return this

    const auto startTime = std::chrono::steady_clock::now();

    _primalLDLT.compute(H);                                                                         // Matrix decomposition

    if (_primalLDLT.info() != Eigen::Success or not _primalLDLT.isPositive())
    {
        throw std::runtime_error("[ERROR] [QP SOLVER] active_set(): "
                                 "Hessian matrix H could not be decomposed. "
                                 "Are you sure it's positive definite?");
    }

    const Eigen::Index n                      = x0.size();
    const Eigen::Index numEqualityConstraints = A.rows();

    const Vector invHf = _primalLDLT.solve(f);                                                      // H^-1 * f

    // Ensure initial guess satisfies A * x0 = y
    Vector x = x0;

    if (numEqualityConstraints > 0) x += A.transpose() * (A * A.transpose()).ldlt().solve(y - A * x); // Project on to the constraint A * x = y

    // Flag the inequality constraints that are violated
    std::vector<Eigen::Index> activeSet;
    std::vector<Eigen::Index> inactiveSet;

    // Activate any violated constraints before beginning
    for (Eigen::Index i = 0; i < z.size(); ++i)
    {
        if (z(i) - B.row(i).dot(x) <= 0.0) activeSet.push_back(i);
        else                               inactiveSet.push_back(i);
    }

    const Eigen::Index numActiveConstraints = static_cast<Eigen::Index>(activeSet.size());

    if (numEqualityConstraints + numActiveConstraints > n)
    {
        throw std::runtime_error("[ERROR] [QP SOLVER] active_set(): "
                                 "Too many constraints at the start point; there were " + std::to_string(numEqualityConstraints) + " "
                                 "equality and " + std::to_string(numActiveConstraints) + " active inequality constraints, but "
                                 "the decision variable has only " + std::to_string(n) + " elements. "
                                 "Try a different start point x0.");
    }

    // Shift the start point onto the equality AND violated inequality constraints simultaneously
    if (numActiveConstraints > 0)
    {
        // C * x = d
        Matrix C(numEqualityConstraints + numActiveConstraints, n);
        Vector d(numEqualityConstraints + numActiveConstraints);

        C.topRows(numEqualityConstraints) = A;
        d.head(numEqualityConstraints)    = y;

        for (Eigen::Index j = 0; j < numActiveConstraints; ++j)
        {
            C.row(numEqualityConstraints + j) = B.row(activeSet[j]);
            d(numEqualityConstraints + j)     = z(activeSet[j]);
        }

        x += C.transpose() * (C * C.transpose()).ldlt().solve(d - C * x);
    }

    // Ensure initial guess is within bounds
    for (const Eigen::Index index : inactiveSet)
    {
        if (B.row(index).dot(x) > z(index) + feasibilityTolerance)
        {
            throw std::runtime_error("[ERROR] [QP SOLVER] active_set(): "
                                     "Could not find a feasible start point; inequality constraint "
                                     + std::to_string(index) + " is violated. Try a different start point x0.");
        }
    }

    // Variables used in the loop, allocated once
    Matrix C;                                                                                       // Active constraint matrix
    Matrix invHCt;                                                                                  // H^-1 * C^T
    Vector invHg(n);                                                                                // H^-1 * gradient
    Vector dx(n);                                                                                   // Step
    Vector lambda;                                                                                  // Lagrange multipliers

    DataType stepSize = 0.0;

    result.status = Result::Status::maxIterationsReached;                                           // Overwritten if we terminate early

    // Run the active set method
    for (unsigned int i = 0; i < _maximumIterations; ++i)
    {
        result.numberOfIterations = i + 1;

        // Terminate early if we're taking too long
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() > _maximumSolutionTime)
        {
            result.status = Result::Status::timeout;
            break;
        }

        const Eigen::Index numActiveNow   = static_cast<Eigen::Index>(activeSet.size());
        const Eigen::Index numConstraints = numEqualityConstraints + numActiveNow;                  // Equality + active inequality

        invHg = x + invHf;                                                                          // H^-1 * (H * x + f)

        if (numConstraints == 0)                                                                    // Unconstrained step
        {
            dx = -invHg;                                                                            // Newton step
            lambda.resize(0);
        }
        else
        {
            // Construct the active constraint matrix
            C.resize(numConstraints, n);
            C.topRows(numEqualityConstraints) = A;

            for (Eigen::Index j = 0; j < numActiveNow; ++j)
            {
                C.row(numEqualityConstraints + j) = B.row(activeSet[j]);                            // Transfer the active constraint in B to C
            }

            invHCt = _primalLDLT.solve(C.transpose());                                              // H^{-1} * C^T

            _dualLDLT.compute(C * invHCt);                                                          // C * H^-1 * C^T

            if (_dualLDLT.info() != Eigen::Success)
            {
                throw std::runtime_error("[ERROR] [QP SOLVER] active_set(): "
                                         "Could not decompose C * H^-1 * C^T. "
                                         "Are the active constraints linearly dependent?");
            }

            lambda = _dualLDLT.solve(C * invHg);                                                    // Lagrange multipliers

            dx = invHCt * lambda - invHg;                                                           // Constrained step
        }

        stepSize = dx.norm();

        if (stepSize <= _minimumStepSize)                                                           // At the minimum for the current active set
        {
            // KKT requires mu = -lambda >= 0, so release the constraint with the most POSITIVE lambda
            DataType largestMultiplier = multiplierTolerance;
            Eigen::Index releasePosition = -1;

            for (Eigen::Index j = 0; j < numActiveNow; ++j)
            {
                if (lambda(numEqualityConstraints + j) > largestMultiplier)                         // Offset for the equality constraints
                {
                    largestMultiplier = lambda(numEqualityConstraints + j);                         // Record largest lagrange multiplier
                    releasePosition   = j;                                                          // Record its index
                }
            }

            if (releasePosition > -1)
            {
                inactiveSet.push_back(activeSet[releasePosition]);                                  // Add to inactive set
                activeSet.erase(activeSet.begin() + releasePosition);                               // Remove from active set
            }
            else
            {
                result.status = Result::Status::converged;                                          // Optimal solution found
                break;
            }
        }
        else
        {
            DataType alpha = 1.0;                                                                   // Scalar for the step size
            Eigen::Index blockingPosition = -1;                                                     // Position in the inactive set

            for (std::size_t j = 0; j < inactiveSet.size(); ++j)
            {
                const Eigen::Index index = inactiveSet[j];

                const DataType addedDistance = B.row(index).dot(dx);                                // How much this step moves toward the constraint

                if (addedDistance > 0.0)                                                            // Moving toward this inactive constraint
                {
                    const DataType ratio = std::max<DataType>(0.0, (z(index) - B.row(index).dot(x)) / addedDistance); // % of distance toward constraint

                    if (ratio < alpha)
                    {
                        alpha            = ratio;                                                   // Record smallest distance toward constraint
                        blockingPosition = static_cast<Eigen::Index>(j);                            // Record constraint number
                    }
                }
            }

            x += alpha * dx;                                                                        // Take a step toward the optimal solution

            if (blockingPosition > -1)                                                              // An inactive constraint blocked us, so activate it
            {
                activeSet.push_back(inactiveSet[blockingPosition]);                                 // Add blocking constraint to active set
                inactiveSet.erase(inactiveSet.begin() + blockingPosition);                          // Remove from inactive set
            }
        }
    }

    // Fill in the results
    result.finalStepSize = stepSize;
    result.solution      = x;
    result.solutionTime  = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

    return result;
}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
   //                               min 1/2 x^T * H * x + x^T * f                                    //
  //                                    subject to: A * x  = y                                      //
 //                                                 B * x <= z                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
typename QPSolver<DataType>::Result
QPSolver<DataType>::interior_point(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                                   const Eigen::Vector<DataType, Eigen::Dynamic> &f,
                                   const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                   const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                   const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                                   const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                                   const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
    using namespace Eigen;

    // h = 0.5*x'*H*x + x'*f - u * sum log(d_i),   d_i = z_i - b_i'*x
    // g = H*x + f + u * sum (1/d_i)*b_i
    // I = H + u * sum (1/d_i^2)*b_i*b_i'

    Result result;                                                                                  // We want to return this

    const auto startTime = std::chrono::steady_clock::now();

    // Variables used in this scope
    DataType stepSize = std::numeric_limits<DataType>::max();                                       // Check for termination
    DataType u = _initialBarrierScalar;                                                             // As it says

    const Index dim                      = x0.size();                                               // Dimensions of the decision variable
    const Index numEqualityConstraints   = A.rows();                                                // Number of equality constraints
    const Index numInequalityConstraints = z.size();                                                // Number of inequality constraints

    Matrix<DataType, Dynamic, Dynamic> I(dim, dim);                                                 // Hessian matrix

    Vector<DataType, Dynamic> g(dim);                                                               // Gradient vector

    std::vector<DataType> d(numInequalityConstraints);                                              // Distance to every constraint
    std::vector<Vector<DataType, Dynamic>> b(numInequalityConstraints);                             // Row vectors of constraint matrix (transposed)
    std::vector<Matrix<DataType, Dynamic, Dynamic>> bbt(numInequalityConstraints);                  // Outer product of row vectors

    // Start at the minimum of the objective subject to A * x = y
    _primalLDLT.compute(H);                                                                         // We need this a couple of times

    Matrix<DataType, Dynamic, Dynamic> invHAT = _primalLDLT.solve(A.transpose());                   // We also need this a couple of times

    _dualLDLT.compute(A * invHAT);

    Matrix<DataType, Dynamic, Dynamic> nullA = Matrix<DataType, Dynamic, Dynamic>::Identity(dim, dim)
                                             - invHAT * _dualLDLT.solve(A);

    Vector<DataType, Dynamic> x = invHAT * _dualLDLT.solve(y) - nullA * _primalLDLT.solve(f);

    // Do some pre-processing on inequality constraints
    std::vector<Index> violatedConstraints;                                                         // To keep track

    for (Index i = 0; i < numInequalityConstraints; ++i)
    {
        b[i]   = B.row(i).transpose();                                                              // Transpose the row vector
        bbt[i] = b[i] * b[i].transpose();                                                           // Compute the outer product

        d[i] = z(i) - b[i].dot(x);                                                                  // Distance to constraint

        if (d[i] <= 0.0) violatedConstraints.push_back(i);
    }

    // Adjust start point for violated inequality constraints.
    // NOTE: Deliberately ignores A * x = y here. If the task and the limits are incompatible,
    //       this trades task accuracy for constraint satisfaction, and the Newton steps preserve it.
    if (not violatedConstraints.empty())
    {
        Matrix<DataType, Dynamic, Dynamic> Bsub(violatedConstraints.size(), dim);
        Vector<DataType, Dynamic> zsub(violatedConstraints.size());

        for (std::size_t i = 0; i < violatedConstraints.size(); ++i)
        {
            Bsub.row(i) = B.row(violatedConstraints[i]);
            zsub(i)     = z(violatedConstraints[i]) - DataType(1e-04);
        }

        x += Bsub.transpose() * (Bsub * Bsub.transpose()).ldlt().solve(zsub - Bsub * x);
    }

    result.status = Result::Status::maxIterationsReached;                                           // Overwritten if we terminate early

    // Run the interior point algorithm
    for (unsigned int i = 0; i < _maximumIterations; ++i)
    {
        result.numberOfIterations = i + 1;                                                          // Increment counter

        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count() > _maximumSolutionTime)
        {
            result.status = Result::Status::timeout;
            break;
        }

        // (Re)set values for new loop
        g = H * x + f;                                                                              // Gradient vector
        I = H;                                                                                      // Hessian matrix

        // Compute the distance to every constraint
        for (Index j = 0; j < numInequalityConstraints; ++j)
        {
            d[j] = z(j) - b[j].dot(x);

            if (i == 0 and d[j] < 0.0)
            {
                throw std::runtime_error("[ERROR] [QP SOLVER] interior_point(): "
                                         "Unable to find a solution that satisfies constraints.");
            }

            if (d[j] <= 0.0) d[j] = DataType(1e-04);                                                // Constraint violated; set a very small, non-zero number

            g += (u / d[j]) * b[j];                                                                 // Add up gradient
            I += (u / (d[j] * d[j])) * bbt[j];                                                      // Add up Hessian
        }

        // Compute Newton step, projected with the barrier Hessian so that A * dx = 0
        _primalLDLT.compute(I);

        const Vector<DataType, Dynamic>          invIg  = _primalLDLT.solve(g);
        const Matrix<DataType, Dynamic, Dynamic> invIAt = _primalLDLT.solve(A.transpose());

        _dualLDLT.compute(A * invIAt);

        Vector<DataType, Dynamic> dx = invIAt * _dualLDLT.solve(A * invIg) - invIg;

        // Compute scalar for step size that we don't violate any constraints
        DataType alpha = 1.0;

        for (Index j = 0; j < numInequalityConstraints; ++j)
        {
            DataType addedDistance = b[j].dot(dx);                                                  // i.e. how much we will move toward inequality constraint

            if (d[j] - addedDistance <= 0.0) alpha = std::min(alpha, DataType(0.9) * d[j] / addedDistance); // Shrink step size so we don't violate constraint
        }

        dx *= alpha;                                                                                // Scale the step size

        stepSize = dx.norm();                                                                       // Save the norm

        if (stepSize <= _minimumStepSize)                                                           // Optimal solution found
        {
            result.status = Result::Status::converged;
            break;
        }

        // Increment values for next loop
        x += dx;
        u *= _barrierReductionRate;
    }

    result.finalStepSize = stepSize;
    result.solution      = x;
    result.solutionTime  = std::chrono::duration<double>(std::chrono::steady_clock::now() - startTime).count();

    return result;
}

#endif
    //
  
/*        NOTE: This code below solve the dual problem. It is very fast, but not 100% reliable.
          // x = xd + W^-1*A'*lambda
          
          // lambda = (A*W^-1*A')^-1*(y - A*xd)
          
          Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> invWAt = W.ldlt().solve(A.transpose()); // Makes calcs a little easier
          
          Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> H = A*invWAt;                       // Hessian matrix for dual problem
          
          Eigen::LDLT<Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic>> Hdecomp(H);            // Saves a bit of time
          
          Eigen::Vector<DataType,Eigen::Dynamic> xr = invWAt*(H,-y,B*invWAt,z,Hdecomp.solve(y));    // Solve the range space
          
          Eigen::Vector<DataType,Eigen::Dynamic> xn = xd - invWAt*Hdecomp.solve(A*xd);              // Compute null space component
          
          DataType alpha = 1.0;
          for(int i = 0; i < z.size(); i++)
          {
               DataType a = B.row(i).dot(xr);
               
               DataType b = B.row(i).dot(xn);
               
               DataType dist = z(i) - a - b;
               
               if(dist <= 0) alpha = min(alpha, 0.99*abs((dist - a)/b));
          }
          
          _results.solution = xr + alpha*xn;
  
          return _results.solution;
*/
