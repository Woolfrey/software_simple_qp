/**
 * @file  : QPSolver.h
 * @author: Jon Woolfrey
 * @date  : August 2023
 * @brief : A class for solving quadratic optimisation problems.
 *
 * This software is publicly available under the GNU General Public License V3.0. You are free to
 * use it and modify it as you see fit. If you find it useful, please acknowledge it.
 *
 * @see https://github.com/Woolfrey/software_simple_qp
 */

#ifndef QP_SOLVER_H
#define QP_SOLVER_H

#include <Eigen/Dense>                                                                              // Linear algebra and matrix decomposition
#include <iostream>                                                                                 // cerr, cout
#include <vector>                                                                                   // vector

/**
 * @brief A data structure for passing options to the QP solver in a single argument.
 */
 template <typename DataType = float>
 struct SolverOptions
 {
    DataType barrierReductionRate = 1e-02;                                                          ///< Multiplier on the barrier size each step
    DataType initialBarrierScalar = 100;                                                            ///< For the log barrier function
    DataType tolerance = 5e-03;                                                                     ///< Terminates solver when step size is below this value.
    unsigned int maxSteps = 20;                                                                     ///< Maximum number of iterations before terminating the algorithm
 };

/**
 * @brief A class for solving convex optimisation problems.
 */
template <class DataType = float>
class QPSolver
{
     public:
     
          /**
           * @brief Constructor.
           * @param options Parameters for the interior point algorithm.
           */
          QPSolver(const SolverOptions<DataType> &options = SolverOptions<DataType>());
               
          /**
           * @brief Minimize 0.5*x'*H*x + x'*f, where x is the decision variable.
           * @param H The Hessian matrix. It is assumed to be positive semi-definite.
           * @param f A vector.
           * @return The optimal solution for x.
           */
          static Eigen::Vector<DataType,Eigen::Dynamic>
          solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                const Eigen::Vector<DataType, Eigen::Dynamic> &f);
     
          /**
           * @brief Linear least squares of a problem y - A*x.
           * @param y The vector of outputs or observations
           * @param A The matrix defining the linear relationship between y and x
           * @param W A positive-definite weighting on the y values.
           * @return The vector x which returns the minimum norm || y - A*x ||
           */             
          static Eigen::Vector<DataType, Eigen::Dynamic>
          least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                        const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                        const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W);

          /**
           * @brief Solve a least squares problem where the decision variable has more elements than the output.
           *        The problem is of the form: min 0.5*(xd - x)'*W*(xd - x) subject to: A*x = y
           * @param xd A desired value for the solution.
           * @param W A weighting on the desired values / solution
           * @param A The matrix for the linear equality constraint
           * @param y Equality constraint vector
           * @return The optimal solution for x.
           */
          static Eigen::Vector<DataType, Eigen::Dynamic>
          redundant_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                  const Eigen::Vector<DataType, Eigen::Dynamic> &y);
                                                                   
          /**
           * @brief Solve linear least squares with upper and lower bounds on the solution.
           * @details The problem is of the form:
           *          min 0.5*(y - A*x)'*W*(y - A*x)
           *          subject to: xMin <= x <= x
           *          This method uses an interior point algorithm to satisfy inequality constraints and thus requires a start point as an argument.
           * @param y The vector component of the linear equation.
           * @param A The matrix that maps x to y.
           * @param xMin The lower bound on the decision variable
           * @param xMax The upper bound on the decision variable.
           * @param x0 A start point for the algorithm.
           * @return The optimal solution within the constraints.
           */
          Eigen::Vector<DataType, Eigen::Dynamic>
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
           *          It uses an interior point algorithm and thus requires a start point as an argument.
           * @param xd Desired value for the solution.
           * @param W Weighting on the desired value / solution.
           * @param A Linear equality constraint matrix.
           * @param y Linear equality constraint vector.
           * @param xMin Lower bound on the solution.
           * @param xMax upper bound on the solution.
           * @param x0 Starting point for the algorithm.
           */                  
          Eigen::Vector<DataType, Eigen::Dynamic>
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
           *          subject to: A*x = y
           *                     B*x < z
           *          It uses an interior point algorithm and thus requires a start point as an argument.
           * @param xd Desired value for the solution.
           * @param W Weighting on the desired value / solution.
           * @param A Equality constraint matrix.
           * @param y Equality constraint vector.
           * @param B Inequality constraint matrix.
           * @param z Inequality constraint vector.
           * @param x0 Starting point for the algorithm.
           */  
          Eigen::Vector<DataType, Eigen::Dynamic>
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
           *          subject to: B*x < z
           *          This method uses an interior point algorithm and thus requires a start point as an argument.
           * @param H A positive semi-definite matrix such that H = H'.
           * @param f A vector for the linear component of the problem.
           * @param B Inequality constraint matrix.
           * @param z Inequality constraint vector.
           * @param x0 Start point for the algorithm.
           * @return x: A solution that minimizes the problem whilst obeying inequality constraints.
           */
          Eigen::Vector<DataType, Eigen::Dynamic>  
          solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                const Eigen::Vector<DataType, Eigen::Dynamic> &f,
                const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                const Eigen::Vector<DataType, Eigen::Dynamic> &x0);
          
          /**
           * @brief Set the tolerance for the step size in the interior point aglorithm.
           * @details The algorithm terminates if alpha*dx < tolerance, where dx is the step and alpha is a scalar.
           * @param tolerance As it says.
           * @return Returns false if the input argument is invalid.
           */
          bool set_tolerance(const DataType &tolerance);
          
          /**
           * @brief Set the maximum number of steps in the interior point algorithm before terminating.
           * @param number As it says.
           * @return Returns false if the input argument is invalid.
           */
          bool set_max_steps(const unsigned int &number);
          
          /**
           * @brief Set the scalar for the constraint barriers in the interior point aglorithm.
           * @details Inequality constraints are converted to a log barrier function. The scalar determines how steep the initial barriers are.
           * @param scalar The initial scalar value when starting the interior point algorithm.
           * @return Returns false if the argument was invalid.
           */
          bool set_barrier_scalar(const DataType &scalar);
          
          /**
           * @brief Set the rate at which the constraint barriers are reduced in the interior point aglorithm.
           * @details The barrier is reduced at a given rate / scale every step in the algorithm to safely approach constraints.
           * @param rate A scalar < 1 which the barrier scalar is reduced by every step.
           * @return Returns false if the argument is invalid.
           */
          bool set_barrier_reduction_rate(const DataType &rate);
          
          /**
           * @brief Returns the step size alpha*||dx|| for the final iteration in the interior point algorithm.
           */
          DataType step_size() const { return this->stepSize; }
          
          /**
           * @brief Returns the number of iterations it took to solve the interior point algorithm.
           */
          unsigned int num_steps() const { return this->numSteps; }
          
          /**
           * @brief Returns the last solution from when the interior point algorithm was previously called.
           */
          Eigen::Vector<DataType, Eigen::Dynamic> last_solution() const { return this->lastSolution; }
          
          /**
           * @brief Clears the last solution such that last_solution().size() == 0.
           */
          void clear_last_solution() { this->lastSolution.resize(0); }
          
          /**
           * @brief The interior point algorithm will use the dual method to solve a redundant QP problem.
           */
          void use_dual();
          
          /**
           * @brief The interior point algorithm will use the primal method to solve a redundant QP problem.
           */
          void use_primal();
          
     private:

          DataType barrierReductionRate = 1e-02;                                                    ///< Constraint barrier scalar is multiplied by this value every step in the interior point algorithm.

          DataType initialBarrierScalar = 100;                                                      ///< Starting value for the constraint barrier scalar in the interior point algorithm.
                    
          DataType tol = 1e-02;                                                                     ///< Minimum value for the step size before terminating the interior point algorithm.
         
          DataType stepSize;                                                                        ///< Step size on the final iteration of the interior point algorithm.
                 
          enum Method {dual, primal} method = primal;                                               ///< Used to select which method to solve for with redundant least squares problems.                                               
          
          unsigned int maxSteps = 20;                                                               ///< Maximum number of iterations to run interior point method before terminating.
          
          unsigned int numSteps = 0;                                                                ///< Records the number of steps it took to solve a problem with the interior point algorithm.
          
          Eigen::Vector<DataType, Eigen::Dynamic> lastSolution;                                     ///< Final solution returned by interior point algorithm. Can be used as a starting point for future calls to the method.
          
          /**
           * @brief The std::min function doesn't like floats, so I had to write my own ಠ_ಠ
           * @return Returns the minimum between to values 'a' and 'b'.
           */
          DataType min(const DataType &a, const DataType &b)
          {
               DataType minimum = (a < b) ? a : b;
               return minimum;
          }
          
};                                                                                                  // Required after class declaration

#include <QPSolver.tpp>

#endif
