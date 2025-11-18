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
    // These are general:
    DataType stepSizeTolerance = 1e-03;                                                             ///< Terminates when norm of change in decision vector is smaller than this
    unsigned int maxSteps = 5;                                                                      ///< Maximum number of iterations before terminating          
    std::string method = "active set";                                                              ///< "interior point" or "active set"
    
    // These are for the interior point method    
    DataType barrierReductionRate = 1e-02;                                                          ///< Multiplier on the barrier size each step
    DataType initialBarrierScalar = 100;                                                            ///< For the log barrier function
};

/**
 * @brief A data structure containing results of the optimisation.
 */
template <typename DataType = float>
struct SolverResults
{
    unsigned int numberOfSteps = 1;                                                                 ///< Number of steps it took to find a solution
    DataType finalStepSize     = std::numeric_limits<float>::max();                                 ///< The final step size at which the algorithm terminated
    DataType objectiveFunction = std::numeric_limits<float>::max();                                 ///< The magnitude of the error for the problem
    Eigen::Vector<DataType,Eigen::Dynamic> solution;                                                ///< The final / last solution when the solver was run
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
        static
        Eigen::Vector<DataType,Eigen::Dynamic>
        solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
              const Eigen::Vector<DataType, Eigen::Dynamic> &f);

        /**
         * @brief Linear least squares of a problem y - A*x.
         * @param y The vector of outputs or observations
         * @param A The matrix defining the linear relationship between y and x
         * @param W A positive-definite weighting on the y values.
         * @return The vector x which returns the minimum norm || y - A*x ||
         */             
        static
        Eigen::Vector<DataType, Eigen::Dynamic>
        least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                      const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                      const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W);

        static
        Eigen::Vector<DataType, Eigen::Dynamic>
        redundant_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                const Eigen::Vector<DataType, Eigen::Dynamic> &y);
        /**
         * @brief Solve an under-determined least squares poblem with lower and upper bounds on the solution.
         * @details The problem is of the form:
         *          min 0.5*(y - A*x)^T*W*(y - A*x)
         *          subject to: xMin <= x <= xMax
         *          It uses an interior point solver and therefore requires an initial guess.
         * @param y The image of x, or output of the system of equations.
         * @param A The linear map from the input x to the output y.
         * @param W A positive-definite weighting matrix on the error.
         * @param xMin A lower bound for values for the solution.
         * @param xMax An upper bound for values for the solution.
         * @param x0 An initial guess for the feasible solution.
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
         *          subject to: A*x = y
         *                      B*x < z
         *          This method uses an interior point algorithm and thus requires a start point as an argument.
         * @param H A positive semi-definite matrix such that H = H'.
         * @param f A vector for the linear component of the problem.
         * @param A Equality constraint matrix.
         * @param y Equality constraint vector.
         * @param B Inequality constraint matrix.
         * @param z Inequality constraint vector.
         * @param x0 Start point for the algorithm.
         * @return x: A solution that minimizes the problem whilst obeying inequality constraints.
         */
        Eigen::Vector<DataType, Eigen::Dynamic>
        solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
              const Eigen::Vector<DataType, Eigen::Dynamic> &f,
              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
              const Eigen::Vector<DataType, Eigen::Dynamic> &y,
              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
              const Eigen::Vector<DataType, Eigen::Dynamic> &z,
              const Eigen::Vector<DataType, Eigen::Dynamic> &x0);

        /**
         * @brief Obtain results for the last solved QP problem of the interior point method.
         * @return A SolverResults stucture, with:
         *         - Number of steps taken to find a solution
         *         - Final step size
         *         - Objective function
         *         - Solution
         */
        SolverResults<DataType>
        results() const { return _results; }

    private:

        SolverResults<DataType> _results;                                                           ///< Performance data on the interior point algorithm

        SolverOptions<DataType> _options;                                                           ///< For the interior point algorithm

        enum Method {activeSet, interiorPoint} _method = activeSet;                                 ///< Determines which algorithm is applied

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
        Eigen::Vector<DataType, Eigen::Dynamic>
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
        Eigen::Vector<DataType, Eigen::Dynamic>
        interior_point(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                       const Eigen::Vector<DataType, Eigen::Dynamic> &f,
                       const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                       const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                       const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                       const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                       const Eigen::Vector<DataType, Eigen::Dynamic> &x0);
        
        /**
        * @brief The std::min function doesn't like floats, so I had to write my own ಠ_ಠ
        * @return Returns the minimum between to values 'a' and 'b'.
        */
        DataType
        min(const DataType &a, const DataType &b)
        {
           return (a < b) ? a : b;
        }
};                                                                                                  // Required after class declaration


  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                            Constructor                                         //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
QPSolver<DataType>::QPSolver(const SolverOptions<DataType> &options)
{
    _options.barrierReductionRate = options.barrierReductionRate;
    _options.initialBarrierScalar = options.initialBarrierScalar;
    _options.stepSizeTolerance    = options.stepSizeTolerance;
    _options.maxSteps             = options.maxSteps;
    
         if (options.method == "active set")     _method = activeSet;
    else if (options.method == "interior point") _method = interiorPoint;
    else
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] set_solver_options(): "
                                    "Method must either be 'active_set' or 'interior point', "
                                    "but received " + options.method);
    }
    
    std::cout << "[INFO] [QP SOLVER] Using the " + options.method + " method." << std::endl;
}
          
  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                     min 1/2 x' * H * x + x' * f                                //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType, Eigen::Dynamic>
QPSolver<DataType>::solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &f)
{
     if(H.rows() != H.cols())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                      "Expected a square matrix for the Hessian H but it was "
                                      + std::to_string(H.rows()) + "x" + std::to_string(H.cols()) + ".");
     }
     else if(H.rows() != f.rows())
     {     
          throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                      "Dimensions of arguments do not match. "
                                      "The Hessian H was " + std::to_string(H.rows()) + "x" + std::to_string(H.cols()) +
                                      " and the f vector was " + std::to_string(f.size()) + "x1.");
     }
     else return H.ldlt().solve(-f);                                                                // Too easy lol ᕙ(▀̿̿ĺ̯̿̿▀̿ ̿) ᕗ
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                               min 1/2 (y -A * x)' *W *(y - A * x)                              //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType, Eigen::Dynamic>
QPSolver<DataType>::least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W)
{
    if(A.rows() < A.cols())                                                                         // Redundant system, use other function
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
                                    "The A matrix has more rows than columns (" + std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + "). "
                                    "Did you mean to call redundant_least_squares()?");                                                      
    }
    
    if(W.rows() != W.cols())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
                                    "Expected a square weighting matrix W but it was "
                                    + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
    }
    else if(y.rows() != W.rows() and W.cols() != A.rows())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
                                    "Dimensions of input arguments do not match. "
                                    "The y vector was " + std::to_string(y.size()) + "x1, "
                                    "the A matrix had " + std::to_string(A.rows()) + " rows, and "
                                    "the weighting matrix W was " + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
    }
    else
    {
        return (A.transpose() * W * A).ldlt().solve(A.transpose() *W * y);
    }
}

   ////////////////////////////////////////////////////////////////////////////////////////////////////
  //                               min  1/2 (x_d - x)' * W * (x_d - x)                              //                         
 //                                      subject to: A * x = y                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType, Eigen::Dynamic>
QPSolver<DataType>::redundant_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                            const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                            const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                            const Eigen::Vector<DataType, Eigen::Dynamic> &y)
{
    if(A.rows() >= A.cols())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
                                    "The equality constraint matrix has more rows than columns ("
                                    + std::to_string(A.rows()) + " >= " + std::to_string(A.cols()) + "). "
                                    "Did you mean to call the other least squares function?");
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
        Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> invWA = W.ldlt().solve(A.transpose()); // Makes calcs a little easier

        return xd + invWA * (A * invWA).ldlt().solve(y - A * xd);
    }
}

   ////////////////////////////////////////////////////////////////////////////////////////////////////
  //                               min 1/2 (y - A * x)' * W * (y - A * x)                           //
 //                                  subject to:  x_min  <= x <= x_max                             //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType,Eigen::Dynamic>
QPSolver<DataType>::constrained_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &xMin,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &xMax,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{    
    // Ensure that the input arguments are sound.
    if (y.size() != A.rows()
    or  A.rows() != W.rows())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                    "Dimensions of the linear equation do not match. "
                                    "The y vector had " + std::to_string(y.size()) + " elements, "
                                    "the A matrix had " + std::to_string(A.rows()) + " rows, and "
                                    "the weighting matrix W had " + std::to_string(W.rows()) + " rows.");
    }
    else if (W.rows() != W.cols())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                    "Expected the weighting matrix W to be square, but it was "
                                    + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
    }
    else if (A.cols()    != xMin.size()
         or  xMin.size() != xMax.size()
         or  xMax.size() != x0.size())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                    "Dimensions for decision variable do not match. "
                                    "The A matrix had " + std::to_string(A.cols()) + " columns, "
                                    "the xMin argument had " + std::to_string(xMin.size()) + " elements, "
                                    "the xMax argument had " + std::to_string(xMax.size()) + " elements, and "
                                    "the start point x0 had " + std::to_string(x0.size()) + " elements.");
    }

    unsigned int n = x0.size();     

    // Convert box constraints to standard form:  
    // B = [  I ] < z = [  xMax ]
    //     [ -I ]       [ -xMin ]

    Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> B(2*n,n);
    B.block(0,0,n,n).setIdentity();
    B.block(n,0,n,n) = -Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic>::Identity(n,n);

    Eigen::Vector<DataType,Eigen::Dynamic> z(2*n);
    z.head(n) =  xMax;
    z.tail(n) = -xMin;

    Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> AtW = A.transpose()*W;                    // Makes calcs a tiny bit faster

    return solve(AtW * A, -AtW * y, B, z, x0);
}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
   //                              min 1/2 (x_d - x)' * W * (x_d - x)                                //
  //                                subject to: A * x = y                                           //
 //                                       x_min <= x <= x_max                                      //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType,Eigen::Dynamic>
QPSolver<DataType>::constrained_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &xMin,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &xMax,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
     // Check the bounds are equal in size
     if(xMin.size() != xMax.size())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                      "Dimensions of inequality constraints do not match. "
                                      "The xMin argument had " + std::to_string(xMin.size()) + " elements, and "
                                      "the xMax argument had " + std::to_string(xMax.size()) + " elements.");
     }
                                 
     unsigned int n = xMin.size();
     
     // Convert box constraints to standard form:
     // B = [  I ] < z = [  xMax ]
     //     [ -I ]     = [ -xMin ]
     
     Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> B(2*n,n);
     B.block(0,0,n,n).setIdentity();
     B.block(n,0,n,n) = -Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic>::Identity(n,n);
     
     Eigen::Vector<DataType,Eigen::Dynamic> z(2*n);
     z.head(n) =  xMax;
     z.tail(n) = -xMin;
     
     return constrained_least_squares(xd, W, A, y, B, z, x0);
}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
   //                                   min 1/2 (xd - x)' * W * (xd - x)                             //
  //                                    subject:  A * x = y                                         //
 //                                               B * x < z                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType,Eigen::Dynamic>
QPSolver<DataType>::constrained_least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &xd,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                                              const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
    // Ensure input arguments are sound
    if(xd.size() != W.rows()
    or  W.rows() != A.cols()
    or  A.cols() != B.cols()
    or  B.cols() != x0.size())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                    "Dimensions for decision variable do not match. "
                                    "The desired value xd had " + std::to_string(xd.size()) + " elements, "
                                    "the weighting matrix W had " + std::to_string(W.rows()) + " rows, "
                                    "the equality constraint matrix A had " + std::to_string(A.cols()) + " columns, "
                                    "the inequality constraint matrix B had " + std::to_string(B.cols()) + " columns, and "
                                    "the start point x0 had " + std::to_string(x0.size()) + " elements.");
    }
    else if(W.rows() != W.cols())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                    "Expected the weighting matrix W to be square, but it was "
                                    + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
    }
    else if(A.rows() != y.size())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                    "Dimensions for equality constraint do not match. "
                                    "The equality constraint matrix A had " + std::to_string(A.rows()) + " rows, and "
                                    "the equality constraint vector y had " + std::to_string(y.size()) + " elements.");
    }
    else if(B.rows() != z.rows())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squared(): "
                                    "Dimensions for inequality constraint do no match. "
                                    "The inequality constraint matrix B had " + std::to_string(B.rows()) + " rows, and "
                                    "the inequality constraint vector z had " + std::to_string(z.size()) + " elements.");
    }

    if(_method == activeSet)
    {
        _results.solution = active_set(W, - W * xd, A, y, B, z, x0);
    }    
    else // _method = interior_point
    { 
        _results.solution = interior_point(W, -W * xd, A, y, B, z, x0);
    }
    
    return _results.solution;                                                                       // Return decision variable x
}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
   //                               min 1/2 x^T * H * x + x^T * f                                    //
  //                                    subject:  A * x = y                                         //
 //                                               B * x < z                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType, Eigen::Dynamic>
QPSolver<DataType>::solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &f,
                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                          const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
    // Check inputs are sound
    if (H.cols() != H.rows())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                    "Expected Hessian matrix H to be square, but it was "
                                    + std::to_string(H.rows()) + " x " + std::to_string(H.cols()));
    }
    else if (H.rows() != f.size()
         or  A.cols() != B.cols()
         or  B.cols() != x0.size())
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                    "Dimensions of arguments do not match. "
                                    "The Hessian matrix H had " + std::to_string(H.cols()) + " columns, "
                                    "the f vector had " + std::to_string(f.size()) + " elements, "
                                    "the equality constraint matrix A had " + std::to_string(A.cols()) + " columns, "
                                    "the inequality constraint matrix B had " + std::to_string(B.cols()) + " columns, and "
                                    "the initial guess vector x0 had " + std::to_string(x0.size()) + " elements.");
    }
    else if (A.rows() != y.size())
    {
            throw std::logic_error("[ERROR] [QP SOLVER] solve(): "
                                   "Dimensions of equality constraint(s) do not match. "
                                   "The matrix A had " + std::to_string(A.rows()) + " rows, and "
                                   "the vector y had " + std::to_string(y.size()) + " elements.");
    }
    else if (B.rows() != z.size())
    {
            throw std::logic_error("[ERROR] [QP SOLVER] solve(): "
                                   "Dimensions of inequality constraint(s) do not match. "
                                   "The matrix B had " + std::to_string(B.rows()) + " rows, and "
                                   "the vector z had " + std::to_string(z.size()) + " elements.");
    }
    
    // Pass on to the appropriate algorithm
    if (_method == activeSet) return     active_set(H, f, A, y, B, z, x0);
    else                      return interior_point(H, f, A, y, B, z, x0);
}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
   //                               min 1/2 x^T * H * x + x^T * f                                    //
  //                                    subject:  A * x = y                                         //
 //                                               B * x < z                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType, Eigen::Dynamic>
QPSolver<DataType>::active_set(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                               const Eigen::Vector<DataType, Eigen::Dynamic> &f,
                               const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                               const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                               const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                               const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                               const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
    using namespace Eigen;                                                                          // For brevity of code

    LDLT<Matrix<DataType, Dynamic, Dynamic>> Hdecomp = H.ldlt();                                    // LDLT decomposition of Hessian matrix
 
    if (Hdecomp.info() != Eigen::Success)
    {
        throw std::runtime_error("[ERROR] [QP SOLVER] active_set(): "
                                 "LDLT decomposition of Hessian matrix failed. "
                                 "Are you sure it's positive definite?");
    }
    
    // Variables used in this scope
    DataType stepSize;
    
    int dim = x0.size();
    int numEqualConstraints   = A.rows();
    int numInequalConstraints = B.rows();
    
    std::vector<int> activeSet;
    std::vector<int> inactiveSet;
    std::vector<int> previousActiveSet;

    // Pre-compute values, allocate memory to save time
    Vector<DataType, Dynamic> x;                                                                    // We want to solve for this
    Vector<DataType, Dynamic> invHf = Hdecomp.solve(f);                                             // H^{-1} * f
    Matrix<DataType, Dynamic, Dynamic> C = A;                                                       // Active constraint matrix

    // Ensure initial guess satisfies A * x0 = y
    x = x0 + A.transpose() * (A * A.transpose()).ldlt().solve(y - A * x0);
    
    // Flag the inequality constraints that are violated
    for (int i = 0; i < z.size(); ++i)
    {
        DataType distance = z(i) - B.row(i).dot(x);
        
        if (distance <= 0.0)   activeSet.push_back(i);
        else                 inactiveSet.push_back(i);
    }
    
    if (activeSet.size() > dim)
    {
        throw std::invalid_argument("[ERROR] [QP SOLVER] active_set(): "
                                    "No feasible solution; detected " + std::to_string(activeSet.size()) + " "
                                    "active constraints, but the decision variable has only "
                                    + std::to_string(x.size()) + " elements.");
    }
    
    // Now shift so that B * x <= z
    Matrix<DataType, Dynamic, Dynamic> Bsub(activeSet.size(), dim);
    Vector<DataType, Dynamic> zsub(activeSet.size());
    
    for (int i = 0; i < activeSet.size(); ++i)
    {
        Bsub.row(i) = B.row(activeSet[i]); 
        zsub(i) = z(activeSet[i]);
    }
    
    x +=  Bsub.transpose() * (Bsub * Bsub.transpose()).ldlt().solve(zsub - Bsub * x);               // Shift the start point
    
    // Run the active set method
    for (int i = 0; i < _options.maxSteps; ++i)
    {
        _results.numberOfSteps = i+1;                                                               // Increment the counter

        Vector<DataType, Dynamic> dx;                                                               // We want to compute this
        Vector<DataType, Dynamic> lambda;                                                           // Lagrange multipliers
        Vector<DataType, Dynamic> invHg = x + invHf;                                                // Gradient to optimal solution
 
        previousActiveSet = activeSet;                                                              // Save this
                
        // Construct the active constraint matrix
        int numConstraints = numEqualConstraints + activeSet.size();
        
        C.conservativeResize(numConstraints, NoChange);
        
        for (int j = 0; j < activeSet.size(); ++j)
        {
            C.row(numEqualConstraints + j) = B.row(activeSet[j]);
        }
        
        Matrix<DataType, Dynamic, Dynamic> invHCt = Hdecomp.solve(C.transpose());                   // Inverse of Hessian * constraint matrix (transposed)
            
        lambda = (C * invHCt).ldlt().solve(C * invHg);                                              // Lagrange multipliers
            
        dx = invHCt * lambda - invHg;                                                               // Constrained step
        
        stepSize = dx.norm();
         
        if (stepSize <= _options.stepSizeTolerance)
        {
            DataType smallestMultiplier = -1e-06;                                                   // Tiny negative number to account for floating point error?
            
            int freeConstraint = -1;                                                                // Use this to track which constraint is most negative
            
            for (int j = 0; j < activeSet.size(); ++j)
            {
                int index = numEqualConstraints + j;                                                // We need to offset the index for all permanent equality constraints
                
                if (lambda[index] < smallestMultiplier)
                {
                    smallestMultiplier = lambda[index];                                             // Save the value
                        freeConstraint = j;                                                         // Remove jth constraint from active set
                }
            }
            
            if (freeConstraint > -1)                                                                // Remove free constraint from active set
            {
                inactiveSet.push_back(activeSet[freeConstraint]);                                   
                activeSet.erase(activeSet.begin() + freeConstraint);
            }
            else break;                                                                             // Optimal solution found
        }
        else
        {
            DataType alpha = 1.0;                                                                   // Scalar for the step size
            
            int blockingConstraint = -1;                                                            // Use this to keep track
                    
            for (int j = 0; j < inactiveSet.size(); ++j)
            {
                int index = inactiveSet[j];                                                         // Makes referencing easier
                
                DataType addedDistance = B.row(index).dot(dx);                                      // i.e. how much this step will move toward inactive constraint

                // Check to see if we are moving toward a currently inactive constraint
                if (addedDistance > 0.0)
                {
                    double ratio = (z(index) - B.row(index).dot(x)) / addedDistance;                // Current distance / added distance
                    
                    if (ratio < alpha)
                    {
                        alpha = ratio;                                                              // Save smallest value
                        blockingConstraint = index;                                                 // Save smallest constraint
                    }
                }
            }
            
            if (alpha * stepSize < _options.stepSizeTolerance) break;                               // Step change is super, duper tiny
                
            x += alpha * dx;                                                                        // Take a step toward optimal solution   
             
            // An inactive constraint blocked us, so add to active set for next loop
            if (alpha < 1.0)
            {
                activeSet.push_back(blockingConstraint);
                
                inactiveSet.erase(std::remove(inactiveSet.begin(), inactiveSet.end(), blockingConstraint),
                                  inactiveSet.end());
            }
            
            // Check to see if constraints have changed
            bool noChange = true;
            if (activeSet.size() == previousActiveSet.size())
            {
                for (int j = 0; j < activeSet.size(); ++j)
                {
                    noChange *= (activeSet[j] == previousActiveSet[j]);
                }
                
                if (noChange) break;
            }
        }
            
    }
    
     _results.finalStepSize     = stepSize;
     _results.objectiveFunction = x.transpose() * (H * x / 2 + f);
     _results.solution          = x;
     
     return x;  
}

    ////////////////////////////////////////////////////////////////////////////////////////////////////
   //                               min 1/2 x^T * H * x + x^T * f                                    //
  //                                    subject:  A * x = y                                         //
 //                                               B * x < z                                        //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType, Eigen::Dynamic>
QPSolver<DataType>::interior_point(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                                   const Eigen::Vector<DataType, Eigen::Dynamic> &f,
                                   const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                   const Eigen::Vector<DataType, Eigen::Dynamic> &y,                                   
                                   const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                                   const Eigen::Vector<DataType, Eigen::Dynamic> &z,
                                   const Eigen::Vector<DataType, Eigen::Dynamic> &x0)
{
    using namespace Eigen;
    
  // std::cout << "Starting the interior point method.\n";
    
    // h = 0.5*x'*H*x + x'*f - sum log(d_i),   d_i = z_i - b_i'*x
    // g = H*x + f + sum (1/d_i)*b_i
    // I = H + sum (1/d_i^2)*b_i*b_i'

    // Variables used in this scope
    DataType stepSize = std::numeric_limits<DataType>::max();                                       // Check for termination
    DataType u = _options.initialBarrierScalar;                                                     // As it says
    
    unsigned int dim = x0.size();                                                                   // Dimensions of the decision variable 
    unsigned int numConstraints = z.size();                                                         // As it says
    
    Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> I(dim,dim);                                      // Hessian matrix
    
    Vector<DataType,Eigen::Dynamic> g(dim);                                                         // Gradient vector
    
    std::vector<DataType> d(numConstraints);                                                        // Distance to every constraint
    std::vector<Eigen::Vector<DataType,Eigen::Dynamic>> b(numConstraints);                          // Row vectors of constraint matrix (transposed)
    std::vector<Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic>> bbt(numConstraints);         // Outer product of row vectors
    
  //  std::cout << "x0 = " << x0.transpose() << "\n";
    
  //  std::cout << "Adjusting the initial guess.\n";
    
    // Shift the initial guess so it satisfies A * x = y
    LDLT<Matrix<DataType, Dynamic, Dynamic>> invH = H.ldlt();                                       // We need this a couple of times
    Matrix<DataType, Dynamic, Dynamic> invHAT = invH.solve(A.transpose());                          // We also need this a couple of times
    LDLT<Matrix<DataType, Dynamic, Dynamic>> AinvHAT = (A * invHAT).ldlt();
    Matrix<DataType, Dynamic, Dynamic> nullA = Matrix<DataType, Dynamic, Dynamic>::Identity(dim, dim)
                                             - invHAT * AinvHAT.solve(A);
                                             
    Vector<DataType, Dynamic> x = invHAT * AinvHAT.solve(y) + nullA * invH.solve(f);

  //  std::cout << "x = " << x.transpose() << "\n";
    
  //  std::cout << "Accounting for inequality constraint violations.\n";
    
    // Do some pre-processing on inequality constraints
    std::vector<int> violatedConstraints;                                                           // To keep track

    for (int i = 0; i < numConstraints; ++i)
    {
        b[i]   = B.row(i).transpose();                                                              // Transpose the row vector
        bbt[i] = b[i]*b[i].transpose();                                                             // Compute the outer product

        d[i] = z(i) - b[i].dot(x);                                                                  // Distance to constraint
        
     //   std::cout << "   Constraint " << i << " distance = " << d[i] << "\n";
        
        if (d[i] <= 0.0) violatedConstraints.push_back(i);
    }
    
    // Adjust start point for violated inequality constraints
    if (not violatedConstraints.empty())
    {
        Matrix<DataType, Dynamic, Dynamic> Bsub(violatedConstraints.size(), dim);
        Vector<DataType, Dynamic> zsub(violatedConstraints.size());
        
        for (int i = 0; i < violatedConstraints.size(); ++i)
        {
            Bsub.row(i) = B.row(violatedConstraints[i]);
            zsub(i) = z(violatedConstraints[i]) - 1e-04;
        }
        
        x += Bsub.transpose() * (Bsub * Bsub.transpose()).ldlt().solve(zsub - Bsub * x);
    }
    
   // std::cout << "y - A * x= " << (y - A * x).transpose() << "\n";
    
  //  std::cout << "x = " << x.transpose() << "\n";
    
    // Run the interior point algorithm
    for (int i = 0; i < _options.maxSteps; ++i)
    {
  //      std::cout << "Step " << i+1 << "\n";
        
        _results.numberOfSteps = i+1;                                                               // Increment counter
        
        // (Re)set values for new loop
        g = H * x + f;                                                                              // Gradient vector
        I = H;                                                                                      // Hessian matrix
        
        // Compute the distance to every constraint
        for (int j = 0; j < numConstraints; ++j)
        {
            d[j] = z(j) - b[j].dot(x);
            
           // std::cout << "   Constraint " << j << " distance = " << d[j] << "\n";
            
            if (i == 0 and d[j] < 0.0)
            {
                throw std::runtime_error("[ERROR] [QP SOLVER] interior_point(): "
                                         "Unable to find a solution that satisfies constraints.");
            }
            
            if (d[j] <= 0.0) d[j] = 1e-04;                                                          // Constraint violated; set a very small, non-zero number

            g += (u / d[j]) * b[j];                                                                 // Add up gradient
            I += (u / (d[j] * d[j])) * bbt[j];                                                      // Add up Hessian
        }

        Vector<DataType, Dynamic> dx = nullA * I.ldlt().solve(-g);                                  // Compute Newton step
        
        // Compute scalar for step size that we don't violate any constraints
        DataType alpha = 1.0;
        
        for (int j = 0; j < numConstraints; ++j)
        {
            DataType addedDistance = b[j].dot(dx);                                                  // i.e. how much we will move toward inequality constraint
            
            if (d[j] - addedDistance <= 0.0) alpha = min(alpha, 0.9 * d[j] / addedDistance);        // Shrink step size multiplier to ensure we don't violated constraint
        }
        
    //    std::cout << "   alpha = " << alpha << "\n";
    //    std::cout << "      dx = " << dx.transpose() << "\n";
        
        dx *= alpha;                                                                                // Scale the step size
        
        stepSize = dx.norm();                                                                       // Save the norm
        
        if (stepSize <= _options.stepSizeTolerance) break;                                          // Optimal solution found
        
        // Increment values for next loop
        x += dx;
        u *= _options.barrierReductionRate;        
    }

    _results.finalStepSize     = stepSize;
    _results.objectiveFunction = x.transpose()*(0.5 * H * x + f);
    _results.solution          = x;

    return x;
}
  
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

#endif
