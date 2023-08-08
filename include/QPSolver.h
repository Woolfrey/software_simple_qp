#ifndef QPSOLVER_H_
#define QPSOLVER_H_

#include <Eigen/Dense>                                                                              // Linear algebra and matrix decomposition
#include <iostream>                                                                                 // std::cerr, std::cout
#include <vector>                                                                                   // std::vector

template <class DataType = float>
class QPSolver
{
	public:
		QPSolver() {}
		
		
		// These methods can be called without creating an object of this class
		
		template <typename Derived1, typename Derived2>
		static typename Derived2::Matrix solve(const Eigen::MatrixBase<Derived1> &H,
		                                       const Eigen::MatrixBase<Derived2> &f);
		
		template <typename Derived1, typename Derived2, typename Derived3>
		static typename Derived1::Matrix least_squares(const Eigen::MatrixBase<Derived1> &y,
				                               const Eigen::MatrixBase<Derived2> &A,
				                               const Eigen::MatrixBase<Derived3> &W);

		template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
		static typename Derived1::Matrix redundant_least_squares(const Eigen::MatrixBase<Derived1> &xd,
		                                                         const Eigen::MatrixBase<Derived2> &W,
		                                                         const Eigen::MatrixBase<Derived3> &A,
		                                                         const Eigen::MatrixBase<Derived4> &y);
		                                                         
		// These methods require an object since they rely on the interior point solver
		
		bool set_step_size(const DataType &stepSize)
		{
			if(stepSize <= 0)
			{
				std::cerr << "[ERROR] [QP SOLVER] set_step_size(): "
				          << "Input argument was " << std::to_string(stepSize) << " "
				          << "but it must be positive.\n";
				
				return false;
			}
			else
			{
				this->alpha0 = stepSize;
				
				return true;
			}
		}
		
		bool set_barrier_reduction_rate(const DataType &rate)
		{
			if(rate <= 0)
			{
				std::cerr << "[ERROR] [QP SOLVER] set_barrier_reduction_rate(): "
				             "Input argument was " << std::to_string(rate) << " "
				             "but it must be positive.\n";
				
				return false;
			}
			else
			{
				this->beta0 = rate;
				
				return true;
			}
		}
		
		bool set_tolerance(const DataType &tolerance)
		{
			if(tolerance <= 0)
			{
				std::cerr << "[ERROR] [QP SOLVER] set_tolerance(): "
				          << "Input argument was " << std::to_string(tolerance) << " "
				          << "but it must be positive.\n";
				 
				return false;
			}
			else
			{
				this->tol = tolerance;
				
				return true;
			}
		}
		 
		bool set_barrier_scalar(const DataType &scalar)
		{
			if(scalar <= 0)
			{
				std::cerr << "[ERROR] [QP SOLVER] set_barrier_scalar(): "
				          << "Input argument was " << std::to_string(scalar) << " "
				          << "but it must be positive.\n";
				
				return false;
			}
			else
			{
				this->u0 = scalar;
				
				return true;
			}
		}
		
	private:
		
		// These are variables used by the interior point method:
		DataType alpha0 = 1.0;                                                              // Scalar for Newton step
		DataType beta0  = 0.01;                                                             // Rate of decreasing barrier function
		DataType tol    = 1e-2;                                                             // Tolerance on step size
		DataType u0     = 100;                                                              // Scalar on barrier function
		
		int   steps     = 20;                                                               // No. of steps to run interior point method
		
		Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> lastSolution;

};                                                                                                  // Required after class declaration

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //              Solve a standard QP problem of the form min 0.5*x'*H*x + x'*f                     //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
template <typename Derived1, typename Derived2>
typename Derived2::Matrix QPSolver<DataType>::solve(const Eigen::MatrixBase<Derived1> &H,
                                                    const Eigen::MatrixBase<Derived2> &f)
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
	else if(f.cols() != 1)
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
		                            "Expected an " + std::to_string(f.rows()) + "x1 vector "
		                            "but it was " + std::to_string(f.rows()) + "x" + std::to_string(f.cols()));
	}
	else 	return H.ldlt().solve(-f);                                                          // Too easy lol ᕙ(▀̿̿ĺ̯̿̿▀̿ ̿) ᕗ
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //           Solve an unconstrained least squares problem: min 0.5(y-A*x)'*W*(y-A*x)              //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
template <typename Derived1, typename Derived2, typename Derived3>
typename Derived1::Matrix QPSolver<DataType>::least_squares(const Eigen::MatrixBase<Derived1> &y,
                                                            const Eigen::MatrixBase<Derived2> &A,
                                                            const Eigen::MatrixBase<Derived3> &W)
{
	if(A.rows() < A.cols())                                                                     // Redundant system, use other function
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
		                            "The A matrix has more rows than columns ("
		                            + std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + "). "
		                            "Did you mean to call the function for redundant least squares?");	                    		                   
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
	else if(y.cols() != 1)
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
		                            "Expected a " + std::to_string(y.rows()) + "x1 vector for the first "
		                            "argument but it was " + std::to_string(y.rows()) + "x" + std::to_string(y.cols()) + ".");
	}
	else	return (A.transpose()*W*A).ldlt().solve(A.transpose()*W*y);                         // x = (A'*W*A)^-1*A'*W*y
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //    Solve least squares problem of the form min 0.5*(xd - x)'*W*(xd - x) subject to: A*x = y    //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
template <typename Derived1, typename Derived2, typename Derived3, typename Derived4>
typename Derived1::Matrix QPSolver<DataType>::redundant_least_squares(const Eigen::MatrixBase<Derived1> &xd,
                                                                      const Eigen::MatrixBase<Derived2> &W,
                                                                      const Eigen::MatrixBase<Derived3> &A,
                                                                      const Eigen::MatrixBase<Derived4> &y)
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
        else if(xd.cols() != 1)
        {
        	throw std::invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
        	                            "Expected a vector for the first argument but it was "
        	                            + std::to_string(xd.rows()) + "x" + std::to_string(xd.cols()) + ".");
       	}
       	else if(y.cols() != 1)
       	{	
       		throw std::invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
       	                                    "Expected a vector for the final argument but it was "
       	                                    + std::to_string(y.rows()) + "x" + std::to_string(y.cols()) + ".");
       	}
        else
        {   		
		Eigen::MatrixXf B = W.ldlt().solve(A.transpose());                                  // Makes calcs a little easier
		
		return xd - B*(A*B).ldlt().solve(y - A*xd);                                         // xd - W^-1*A'*(A*W^-1*A')^-1*(y-A*xd)
	}
}

#endif
