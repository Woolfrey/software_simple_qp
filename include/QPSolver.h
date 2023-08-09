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
		
		template <typename Derived1, typename Derived2,
		          typename Derived3, typename Derived4, typename Derived5>  
		typename Derived5::Matrix solve(const Eigen::MatrixBase<Derived1> &H,
		                                const Eigen::MatrixBase<Derived2> &f,
		                                const Eigen::MatrixBase<Derived3> &B,
		                                const Eigen::MatrixBase<Derived4> &z,
		                                const Eigen::MatrixBase<Derived5> &x0);
		
		// Methods for setting properties in the interior point solver
		
		bool set_step_size(const DataType &scalar);

		bool set_tolerance(const DataType &tolerance);
		
		bool set_num_steps(const unsigned int &number);
		
		bool set_barrier_scalar(const DataType &scalar);
		
		bool set_barrier_reduction_rate(const DataType &rate);
		
		Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> last_solution() const { return this->lastSolution; }
		
	private:
		
		// These are variables used by the interior point method:
		DataType alpha0 = 1.0;                                                              // Scalar for Newton step
		DataType beta   = 0.01;                                                             // Rate of decreasing barrier function
		DataType tol    = 1e-2;                                                             // Tolerance on step size
		DataType u0     = 100;                                                              // Scalar on barrier function
		
		int steps = 20;                                                                     // No. of steps to run interior point method
		
		Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> lastSolution;               // Can be used for future use

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

  ///////////////////////////////////////////////////////////////////////////////////////////////////
 //            Set the step size scalar in the interior point algorith: x += alpha*dx             //
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
bool QPSolver<DataType>::set_step_size(const DataType &scalar)
{
	if(scalar <= 0)
	{
		std::cerr << "[ERROR] [QP SOLVER] set_step_size(): "
		          << "Input argument was " << std::to_string(scalar) << " "
		          << "but it must be positive.\n";
		
		return false;
	}
	else
	{
		this->alpha0 = scalar;
		
		return true;
	}
}

  ///////////////////////////////////////////////////////////////////////////////////////////////////
 //                  Set the rate at which the barrier scalar reduces: u *= beta                  //
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
bool QPSolver<DataType>::set_barrier_reduction_rate(const DataType &rate)
{
	if(rate <= 0 or rate >= 1)
	{
		std::cerr << "[ERROR] [QP SOLVER] set_barrier_reduction_rate(): "
		             "Input argument was " << std::to_string(rate) << " "
		             "but it must be between 0 and 1.\n";
		             
		return false;
	}
	else
	{
		this->beta = rate;
		
		return true;
	}
}

  ///////////////////////////////////////////////////////////////////////////////////////////////////
 //       Set the magnitude of the step size for which the interior point method terminates       //
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
bool QPSolver<DataType>::set_tolerance(const DataType &tolerance)
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

  ///////////////////////////////////////////////////////////////////////////////////////////////////
 //            Set the number of steps in the interior point method before terminating            //
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
bool QPSolver<DataType>::set_num_steps(const unsigned int &number)
{
	if(number == 0)
	{
		std::cerr << "[ERROR] [QP SOLVER] set_num_steps(): "
		          << "Input argument was 0 but it must be greater than zero.\n";
		
		return false;
	}
	else
	{
		this->steps = number;
		
		return true;
	}
}

  ///////////////////////////////////////////////////////////////////////////////////////////////////
 //               Set the scalar on the barrier function for inequality constraints               //
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
bool QPSolver<DataType>::set_barrier_scalar(const DataType &scalar)
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

  ///////////////////////////////////////////////////////////////////////////////////////////////////
 //          Solve a problem of the form: min 0.5*x'*H*x + x'*f subject to: B*x <= z              //        
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
template <typename Derived1, typename Derived2,
          typename Derived3, typename Derived4, typename Derived5>
typename Derived5::Matrix QPSolver<DataType>::solve(const Eigen::MatrixBase<Derived1> &H,
                                                    const Eigen::MatrixBase<Derived2> &f,
                                                    const Eigen::MatrixBase<Derived3> &B,
                                                    const Eigen::MatrixBase<Derived4> &z,
                                                    const Eigen::MatrixBase<Derived5> &x0)
{
	unsigned int dim = x0.rows();                                                               // Number of dimensions

	unsigned int numConstraints = B.rows();                                                     // As it says
	
	// Check that the inputs are sound
	if(H.rows() != H.cols())
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): Expected a square matrix for the Hessian "
		                            "but it was " + std::to_string(H.rows()) + "x" + std::to_string(H.cols()) + ".");
	}
	else if(x0.cols() != 1)
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): Expected a column vector for the start point argument "
		                            "x0 but it was " + std::to_string(x0.rows()) + "x" + std::to_string(x0.cols()) << ".");
	}
	else if(f.cols() != 1)
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): Expected a column vector for the f vector argument "
		                            "but it was " + std::to_string(f.rows()) + "x" + std::to_string(f.cols()) << ".");
	}
	else if(z.cols() != 1)
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): Expected a column vector for the constraint vector "
					    "argument but it was " + std::to_string(z.rows()) + "x" + std::to_string(z.cols()) << ".");
	}
	else if(H.rows() != dim or f.rows() != dim or B.cols() != dim)
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): Dimensions for the decision variable do not match. "
		                            "The Hessian was " + std::to_string(H.rows()) + "x" + std::to_string(H.cols()) + ", "
		                            "the f vector was " + std::to_string(f.rows()) + "x1, "
		                            "the constraint matrix had " + std::to_string(B.cols()) + " columns, and "
		                            "the start point was " + std::to_string(x0.rows()) + "x1.");
	}
	else if(B.rows() != z.rows())
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): Dimensions for the constraints do not match. "
		                            "The constraint matrix had " + std::to_string(B.rows()) + " rows, and "
		                            "the constraint vector had " + std::to_string(z.rows()) + " rows.");
	}
	
	// Solve the following optimization problem with Guass-Newton method:
	//
	//    min f(x) = 0.5*x'*H*x + x'*f - u*sum(log(d_i))
	//
	// where d_i = z_i - b_i*x is the distance to the constraint
	//
	// Then the gradient and Hessian are:
	//
	//    g(x) = H*x + f - u*sum((1/d_i)*b_i')
	//
	//    I(x) = H + u*sum((1/(d_i^2))*b_i'*b_i)
	
	// Local variables
	DataType u;                                                                                 // Scalar for the barrier function
	DataType alpha;                                                                             // Scalar on the Newton step
	
	Eigen::Matrix<DataType, Eigen::Dynamic, 1> g(dim);                                          // Gradient vector
	Eigen::Matrix<DataType, Eigen::Dynamic, 1> x = x0;                                          // Assign initial state variable
	Eigen::Matrix<DataType, Eigen::Dynamic, 1> dx(dim);                                         // Newton step = -I^-1*g
	Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> I;                                  // Hessian with added barrier function

	std::vector<DataType> d; d.resize(numConstraints);                                          // Distance to every constraint
	
	// Do some pre-processing
	std::vector<Eigen::Matrix<DataType,Eigen::Dynamic,1>> bt(numConstraints);                   // Row vectors of B matrix transposed
	std::vector<Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic>> btb(numConstraints);     // Outer product of row vectors
	
	for(int j = 0; j < numConstraints; j++)
	{
		bt[j]  = B.row(j).transpose();                                                      // Row vector converted to column vector
		btb[j] = B.row(j).transpose()*B.row(j);                                             // Outer product of row vectors
	}
	
	// Run the interior point algorithm
	for(int i = 0; i < this->steps; i++)
	{
		// (Re)set values for new loop
		I = H;                                                                              // Hessian for log-barrier function
		g.setZero();                                                                        // Gradient vector
		
		// Compute distance to each constraint
		for(int j = 0; j < numConstraints; j++)
		{
			d[j] = z(j) - bt[j].dot(x);                                                 // Distance to the jth constraint
			
			if(d[j] <= 0)
			{
				if(i == 0) throw std::runtime_error("[ERROR] [QP SOLVER] solve(): Start point is outside the constraints.");
				
				u   *= 100;                                                         // Increase the barrier function
				d[j] = 1e-03;                                                       // Set a small, non-zero value
			}
					
			g += -(u/d[j])*bt[j];                                                       // Add up gradient vector
			I +=  (u/(d[j]*d[j]))*btb[j];                                               // Add up Hessian
		}
		
		g += H*x + f;                                                                       // Finish summation of gradient vector

		dx = I.ldlt().solve(-g);                                                            // Robust Cholesky decomp
		
		// Ensure the next position is within the constraint
		alpha = this->alpha0;                                                               // Reset the scalar for the step size
		
		for(int j = 0; j < numConstraints; j++)
		{
			DataType dotProduct = bt[j].dot(dx);                                        // Makes calcs a little easier
			
			if( d[j] + alpha*dotProduct < 0 )                                           // If constraint violated on next step...
			{
				DataType temp = (1e-04 - d[j])/dotProduct;                          // Compute optimal scalar to avoid constraint violation
				
				if(temp < alpha) alpha = temp;                                      // If smaller, override
			}
		}

		if(alpha*dx.norm() < this->tol) break;                                              // Change in position is insignificant; must be optimal
		
		// Update values for next loop
		u *= beta;                                                                          // Decrease barrier function
		x += alpha*dx;                                                                      // Increment state
	}
		
	this->lastSolution = x;                                                                     // Save this value for future use
	
	return x;                                                                                   // Return the solution
}

#endif
