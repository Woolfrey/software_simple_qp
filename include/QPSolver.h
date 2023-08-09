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
		
		static Eigen::Matrix<DataType, Eigen::Dynamic, 1>
		solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                      const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &f);
	
		                                      
		static Eigen::Matrix<DataType, Eigen::Dynamic, 1>
		least_squares(const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &y,
			      const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
			      const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W);

		static Eigen::Matrix<DataType, Eigen::Dynamic, 1>
		redundant_least_squares(const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xd,
		                        const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
		                        const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
		                        const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &y);
		                                                         
		// These methods require an object since they rely on the interior point solver
		
		Eigen::Matrix<DataType, Eigen::Dynamic, 1>
		constrained_least_squares(const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &y,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xMin,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xMax,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &x0);
		                                        
		Eigen::Matrix<DataType, Eigen::Dynamic, 1>
		constrained_least_squares(const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xd,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &y,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xMin,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xMax,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &x0);
		
		Eigen::Matrix<DataType, Eigen::Dynamic, 1>
		constrained_least_squares(const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xd,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &y,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &z,
		                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &x0);
		
		Eigen::Matrix<DataType, Eigen::Dynamic, 1>  
		solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
		      const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &f,
		      const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
		      const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &z,
		      const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &x0);
		      
		      
		void use_dual() { this->dual = true; }                                              // Solve constrained least squares with dual method
		
		void use_primal() { this->dual = false; }                                           // Solve constrained least squares with primal method
		
		// Methods for setting properties in the interior point solver
		
		bool set_step_size(const DataType &scalar);

		bool set_tolerance(const DataType &tolerance);
		
		bool set_num_steps(const unsigned int &number);
		
		bool set_barrier_scalar(const DataType &scalar);
		
		bool set_barrier_reduction_rate(const DataType &rate);
		
		Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> last_solution() const { return this->lastSolution; }
		
	private:
	
		bool dual = true;
		
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
Eigen::Matrix<DataType, Eigen::Dynamic, 1>
QPSolver<DataType>::solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
		          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &f)
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
	else 	return H.ldlt().solve(-f);                                                          // Too easy lol ᕙ(▀̿̿ĺ̯̿̿▀̿ ̿) ᕗ
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //           Solve an unconstrained least squares problem: min 0.5(y-A*x)'*W*(y-A*x)              //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
Eigen::Matrix<DataType, Eigen::Dynamic, 1>
QPSolver<DataType>::least_squares(const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &y,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W)
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
	else	return (A.transpose()*W*A).ldlt().solve(A.transpose()*W*y);                         // x = (A'*W*A)^-1*A'*W*y
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //    Solve least squares problem of the form min 0.5*(xd - x)'*W*(xd - x) subject to: A*x = y    //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
Eigen::Matrix<DataType, Eigen::Dynamic, 1>
QPSolver<DataType>::redundant_least_squares(const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xd,
                                            const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                            const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                            const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &y)
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
		Eigen::MatrixXf B = W.ldlt().solve(A.transpose());                                  // Makes calcs a little easier
		
		return xd - B*(A*B).ldlt().solve(y - A*xd);                                         // xd - W^-1*A'*(A*W^-1*A')^-1*(y-A*xd)
	}
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //      Solve a constrained problem: min 0.5*(y - A*x)'*W*(y - A*x) s.t. xMin <= x <= xMax        //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
Eigen::Matrix<DataType, Eigen::Dynamic, 1>
QPSolver<DataType>::constrained_least_squares(const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &y,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xMin,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xMax,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &x0)
{
	// Check that the inputs are sound
	if(y.rows() != A.rows() or A.rows() != W.rows())
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): Dimensions of arguments do not match. "
		                            "The y vector argument had " + std::to_string(y.rows()) + " rows, "
		                            "the A matrix argument had " + std::to_string(A.rows()) + " rows, and "
		                            "the weighting matrix argument W was " + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
	}
	else if(W.rows() != W.cols())
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): Expected the weighting matrix W to be "
		                            "square but it was " + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
	}
	else if(A.cols() != xMin.rows() or xMin.rows() != xMax.rows() or xMax.rows() != x0.rows())
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): Dimensions of arguments do not match. "
		                            "The A matrix had " + std::to_string(A.cols()) + " columns, "
		                            "the xMin vector had " + std::to_string(xMin.rows()) + " rows, "
		                            "the xMax vector had " + std::to_string(xMax.rows()) + " rows, "
		                            "and the start point vector x0 had " + std::to_string(x0.rows()) + " rows.");
	}
	
	unsigned int n = x0.size();
	
	// Put the constraints in to standard form Bx <= z where:
	// B = [  I ]   z = [ xMax ]
	//     [ -I ]       [ xMin ]
	
	Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> B(2*n,n);
	B.block(0,0,n,n).setIdentity();
	B.block(n,0,n,n) = -B.block(0,0,n,n);
	
	Eigen::Matrix<DataType,Eigen::Dynamic,1> z(2*n);
	z.head(n) =  xMax;
	z.tail(n) = -xMin;
	
	Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> AtW = A.transpose()*W;              // Makes calcs a little easier
	
	this->lastSolution = solve(AtW*A, -AtW*y, B, z, x0);                                        // Convert to standard QP problem and solve
	
	return this->lastSolution;
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //         Solve a constrained problem min 0.5*(xd - x)'*W*(xd - x) s.t. A*x = y, B*x <= z        //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
Eigen::Matrix<DataType, Eigen::Dynamic, 1>
QPSolver<DataType>::constrained_least_squares(const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xd,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &y,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xMin,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &xMax,
                                              const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &x0)
{
	// Check that the dimensions of the inputs are sound
	if(xd.size() != W.rows()    or W.rows()    != A.cols()
	or A.cols()  != xMin.size() or xMin.size() != xMax.size() or xMax.size() != x0.size())
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                            "Dimensions of input arguments do not match. "
		                            "The desired vector xd had " + std::to_string(xd.size()) + " elements, "
		                            "the weighting matrix W was " + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ", "
		                            "the A matrix had " + std::to_string(A.cols()) + " columns, "
		                            "the xMin vector had " + std::to_string(xMin.size()) + " elements, "
		                            "the xMax vector had " + std::to_string(xMax.size()) + " elements, and "
		                            "the start point x0 had " + std::to_string(x0.size()) + " elements.");
	}
	else if(W.rows() != W.cols())
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                            "Expected the weighting matrix W to be squares, "
		                            "but it was " + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
	}
	else if(A.rows() != y.size())
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                            "Dimensions for the equality constraint do not match. "
		                            "The A matrix had " + std::to_string(A.rows()) + " rows, and "
		                            "the y vector had " + std::to_string(y.size()) + " elements.");
	}
	
	// Primal:
	// min 0.5*(xd - x)'*W*(xd - x)
	// subject to: A*x = y
	//             B*x < z
	
	// Lagrangian L = 0.5*(xd - x)'*W*(xd - x) + lambda'*(y - A*xd)
	
	// Dual:
	// min 0.5*lambda'*A*W^-1*A'*lambda - lambda'*(y - A*xd)
	// subject to B*x < z
	
	// Variables used in this scope:
	Eigen::Matrix<DataType, Eigen::Dynamic, 1> f;                                               // Linear part of QP
	Eigen::Matrix<DataType, Eigen::Dynamic, 1> z;                                               // Constraint vector	
	Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> H;                                  // Hessian
	Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> B;                                  // Constraint matrix
	
	unsigned int n = x0.size();                                                                 // Number of dimensions in the problem
		
	if(this->dual) // Faster, but sensitive to initial conditions
	{
		
		Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> invWAt = W.ldlt().solve(A.transpose()); // Makes calcs a little easier
		
		H = A*invWAt;                                                                       // Hessian matrix for the dual problem
		
		Eigen::LDLT<Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic>> Hdecomp;         // LDLT decomposition of the Hessian
		Hdecomp.compute(H);                                                                 
		
		// Convert the constraints to standard form: B*x <= z
		// B = [  I ]    z = [  xMax ]
		//     [ -I ]        [ -xMin ]
		
		B.resize(2*n,n);
		B.block(0,0,n,n).setIdentity();
		B.block(n,0,n,n) = -B.block(0,0,n,n);
		
		z.resize(2*n);
		z.head(n) =  xMax;
		z.tail(n) = -xMin;
		
		// Ensure null space projection of the desired solution xd is feasible
		Eigen::Matrix<DataType,Eigen::Dynamic,1> xn = xd - invWAt*Hdecomp.solve(A*xd);      // xd projected on to null space of A matrix
		
		float alpha = 1.0;                                                                  // Scalar
		
		for(int i = 0; i < n; i++)
		{
			     if(xn(i) >= xMax(i)) alpha = std::min(xMax(i)/(1.1*xn(i)), alpha);     // If over the limit, reduce alpha
			else if(xn(i) <= xMin(i)) alpha = std::min(xMin(i)/(1.1*xn(i)), alpha);
		}
		
		xn = alpha*xd;                                                                      // New desired vector
		
		f = y - A*xn;                                                                       // Linear component of QP
		
		this->lastSolution = xn - invWAt*solve(H, f, B*invWAt, z - B*xn, Hdecomp.solve(A*(x0 - xn)));
		
		return this->lastSolution;	
	}
	else // primal; slower, but more robust
	{
		// Convert to standard form 0.5*x'*H*x + x'*f subject to B*x >= z
		// where "x" is now [lambda' x' ]'
		
		unsigned int m = y.size();

		// H = [ 0  A ]
		//     [ A' W ]
		H.resize(m+n,m+n);
		H.block(0,0,m,m).setZero();
		H.block(0,m,m,n) = A;
		H.block(m,0,n,m) = A.transpose();
		H.block(m,m,n,n) = W;

		// B = [ 0  I ]
		//     [ 0 -I ]
		B.resize(2*n,m+n);
		B.block(0,0,2*n,m).setZero();
		B.block(0,m,n,n).setIdentity();
		B.block(n,m,n,n) = -B.block(0,m,n,n);

		// z = [  xMax ]
		//     [ -xMin ]
		z.resize(2*n);
		z.head(n) =  xMax;
		z.tail(n) = -xMin;

		// f = [   -y  ]
		//     [ -W*xd ]
		f.resize(m+n);
		f.head(m) = -y;
		f.tail(n) = -W*xd;

		Eigen::VectorXf startPoint(m+n);
		startPoint.head(m) = (A*W.partialPivLu().inverse()*A.transpose()).partialPivLu().solve(A*xd - y);
		startPoint.tail(n) = x0;

		return (solve(H,f,B,z,startPoint)).tail(n);                                         // Convert to standard form and solve
	}
}

  ///////////////////////////////////////////////////////////////////////////////////////////////////
 //          Solve a problem of the form: min 0.5*x'*H*x + x'*f subject to: B*x <= z              //        
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
Eigen::Matrix<DataType, Eigen::Dynamic, 1>
QPSolver<DataType>::solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &f,
                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &z,
                          const Eigen::Matrix<DataType, Eigen::Dynamic, 1>              &x0)
{
	unsigned int dim = x0.rows();                                                               // Number of dimensions

	unsigned int numConstraints = B.rows();                                                     // As it says
	
	// Check that the inputs are sound
	if(H.rows() != H.cols())
	{
		throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): Expected a square matrix for the Hessian "
		                            "but it was " + std::to_string(H.rows()) + "x" + std::to_string(H.cols()) + ".");
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
			
			if( alpha*dotProduct >= d[j] )                                              // If constraint violated on next step...
			{
				DataType temp = (d[j] - 1E-04)/dotProduct;                          // Compute optimal scalar to avoid constraint violation
				
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

#endif
