/**
 * @file  : QPSolver.h
 * @author: Jon Woolfrey
 * @date  : August 2023
 * @brief : Header file containing template class.
 *
 * This software is publicly available under the GNU General Public License V3.0. You are free to
 * use it and modify it as you see fit. If you find it useful, please acknowledge it.
 *
 * @see https://github.com/Woolfrey/SimpleQPSolver
 */

#ifndef QPSOLVER_H_
#define QPSOLVER_H_

#include <Eigen/Dense>                                                                              // Linear algebra and matrix decomposition
#include <iostream>                                                                                 // cerr, cout
#include <vector>                                                                                   // vector

using namespace Eigen;                                                                              // Eigen::Dynamic, Eigen::Matrix
using namespace std;                                                                                // std::invalid_argument, std::to_string

template <class DataType = float>
class QPSolver
{
	public:
		QPSolver() {}
			
		// These methods can be called without creating an object of this class
		
		static Vector<DataType,Dynamic>
		solve(const Matrix<DataType, Dynamic, Dynamic>  &H,
                      const Vector<DataType, Dynamic>           &f);
	
		                                      
		static Vector<DataType,Dynamic>
		least_squares(const Vector<DataType,Dynamic>           &y,
			      const Matrix<DataType, Dynamic, Dynamic> &A,
			      const Matrix<DataType, Dynamic, Dynamic> &W);

		static Vector<DataType,Dynamic>
		redundant_least_squares(const Vector<DataType, Dynamic>          &xd,
		                        const Matrix<DataType, Dynamic, Dynamic> &W,
		                        const Matrix<DataType, Dynamic, Dynamic> &A,
		                        const Vector<DataType, Dynamic>          &y);
		                                                         
		// These methods require an object since they rely on the interior point solver
		
		Vector<DataType,Dynamic>
		constrained_least_squares(const Vector<DataType, Dynamic>          &y,
		                          const Matrix<DataType, Dynamic, Dynamic> &A,
		                          const Matrix<DataType, Dynamic, Dynamic> &W,
		                          const Vector<DataType, Dynamic>          &xMin,
		                          const Vector<DataType, Dynamic>          &xMax,
		                          const Vector<DataType, Dynamic>          &x0);
		                                        
		Vector<DataType,Dynamic>
		constrained_least_squares(const Vector<DataType, Dynamic>          &xd,
		                          const Matrix<DataType, Dynamic, Dynamic> &W,
		                          const Matrix<DataType, Dynamic, Dynamic> &A,
		                          const Vector<DataType, Dynamic>          &y,
		                          const Vector<DataType, Dynamic>          &xMin,
		                          const Vector<DataType, Dynamic>          &xMax,
		                          const Vector<DataType, Dynamic>          &x0);
		
		Vector<DataType,Dynamic>
		constrained_least_squares(const Vector<DataType, Dynamic>          &xd,
		                          const Matrix<DataType, Dynamic, Dynamic> &W,
		                          const Matrix<DataType, Dynamic, Dynamic> &A,
		                          const Vector<DataType, Dynamic>          &y,
		                          const Matrix<DataType, Dynamic, Dynamic> &B,
		                          const Vector<DataType, Dynamic>          &z,
		                          const Vector<DataType, Dynamic>          &x0);
		
		Vector<DataType,Dynamic>  
		solve(const Matrix<DataType, Dynamic, Dynamic> &H,
		      const Vector<DataType, Dynamic>          &f,
		      const Matrix<DataType, Dynamic, Dynamic> &B,
		      const Vector<DataType, Dynamic>          &z,
		      const Vector<DataType, Dynamic>          &x0);
		      
		// Methods for setting properties in the interior point solver
		
		bool set_step_scalar(const DataType &scalar);

		bool set_tolerance(const DataType &tolerance);
		
		bool set_num_steps(const unsigned int &number);
		
		bool set_barrier_scalar(const DataType &scalar);
		
		bool set_barrier_reduction_rate(const DataType &rate);
		
		DataType step_size() const { return this->stepSize; }
		
		unsigned int num_steps() const { return this->numSteps; }
		
		Vector<DataType, Dynamic> last_solution() const { return this->lastSolution; }
		
		void clear_last_solution() { this->lastSolution.resize(0); }
		
		void use_dual();
		
		void use_primal();
		
	private:
		
		// These are variables used by the interior point method:
		
		DataType alpha0 = 1.0;                                                              // Scalar for Newton step
		DataType beta   = 0.01;                                                             // Rate of decreasing barrier function
		DataType tol    = 1e-4;                                                             // Tolerance on step size
		DataType u0     = 1000;                                                             // Scalar on barrier function
		
		DataType stepSize = 0.0;
		
		enum Method {dual, primal} method = dual;                                                 
		
		unsigned int maxSteps = 20;                                                         // No. of iterations to run interior point method
		
		unsigned int numSteps = 0;                                                          // Number of steps taken to compute a solution
		
		Vector<DataType,Dynamic> lastSolution;                                              // Can be used for future use
		
		DataType min(const DataType &a, const DataType &b)
		{
			DataType minimum = (a < b) ? a : b;                                         // std::min doesn't like floats ಠ_ಠ
			return minimum;
		}
		
};                                                                                                  // Required after class declaration

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //              Solve a standard QP problem of the form min 0.5*x'*H*x + x'*f                     //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> inline
Vector<DataType,Dynamic>
QPSolver<DataType>::solve(const Matrix<DataType, Dynamic, Dynamic> &H,
		          const Vector<DataType, Dynamic>          &f)
{
	if(H.rows() != H.cols())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] solve(): "
		                       "Expected a square matrix for the Hessian H but it was "
		                       + to_string(H.rows()) + "x" + to_string(H.cols()) + ".");
	}
	else if(H.rows() != f.rows())
	{	
		throw invalid_argument("[ERROR] [QP SOLVER] solve(): "
		                       "Dimensions of arguments do not match. "
		                       "The Hessian H was " + to_string(H.rows()) + "x" + to_string(H.cols()) +
		                       " and the f vector was " + to_string(f.size()) + "x1.");
	}
	else 	return H.ldlt().solve(-f);                                                          // Too easy lol ᕙ(▀̿̿ĺ̯̿̿▀̿ ̿) ᕗ
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //           Solve an unconstrained least squares problem: min 0.5(y-A*x)'*W*(y-A*x)              //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> inline
Vector<DataType,Dynamic>
QPSolver<DataType>::least_squares(const Vector<DataType, Dynamic>          &y,
                                  const Matrix<DataType, Dynamic, Dynamic> &A,
                                  const Matrix<DataType, Dynamic, Dynamic> &W)
{
	if(A.rows() < A.cols())                                                                     // Redundant system, use other function
	{
		throw invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
		                       "The A matrix has more rows than columns ("
		                       + to_string(A.rows()) + "x" + to_string(A.cols()) + "). "
		                       "Did you mean to call redundant_least_squares()?");	                    		                   
	}
	if(W.rows() != W.cols())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
		                       "Expected a square weighting matrix W but it was "
		                       + to_string(W.rows()) + "x" + to_string(W.cols()) + ".");
	}
	else if(y.rows() != W.rows() and W.cols() != A.rows())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
		                       "Dimensions of input arguments do not match. "
		                       "The y vector was " + to_string(y.size()) + "x1, "
		                       "the A matrix had " + to_string(A.rows()) + " rows, and "
		                       "the weighting matrix W was " + to_string(W.rows()) + "x" + to_string(W.cols()) + ".");
	}
	else	return (A.transpose()*W*A).ldlt().solve(A.transpose()*W*y);                         // x = (A'*W*A)^-1*A'*W*y
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //    Solve least squares problem of the form min 0.5*(xd - x)'*W*(xd - x) subject to: A*x = y    //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> inline
Vector<DataType,Dynamic>
QPSolver<DataType>::redundant_least_squares(const Vector<DataType, Dynamic>          &xd,
                                            const Matrix<DataType, Dynamic, Dynamic> &W,
                                            const Matrix<DataType, Dynamic, Dynamic> &A,
                                            const Vector<DataType, Dynamic>          &y)
{
	if(A.rows() >= A.cols())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
		                       "The equality constraint matrix has more rows than columns ("
		                       + to_string(A.rows()) + " >= " + to_string(A.cols()) + "). "
		                       "Did you mean to call the other least squares function?");
	}
	else if(W.rows() != W.cols())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
		                       "Expected the weighting matrix to be square but it was "
		                       + to_string(W.rows()) + "x" + to_string(W.cols()) + ".");
	}
	else if(xd.size() != W.rows() or W.cols() != A.cols())
	{	
		throw invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
		                       "Dimensions for the decision variable do not match. "
		                       "The desired vector had " + to_string(xd.size()) + " elements, "
		                       "the weighting matrix was " + to_string(W.rows()) + "x" + to_string(W.cols()) + ", and "
		                       "the constraint matrix had " + to_string(A.cols()) + " columns.");
        }
        else if(y.size() != A.rows())
        {    	
        	throw invalid_argument("[ERROR] [QP SOLVER] redundant_least_squares(): "
        	                       "Dimensions for the equality constraint do not match. "
        	                       "The constraint vector had " + to_string(y.size()) + " elements, and "
        	                       "the constraint matrix had " + to_string(A.rows()) + " rows.");
        }
        else
        {   		
		MatrixXf B = W.ldlt().solve(A.transpose());                                         // Makes calcs a little easier
		
		return xd + B*(A*B).ldlt().solve(y - A*xd);                                         // xd - W^-1*A'*(A*W^-1*A')^-1*(y-A*xd)
	}
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //      Solve a constrained problem: min 0.5*(y - A*x)'*W*(y - A*x) s.t. xMin <= x <= xMax        //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> inline
Vector<DataType,Dynamic>
QPSolver<DataType>::constrained_least_squares(const Vector<DataType,Dynamic>           &y,
                                              const Matrix<DataType, Dynamic, Dynamic> &A,
                                              const Matrix<DataType, Dynamic, Dynamic> &W,
                                              const Vector<DataType,Dynamic>           &xMin,
                                              const Vector<DataType,Dynamic>           &xMax,
                                              const Vector<DataType,Dynamic>           &x0)
{
	// Check that the inputs are sound
	if(y.rows() != A.rows() or A.rows() != W.rows())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                       "Dimensions of arguments do not match. "
		                       "The y vector argument had " + to_string(y.rows()) + " rows, "
		                       "the A matrix argument had " + to_string(A.rows()) + " rows, and "
		                       "the weighting matrix argument W was " + to_string(W.rows()) + "x" + to_string(W.cols()) + ".");
	}
	else if(W.rows() != W.cols())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                       "Expected the weighting matrix W to be square but "
		                       "it was " + to_string(W.rows()) + "x" + to_string(W.cols()) + ".");
	}
	else if(A.cols() != xMin.rows() or xMin.rows() != xMax.rows() or xMax.rows() != x0.rows())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                       "Dimensions of arguments do not match. "
		                       "The A matrix had " + to_string(A.cols()) + " columns, "
		                       "the xMin vector had " + to_string(xMin.rows()) + " rows, "
		                       "the xMax vector had " + to_string(xMax.rows()) + " rows, "
		                       "and the start point vector x0 had " + to_string(x0.rows()) + " rows.");
	}
	
	unsigned int n = x0.size();
	
	// Put the constraints in to standard form Bx <= z where:
	// B = [  I ]   z = [  xMax ]
	//     [ -I ]       [ -xMin ]
	
	Matrix<DataType,Dynamic,Dynamic> B(2*n,n);
	B.block(0,0,n,n).setIdentity();
	B.block(n,0,n,n) = -B.block(0,0,n,n);
	
	Matrix<DataType,Dynamic,1> z(2*n);
	z.head(n) =  xMax;
	z.tail(n) = -xMin;
	
	Matrix<DataType, Dynamic, Dynamic> AtW = A.transpose()*W;                                   // Makes calcs a little easier
	
	this->lastSolution = solve(AtW*A, -AtW*y, B, z, x0);                                        // Convert to standard QP problem and solve
	
	return this->lastSolution;
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //    Solve a constrained problem min 0.5*(xd - x)'*W*(xd - x) s.t. A*x = y, xMin <= x <= xMax    //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> inline
Vector<DataType,Dynamic>
QPSolver<DataType>::constrained_least_squares(const Vector<DataType,Dynamic>           &xd,
                                              const Matrix<DataType, Dynamic, Dynamic> &W,
                                              const Matrix<DataType, Dynamic, Dynamic> &A,
                                              const Vector<DataType,Dynamic>           &y,
                                              const Vector<DataType,Dynamic>           &xMin,
                                              const Vector<DataType,Dynamic>           &xMax,
                                              const Vector<DataType,Dynamic>           &x0)
{
	// Check that the dimensions of the inputs are sound
	if(xd.size() != W.rows()    or W.rows()    != A.cols()
	or A.cols()  != xMin.size() or xMin.size() != xMax.size() or xMax.size() != x0.size())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                       "Dimensions of input arguments do not match. "
		                       "The desired vector xd had " + to_string(xd.size()) + " elements, "
		                       "the weighting matrix W was " + to_string(W.rows()) + "x" + to_string(W.cols()) + ", "
		                       "the A matrix had " + to_string(A.cols()) + " columns, "
		                       "the xMin vector had " + to_string(xMin.size()) + " elements, "
		                       "the xMax vector had " + to_string(xMax.size()) + " elements, and "
		                       "the start point x0 had " + to_string(x0.size()) + " elements.");
	}
	else if(W.rows() != W.cols())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                       "Expected the weighting matrix W to be squares, "
		                       "but it was " + to_string(W.rows()) + "x" + to_string(W.cols()) + ".");
	}
	else if(A.rows() != y.size())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                       "Dimensions for the equality constraint do not match. "
		                       "The A matrix had " + to_string(A.rows()) + " rows, and "
		                       "the y vector had " + to_string(y.size()) + " elements.");
	}
	

	// Lagrangian L = 0.5*(xd - x)'*W*(xd - x) + lambda'*(y - A*xd)
	
	// Primal:
	// min 0.5*(xd - x)'*W*(xd - x)
	// subject to: A*x = y
	//             B*x < z
	
	// Dual:
	// min 0.5*lambda'*A*W^-1*A'*lambda - lambda'*(y - A*xd)
	// subject to B*x < z
	
	unsigned int n = x0.size();                                                                 // Number of dimensions in the problem

	Matrix<DataType,Dynamic,Dynamic> invWAt = W.ldlt().solve(A.transpose());                    // Makes calcs a little easier
	
	switch(method)
	{
		case dual:
		{
			Matrix<DataType,Dynamic,Dynamic> H = A*invWAt;                              // Hessian matrix for the dual problem
			
			Eigen::LDLT<Matrix<DataType,Dynamic,Dynamic>> Hdecomp; Hdecomp.compute(H);  // LDL' decomposition                                                         
			
			// Convert the constraints to standard form: B*x <= z
			// B = [  I ]    z = [  xMax ]
			//     [ -I ]        [ -xMin ]
			Matrix<DataType, Dynamic, Dynamic> B(2*n,n);                                // Constraint matrix
			B.block(0,0,n,n).setIdentity();
			B.block(n,0,n,n) = -B.block(0,0,n,n);
			
			Vector<DataType,Dynamic> z(2*n);                                            // Constraint vector	
			z.head(n) =  xMax;
			z.tail(n) = -xMin;
	
			// Ensure the desired task is feasible after null space projection
			// or the dual method might fail
			
			Vector<DataType,Dynamic> xn = xd - invWAt*Hdecomp.solve(A*xd);              // Null space projection of A matrix
			
			DataType scalingFactor = 1.0;                                               // As it says
			
			for(int i = 0; i < n; i++)
			{
				DataType ratio = 1.0;
				
				     if( xn(i) >= xMax(i) ) ratio = abs(xMax(i) / xn(i));
				else if( xn(i) <= xMin(i) ) ratio = abs(xMin(i) / xn(i));
				
				if(ratio <= scalingFactor) scalingFactor = 0.95*ratio;              // Override if smaller
			}
			
			xn = scalingFactor*xd;                                                      // New desired value for solution
			
			Vector<DataType,Dynamic> f = A*xn - y;                                      // Linear component of QP
			
			this->lastSolution = xn + invWAt*solve(H, f, B*invWAt, z - B*xn, Hdecomp.solve(A*(x0 - xn)));
		
			break;
		}
		case primal:
		{
			unsigned int m = A.rows();
		
			// H = [ 0  A ]
			//     [ A' W ]
			Matrix<DataType,Dynamic,Dynamic> H(m+n,m+n);
			H.block(0,0,m,m).setZero();
			H.block(0,m,m,n) = A;
			H.block(m,0,n,m) = A.transpose();
			H.block(m,m,n,n) = W;
			
			// B = [ 0  I ]
			//     [ 0 -I ]
			Matrix<DataType,Dynamic,Dynamic> B(2*n,m+n);
			B.block(0,0,2*n,m).setZero();
			B.block(0,m,  n,n).setIdentity();
			B.block(n,m,  n,n) = -B.block(0,m,n,n);

			// z = [  xMax ]
			//     [ -xMin ]
			Vector<DataType,Dynamic> z(2*n);
			z.head(n) =  xMax;
			z.tail(n) = -xMin;

			// f = [   -y  ]
			//     [ -W*xd ]
			Vector<DataType,Dynamic> f(m+n);
			f.head(m) = -y;
			f.tail(n) = -W*xd;
			
			// Compute start point with added Lagrange multipliers
			Vector<DataType,Dynamic> startPoint(m+n);
			startPoint.head(m) = (A*invWAt).ldlt().solve(A*xd - y);                     // Initial guess for the Lagrange multipliers
			startPoint.tail(n) = x0;
			
			this->lastSolution = solve(H,f,B,z,startPoint).tail(n);                     // Discard Lagrange multipliers
					
			break;
		}
	}
	
	return this->lastSolution;
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //        Solve a constrained problem min 0.5*(xd - x)'*W*(xd - x) s.t. A*x = y, B*x < z          //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> inline
Vector<DataType,Dynamic>
QPSolver<DataType>::constrained_least_squares(const Vector<DataType, Dynamic>          &xd,
                                              const Matrix<DataType, Dynamic, Dynamic> &W,
                                              const Matrix<DataType, Dynamic, Dynamic> &A,
                                              const Vector<DataType, Dynamic>          &y,
                                              const Matrix<DataType, Dynamic, Dynamic> &B,
                                              const Vector<DataType, Dynamic>          &z,
                                              const Vector<DataType, Dynamic>          &x0)
{
	if(xd.size() = W.rows() or W.rows() != A.cols() or A.cols() != B.cols() or B.cols() != x0.size())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                       "Dimensions of input arguments do not match. "
		                       "The desired vector xd had " + to_string(xd.size()) + " elements, "
		                       "the weighting matrix was " + to_string(W.rows()) + " x " + W.cols() + ", "
		                       "the equality constraint matrix A had " + to_string(A.cols()) + " columns, "
		                       "the inequality constraint matrix B had " + to_string(B.cols()) + " columns,  and "
		                       "the start point x0 had " + to_string(x0.size()) + " elements.");
	}
	else if(W.rows() != W.cols())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                       "Expected the weighting matrix W to be squares, "
		                       "but it was " + to_string(W.rows()) + "x" + to_string(W.cols()) + ".");
	}
	else if(A.rows() != y.size())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                       "Dimensions for the equality constraint do not match. "
		                       "The A matrix had " + to_string(A.rows()) + " rows, and "
		                       "the y vector had " + to_string(y.size()) + " elements.");
	}
	else if(B.rows() != z.size())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
		                       "Dimensions for inequality constraint do not match. "
		                       "The B matrix had " + to_string(B.rows()) + " rows, and "
		                       "the z vector had " + to_string(z.size()) + " elements.");
	}

	// Primal:
	// min 0.5*(xd - x)'*W*(xd - x)
	// subject to: A*x = y
	//             B*x < z
	
	// Lagrangian L = 0.5*(xd - x)'*W*(xd - x) + lambda'*(y - A*xd)
	
	// Dual:
	// min 0.5*lambda'*A*W^-1*A'*lambda - lambda'*(y - A*xd)
	// subject to B*x < z
	
	unsigned int n = x0.size();                                                                 // Number of dimensions in the problem
		
	Matrix<DataType,Dynamic,Dynamic> invWAt = W.ldlt().solve(A.transpose());                    // Makes calcs a little easier
	
	Matrix<DataType,Dynamic,Dynamic> H = A*invWAt;                                              // Hessian matrix for the dual problem
	
	Eigen::LDLT<Matrix<DataType,Dynamic,Dynamic>> Hdecomp; Hdecomp.compute(H);                  // LDL' decomposition                                                         
	
	// Ensure null space projection of the desired solution xd is feasible
	Matrix<DataType,Dynamic,1> xn = xd - invWAt*Hdecomp.solve(A*xd);                            // xd projected on to null space of A matrix
	
	for(int i = 0; i < B.rows(); i++)
	{
		if(z(i) - B.row(i).dot(xn) <= 0)
		{
			throw runtime_error("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                            "Desired point xd violates inequality constraint after projection "
                                            "on to null space of equality constraint matrix A.");
		}
	}
	
	this->lastSolution = xn + invWAt*solve(H, A*xd - y, B*invWAt, z - B*xd, Hdecomp.solve(y - A*xn));
	
	return this->lastSolution;
}

  ///////////////////////////////////////////////////////////////////////////////////////////////////
 //          Solve a problem of the form: min 0.5*x'*H*x + x'*f subject to: B*x <= z              //        
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> inline
Vector<DataType,Dynamic>
QPSolver<DataType>::solve(const Matrix<DataType, Dynamic, Dynamic>  &H,
                          const Vector<DataType, Dynamic>           &f,
                          const Matrix<DataType, Dynamic, Dynamic>  &B,
                          const Vector<DataType, Dynamic>           &z,
                          const Vector<DataType, Dynamic>           &x0)
{
	unsigned int dim = x0.rows();                                                               // Number of dimensions

	unsigned int numConstraints = B.rows();                                                     // As it says
	
	// Check that the inputs are sound
	if(H.rows() != H.cols())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] solve(): Expected a square matrix for the Hessian "
		                       "but it was " + to_string(H.rows()) + "x" + to_string(H.cols()) + ".");
	}
	else if(H.rows() != dim or f.rows() != dim or B.cols() != dim)
	{
		throw invalid_argument("[ERROR] [QP SOLVER] solve(): Dimensions for the decision variable do not match. "
		                       "The Hessian was " + to_string(H.rows()) + "x" + to_string(H.cols()) + ", "
		                       "the f vector was " + to_string(f.rows()) + "x1, "
		                       "the constraint matrix had " + to_string(B.cols()) + " columns, and "
		                       "the start point was " + to_string(x0.rows()) + "x1.");
	}
	else if(B.rows() != z.rows())
	{
		throw invalid_argument("[ERROR] [QP SOLVER] solve(): Dimensions for the constraints do not match. "
		                       "The constraint matrix had " + to_string(B.rows()) + " rows, and "
		                       "the constraint vector had " + to_string(z.rows()) + " rows.");
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
	
	Vector<DataType, Dynamic> g(dim);                                                           // Gradient vector
	Vector<DataType, Dynamic> x = x0;                                                           // Value to be returned
	Vector<DataType, Dynamic> dx(dim);                                                          // Newton step = -I^-1*g
	Matrix<DataType, Dynamic, Dynamic> I;                                                       // Hessian with added barrier function

	vector<DataType> d; d.resize(numConstraints);                                               // Distance to every constraint
	
	// Do some pre-processing
	vector<Vector<DataType, Dynamic>>          bt(numConstraints);                              // Row vectors of B matrix transposed
	vector<Matrix<DataType, Dynamic, Dynamic>> btb(numConstraints);                             // Outer product of row vectors
	for(int j = 0; j < numConstraints; j++)
	{
		bt[j]  = B.row(j).transpose();                                                      // Row vector converted to column vector
		btb[j] = B.row(j).transpose()*B.row(j);                                             // Outer product of row vectors
		
		/*
		if(z(j) - bt[j].dot(x0) < 0)
		{
			throw invalid_argument("[ERROR] [QP SOLVER] solve(): "
			                       "Start point for solver lies outside the constraints. "
			                       "Cannot guarantee a feasible solution.");
		}
		*/		
	}
	
	// Run the interior point algorithm
	for(int i = 0; i < this->maxSteps; i++)
	{
		// (Re)set values for new loop
		I = H;                                                                              // Hessian for log-barrier function
		g.setZero();                                                                        // Gradient vector
		this->numSteps = i+1;                                                               // Save the number of steps taken
		
		// Compute distance to each constraint
		for(int j = 0; j < numConstraints; j++)
		{
			d[j] = z(j) - bt[j].dot(x);                                                 // Distance to the jth constraint
			
			if(d[j] <= 0)                                                               // Constraint violated?!
			{
				d[j] = this->tol;                                                   // Set a small, but non-zero value
				  u /= this->beta;                                                  // Increase the barrier to push the solution away
			}
				  	
			g   += -(u/d[j])*bt[j];                                                     // Add up gradient vector
			I   +=  (u/(d[j]*d[j]))*btb[j];                                             // Add up Hessian
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
				DataType temp = 0.95*d[j]/dotProduct;                               // Compute optimal scalar to avoid constraint violation
				
				if(temp < alpha) alpha = temp;                                      // If smaller, override
			}
		}
		
		this->stepSize = alpha*dx.norm();                                                   // As it says on the label
		
		if( this->stepSize <= this->tol and (z - B*x).dot(z - B*dx) >= 0 ) break;           // If below tolerance AND moving toward constraint, then terminate
		
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
bool QPSolver<DataType>::set_step_scalar(const DataType &scalar)
{
	if(scalar <= 0)
	{
		cerr << "[ERROR] [QP SOLVER] set_step_size(): "
		     << "Input argument was " << to_string(scalar) << " but it must be positive.\n";
	
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
		cerr << "[ERROR] [QP SOLVER] set_barrier_reduction_rate(): "
		        "Input argument was " << to_string(rate) << " but it must be between 0 and 1.\n";
		             
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
		cerr << "[ERROR] [QP SOLVER] set_tolerance(): "
		     << "Input argument was " << to_string(tolerance) << " but it must be positive.\n";
		 
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
		cerr << "[ERROR] [QP SOLVER] set_num_steps(): "
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
		cerr << "[ERROR] [QP SOLVER] set_barrier_scalar(): "
		     << "Input argument was " << to_string(scalar) << " but it must be positive.\n";
		
		return false;
	}
	else
	{
		this->u0 = scalar;
		
		return true;
	}
}
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                           Use the dual method to solve the QP problem                          //
////////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataType>
void QPSolver<DataType>::use_dual()
{
	this->method = dual;
	
	std::cout << "[INFO] [QP SOLVER] Using the dual method to solve.\n";
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                         Use the primal method to solve the QP problem                          //
////////////////////////////////////////////////////////////////////////////////////////////////////
template<class DataType>
void QPSolver<DataType>::use_primal()
{
	this->method = primal;
	
	std::cout << "[INFO] [QP SOLVER] Using the primal method to solve.\n";
}

#endif
