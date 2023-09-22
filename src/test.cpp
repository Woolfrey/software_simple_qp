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

#include <iostream>                                                                                 // std::cout
#include <fstream>                                                                                  // std::ofstream
#include <QPSolver.h>                                                                               // Custom cass
#include <time.h>                                                                                   // clock_t

int main(int argc, char *argv[])
{
	// Variables used in this scope
	float t;
	clock_t timer;
	unsigned int m, n;                                                                          // Dimensions for the various problems.
	Eigen::MatrixXf H, A, comparison;                                                                      
	Eigen::VectorXf f, x, xMin, xMax, y;                                                                       
	QPSolver<float> solver;                                                                     // Create an instance of the class
	srand((unsigned int) time(NULL));                                                           // Seed the random number generator
	
	std::cout << "\n**********************************************************************\n"
	          <<   "*                        A GENERIC QP PROBLEM                        *\n"
	          <<   "**********************************************************************\n" << std::endl;
	          
	m = 5;
	n = 5;
	
	Eigen::MatrixXf temp = Eigen::MatrixXf::Random(n,n);
	H = (temp + temp.transpose())/2;
	f = Eigen::VectorXf::Random(n);
	
	std::cout << "\nThe SimpleQPSolver can be used to find solutions to generic quadratic programming "
	          << "problems of the form:\n"
	          << "\n     min 0.5*x'*H*x + x'*f \n"
	          << "\nwhere:\n"
	          << "   - x (nx1) is the decision variable,\n"
	          << "   - H (nxn) is a positive semi-definite matrix such that H = H', and\n"
	          << "   - f (nx1) is a vector for the linear component of the quadratic equation.\n"
	          << "\nFor this type of problem is is possible to call the class method without creating "
	          << "an object.\n For example, here is the H matrix (Hessian) for a " << m << "x" << n << " system:\n";
	
	std::cout << "\n" << H << std::endl;
	
	std::cout << "\nAnd here is the f vector (transposed):\n";
	
	std::cout << "\n" << f.transpose() << std::endl;
	
	std::cout << "\nThen we can call `Eigen::VectorXf x = QPSolver<float>::solve(H,f);' to get:\n";
	
	timer = clock();
	x = QPSolver<float>::solve(H,f);
	timer = clock() - timer;
	t  = (float)timer/CLOCKS_PER_SEC;
	
	std::cout << "\n" << x.transpose() << std::endl;
	
	std::cout << "\nIt took " << t*1000 << " ms to solve (" << 1/t << " Hz).\n"
	          << "\n(You could also use doubles here with Eigen::VectorXd x = QPSolver<double>::solve(H,f)).\n";
	          
	std::cout << "\n**********************************************************************\n"
	          <<   "*                      UNDERDETERMINED SYSTEMS                       *\n"
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
	          << "\nIt is assumed that m > n so the equations are underdetermined.\n";
	
	std::cout << "\nFor example, here is a " << m << "x" << n << " system where A is:\n";
	
	std::cout << "\n" << A << std::endl;
	
	std::cout << "\nAnd y (transposed) is:\n";
	
	std::cout << "\n" << y.transpose() << std::endl;
	
	std::cout << "\nWe can find an approximate solution by calling "
	          << "'Eigen::VectorXf x = QPSolver<float>least_squares(y,A,W)':\n";
	          
	timer = clock();
	x = QPSolver<float>::least_squares(y,A,Eigen::MatrixXf::Identity(m,m));
	timer = clock() - timer;
	t  = (float)timer/CLOCKS_PER_SEC;	
	
	std::cout << "\n" << x.transpose() << std::endl;
	
	std::cout << "\nThe error ||y - A*x|| is: " << (y - A*x).norm() << ", "
	          <<   "and it took " << t*1000 << " ms to solve (" << 1/t << " Hz).\n";
	          
	std::cout << "\n**********************************************************************\n"
	          <<   "*                      OVERDETERMINED SYSTEMS                        *\n"
	          <<   "**********************************************************************\n" << std::endl;	 
	          
	m = 6;
	n = 7;
	
	A = Eigen::MatrixXf::Random(m,n);
	y = A*Eigen::VectorXf::Random(n);
	
	std::cout << "\nWe can also solve systems where the solution is *over* determined. "
	          <<   "For example, the matrix A here is " << m << "x" << n << ":\n";
		
	std::cout << "\n" << A << std::endl;
	
	std::cout << "\nAnd the y vector (transposed) is:\n";
	
	std::cout << "\n" << y.transpose() << std::endl;
	         
	std::cout << "\nSince m < n there are infinite possible solutions. We can give the QP solver a "
	          <<   "desired value xd for the solution.\n This problem takes the form:\n";
	          
	std::cout << "\n      min 0.5*(xd - x)'*W*(xd - x)\n"
	          <<   "     subject to: A*x = y\n";
	
	std::cout << "\nWhere W (nxn) is a weighting matrix. We can then call "
	          <<   "'Eigen::VectorXf x = QPSolver<float>redundant_least_squares(xd,W,A,y);' to get:\n";
	          
	timer = clock();
	x = QPSolver<float>::redundant_least_squares(Eigen::VectorXf::Random(n),
	                                             Eigen::MatrixXf::Identity(n,n),
	                                             A, y);
	timer = clock() - timer;
	t  = (float)timer/CLOCKS_PER_SEC;	
	
	std::cout << "\n" << x.transpose() << std::endl;

	std::cout << "\nThe error ||y - A*x|| is: " << (y - A*x).norm() << ", "
	          <<   "and it took " << t*1000 << " ms to solve (" << 1/t << " Hz).\n";
	          
	std::cout << "\n**********************************************************************\n"
	          <<   "*                      CONSTRAINED SYSTEMS                           *\n"
	          <<   "**********************************************************************\n" << std::endl;
	          
	m = 9;
	n = 6;
	
	A = Eigen::MatrixXf::Random(m,n);
	x.resize(n); x << -3, 5, -1, 6, 0, -2;
	y = A*x;
	
	xMin = -5*Eigen::VectorXf::Ones(n);
	xMax =  5*Eigen::VectorXf::Ones(n); xMax(1) = 2.7; xMax(3) = 3.6;                           // Manually override the limits
	          
	std::cout << "\nOften there are constrained on the solution. A generic form of this problem is:\n"
	          << "\n     min 0.5*x'*H*x + x'*f\n"
	          <<   "     subject to: B*x <= z\n"
	          << "\nwhere:\n"
	          << "   - B (cxn) is a constraint matrix, and\n"
	          << "   - z (cx1) is a constraint vector.\n"
	          << "\nFor this problem, we need to create an object: 'QPSolver<float> solver;\n"
	          << "Since this uses an iterative method for determining a solution, the solver also requires an initial guess x0 (nx1).\n"
	          << "We would call: `Eigen::VectorXf x = solver.solve(H,f,B,z,x0);'\n";
	          
	std::cout << "\nA more specific example is for a constrained least squares problem of the form:\n"
                  << "\n     min 0.5*(y - A*x)'*W*(y - A*x)\n"
                  << "\n     subject to: xMin <= x <= xMax\n"
                  << "\nwhere xMin (nx1) and xMax(nx1) are lower and upper bounds on the solution.\n"
                  << "\nThe equivalent constraints here are:\n"
                  << "\n    B = [  I ]  z = [  xMax ]\n"
                  <<   "        [ -I ]      [ -xMin ]\n"
                  << "\n(I wrote special function for this case because I am lazy.)\n";     
                  
        std::cout << "\nFor the following system of A:\n";
        
        std::cout << "\n" << A << std::endl;
        
        std::cout << "\n and y (transposed):\n";
        
        std::cout << "\n" << y.transpose() << std::endl;
        
        std::cout << "\nWe can call 'Eigen::VectorXf x = solver.least_squares(y,A,W,xMin,xMax,x0);'\n";
        
	timer = clock();
	x = solver.constrained_least_squares(y,A,Eigen::MatrixXf::Identity(m,m),xMin,xMax,0.5*(xMin+xMax));
	timer = clock() - timer;
	t  = (float)timer/CLOCKS_PER_SEC;
	
	std::cout << "\nHere is xMin, the solution x, and xMax side-by-side:\n";
	comparison.resize(n,3); 
	comparison.col(0) = xMin;
	comparison.col(1) = x;
	comparison.col(2) = xMax;
	std::cout << "\n" << comparison << std::endl;
	
	for(int i = 0; i < x.size(); i++)
	{
		if(x(i) <= xMin(i) or x(i) >= xMax(i))
		{
			std::cerr << "\n[FLAGRANT SYSTEM ERROR] CONSTRAINT VIOLATED!\n";
			break;
		}
	}

	std::cout << "\nThe error ||y - A*x|| is: " << (y - A*x).norm() << ", "
	          <<   "and it took " << t*1000 << " ms to solve (" << 1/t << " Hz).\n";
	          
	std::cout << "\nThere is signicant error because the real solution lies outside the constraints.\n"
	          << "BUT, the QP solver is able to satisfy them!\n";
	          

	std::cout << "\n**********************************************************************\n"
	          <<   "*                CONSTRAINED SYSTEMS (REDUNDANT CASE)                *\n"
	          <<   "**********************************************************************\n" << std::endl;
	          
	m = 12;
	n = 17;
	
	xMin = -5*Eigen::VectorXf::Ones(n);
	xMax =  5*Eigen::VectorXf::Ones(n);
	
	A = Eigen::MatrixXf::Random(m,n);
	Eigen::VectorXf xTrue = Eigen::VectorXf::Random(n);
	y = A*xTrue;
	
	Eigen::VectorXf xd = 10*Eigen::VectorXf::Random(n);
	
	Eigen::VectorXf x0 = 0.5*(xMin + xMax);
	
	std::cout << "\nWe can even solve redundant systems subject to constraint:\n"
	          << "\n      min 0.5*(xd - x)'*W*(xd - x)\n"
	          << "     subject to: A*x = y\n"
	          << "                 B*x < z\n";
	          
	std::cout << "\nAgain, there is a function for least squares problems with lower and upper "
	          << "bounds on the solution:\n"
	          << "\n      B*x <= z  <--->  xMin <= x <= xMax\n";
	
	std::cout << "\nWe would call: 'solver.constrained_least_squares(xd,W,A,y,xMin,xMax,x0)'\n";
	
	std::cout << "\nHere is the solution for a " << m << "x" << n << " system using the dual method:\n";
	
	timer = clock();
	try
	{
		solver.set_tolerance(1e-04);
		
		x = solver.constrained_least_squares(xd,Eigen::MatrixXf::Identity(n,n),A,y,xMin,xMax,x0);

		timer = clock() - timer;
		float t1  = (float)timer/CLOCKS_PER_SEC;
		
		comparison.resize(n,3); 
		comparison.col(0) = xMin;
		comparison.col(1) = x;
		comparison.col(2) = xMax;
		std::cout << "\n" << comparison << std::endl;
		
		for(int i = 0; i < x.size(); i++)
		{
			if(x(i) <= xMin(i) or x(i) >= xMax(i))
			{
				std::cerr << "\n[FLAGRANT SYSTEM ERROR] CONSTRAINT VIOLATED!\n";
				break;
			}
		}
		
		float error1 = (y - A*x).norm();
		
		std::cout << "\nThe error ||y - A*x|| is: " << error1/y.norm() << ", "
			  <<   "and it took " << t1*1000 << " ms to solve (" << 1/t1 << " Hz).\n";
			  
		std::cout << "\nIt took " << solver.num_steps() << " steps to solve.\n\n";

		solver.use_primal();
		
		solver.set_tolerance(1e-03);
		
		std::cout << "\nUsing the primal method we get:\n";

		timer = clock();
		x = solver.constrained_least_squares(xd,Eigen::MatrixXf::Identity(n,n),A,y,xMin,xMax,x0);
		timer = clock() - timer;
		float t2  = (float)timer/CLOCKS_PER_SEC;
		
		float error2 = (y - A*x).norm();
		
		comparison.col(1) = x;
		std::cout << "\n" << comparison << std::endl;
		
		std::cout << "\nThe error ||y - A*x|| is: " << error2/y.norm() << ", "
			  <<   "and it took " << t2*1000 << " ms to solve (" << 1/t2 << " Hz).\n";
			  
		std::cout << "\nThe dual method was " << t2/t1 << " times faster. ";
		
		if(error1 > error2) std::cout << "The primal method was " << error1/error2 << " times more accurate.\n";
		else                std::cout << "The dual method was " << error2/error1 << " times more accurate.\n";
	}
	catch(const std::exception &exception)
	{
		std::cout << exception.what() << std::endl;
	}     
	return 0; 
}
