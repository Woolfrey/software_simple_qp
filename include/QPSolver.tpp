/**
 * @file  : QPSolver.tpp
 * @author: Jon Woolfrey
 * @date  : August 2023
 * @brief : Source files for the QP solver class.
 *
 * This software is publicly available under the GNU General Public License V3.0. You are free to
 * use it and modify it as you see fit. If you find it useful, please acknowledge it.
 *
 * @see https://github.com/Woolfrey/software_simple_qp
 */

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //                                            Constructor                                         //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType>
QPSolver<DataType>::QPSolver(const SolverOptions<DataType> &options)
{
    barrierReductionRate = options.barrierReductionRate;
    initialBarrierScalar = options.initialBarrierScalar;
    tol = options.tolerance;
    maxSteps = options.maxSteps;
}
          
  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //              Solve a standard QP problem of the form min 0.5*x'*H*x + x'*f                     //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType,Eigen::Dynamic>
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
     else      return H.ldlt().solve(-f);                                                          // Too easy lol ᕙ(▀̿̿ĺ̯̿̿▀̿ ̿) ᕗ
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //           Solve an unconstrained least squares problem: min 0.5(y-A*x)'*W*(y-A*x)              //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType,Eigen::Dynamic>
QPSolver<DataType>::least_squares(const Eigen::Vector<DataType, Eigen::Dynamic> &y,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &A,
                                  const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &W)
{
     if(A.rows() < A.cols())                                                                        // Redundant system, use other function
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] least_squares(): "
                                      "The A matrix has more rows than columns ("
                                      + std::to_string(A.rows()) + "x" + std::to_string(A.cols()) + "). "
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
     else     return (A.transpose()*W*A).ldlt().solve(A.transpose()*W*y);                               // x = (A'*W*A)^-1*A'*W*y
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //    Solve least squares problem of the form min 0.5*(xd - x)'*W*(xd - x) subject to: A*x = y    //
////////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType,Eigen::Dynamic>
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
             Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> invWA = W.ldlt().solve(A.transpose()); // Makes calcs a little easier

          return xd + invWA*(A*invWA).ldlt().solve(y - A*xd);                                       // xd + W^-1*A'*(A*W^-1*A')^-1*(y-A*xd)
     }
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //      Solve a constrained problem: min 0.5*(y - A*x)'*W*(y - A*x) s.t. xMin <= x <= xMax        //
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
     if(y.size() != A.rows() or A.rows() != W.rows())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                      "Dimensions of the linear equation do not match. "
                                      "The y vector had " + std::to_string(y.size()) + " elements, "
                                      "the A matrix had " + std::to_string(A.rows()) + " rows, and "
                                      "the weighting matrix W had " + std::to_string(W.rows()) + " rows.");
     }
     else if(W.rows() != W.cols())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                      "Expected the weighting matrix W to be square, but it was "
                                      + std::to_string(W.rows()) + "x" + std::to_string(W.cols()) + ".");
     }
     else if(A.cols() != xMin.size() or xMin.size() != xMax.size() or xMax.size() != x0.size())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                      "Dimensions for decision variable do not match. "
                                      "The A matrix had " + std::to_string(A.cols()) + " columns, "
                                      "the xMin argument had " + std::to_string(xMin.size()) + " elements, "
                                      "the xMax argument had " + std::to_string(xMax.size()) + " elements, and "
                                      "the start point x0 had " + std::to_string(x0.size()) + " elements.");
     }
     
     // Convert to standard form and solve with the interior point algorithm
     
     unsigned int n = x0.size();
     
     // B = [  I ] < z = [  xMax ]
     //     [ -I ]       [ -xMin ]
     
     Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> B(2*n,n);
     B.block(0,0,n,n).setIdentity();
     B.block(n,0,n,n) = -Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic>::Identity(n,n);
     
     Eigen::Vector<DataType,Eigen::Dynamic> z(2*n);
     z.head(n) =  xMax;
     z.tail(n) = -xMin;
     
     Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> AtW = A.transpose()*W;                   // Makes calcs a tiny bit faster

     return solve(AtW*A, -AtW*y, B, z, x0);                                                         // Send to interior point algorithm and solve
}

  ////////////////////////////////////////////////////////////////////////////////////////////////////
 //    Solve a constrained problem min 0.5*(xd - x)'*W*(xd - x) s.t. A*x = y, xMin <= x <= xMax    //
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
     if(xMin.size() != xMax.size())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                      "Dimensions of inequality constraints do not match. "
                                      "The xMin argument had " + std::to_string(xMin.size()) + " elements, and "
                                      "the xMax argument had " + std::to_string(xMax.size()) + " elements.");
     }
                                 
     // Convert constraints to standard form and pass on to generic function
     
     unsigned int n = xMin.size();
     
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
 //        Solve a constrained problem min 0.5*(xd - x)'*W*(xd - x) s.t. A*x = y, B*x < z          //
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
     if(xd.size() != W.rows() or W.rows() != A.cols() or A.cols() != B.cols() or B.cols() != x0.size())
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
     
     if(this->method == primal)
     {     
          unsigned int c = B.rows();                                                                // Number of inequality constraints
          unsigned int m = A.rows();                                                                // Number of equality constraints
          unsigned int n = A.cols();                                                                // Decision variable
          
          // H = [  0  -A ]
          //     [ -A'  W ]
          Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> H(m+n,m+n);
          H.block(0,0,m,m).setZero();
          H.block(0,m,m,n) = -A;
          H.block(m,0,n,m) = -A.transpose();
          H.block(m,m,n,n) = W;
          
          // f = [    y  ]
          //     [ -W*xd ]
          Eigen::Vector<DataType,Eigen::Dynamic> f(m+n);
          f.head(m) = y;
          f.tail(n) = -W*xd;
          
          // new_x0 = [ lambda ]
          //          [   x0   ]
          Eigen::Vector<DataType,Eigen::Dynamic> new_x0(m+n);
          new_x0.head(m) = (A*W.ldlt().solve(A.transpose())).ldlt().solve(y - A*xd);                // Initial guess for Lagrange multipliers
          new_x0.tail(n) = x0;
          
          // newB = [ 0 B ]
          Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> newB(c,m+n);
          newB.block(0,0,c,m).setZero();
          newB.block(0,m,c,n) = B;
          
          this->lastSolution = solve(H,f,newB,z,new_x0).tail(n);                                    // We don't need the Lagrange multipliers
          
          return this->lastSolution;                                                                // Return decision variable x
     }
     else if(this->method == dual) // STILL UNDERGOING TESTING; FAST BUT NOT 100% RELIABLE
     {
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
          
          this->lastSolution = xr + alpha*xn;
          
          return this->lastSolution;
     }
     else
     {
          throw std::runtime_error("[ERROR] [QP SOLVER] constrained_least_squares(): "
                                   "Method for solving redundant least squares was neither 'dual' "
                                   "nor 'primal'. How did that happen?");
     }
}

  ///////////////////////////////////////////////////////////////////////////////////////////////////
 //          Solve a problem of the form: min 0.5*x'*H*x + x'*f subject to: B*x <= z              //        
///////////////////////////////////////////////////////////////////////////////////////////////////
template <class DataType> 
Eigen::Vector<DataType,Eigen::Dynamic>
QPSolver<DataType>::solve(const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &H,
                          const Eigen::Vector<DataType,Eigen::Dynamic> &f,
                          const Eigen::Matrix<DataType, Eigen::Dynamic, Eigen::Dynamic> &B,
                          const Eigen::Vector<DataType,Eigen::Dynamic> &z,
                          const Eigen::Vector<DataType,Eigen::Dynamic> &x0)
{
     // Ensure arguments are sound
     if(H.rows() != H.cols())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                      "Expected the Hessian matrix H to be square but it was " 
                                      + std::to_string(H.rows()) + "x" + std::to_string(H.cols()) + ".");
     }
     else if(H.cols() != f.size() or f.size() != B.cols() or B.cols() != x0.size())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                      "Dimensions of arguments for decision variable do not match. "
                                      "The Hessian matrix had " + std::to_string(H.cols()) + " rows/columns, "
                                      "the vector f had " + std::to_string(f.size()) + " elements, "
                                      "the inequality constraint matrix B had " + std::to_string(B.cols()) + " columns, and "
                                      "the start point x0 had " + std::to_string(x0.size()) + " elements.");
     }
     else if(B.rows() != z.size())
     {
          throw std::invalid_argument("[ERROR] [QP SOLVER] solve(): "
                                      "Dimensions for inequality constraint do not match. "
                                      "The inequality constraint matrix B had " + std::to_string(B.rows()) + " rows, and "
                                      "the inequality constraint vector z had " + std::to_string(z.size()) + " elements.");
     }
     
     // h = 0.5*x'*H*x + x'*f - sum log(d_i),   d_i = z_i - b_i'*x
     // g = H*x + f + sum (1/d_i)*b_i
     // I = H + sum (1/d_i^2)*b_i*b_i'
     
     // Variables used in this scope
     DataType u = this->initialBarrierScalar;                                                       // As it says
     unsigned int dim = x0.size();                                                                  // Dimensions of the decision variaBL   e
     unsigned int numConstraints = z.size();                                                        // As it says
     Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic> I(dim,dim);                              // Hessian matrix
     Eigen::Vector<DataType,Eigen::Dynamic> g(dim);                                                 // Gradient vector
     std::vector<DataType> d(numConstraints);                                                       // Distance to every constraint
     std::vector<Eigen::Vector<DataType,Eigen::Dynamic>> b(numConstraints);                         // Row vectors of constraint matrix (transposed)
     std::vector<Eigen::Matrix<DataType,Eigen::Dynamic,Eigen::Dynamic>> bbt(numConstraints);        // Outer product of row vectors
     Eigen::Vector<DataType,Eigen::Dynamic> x(dim);                                                 // We want to solve for this
     
     // Do some pre-processing
     bool initialConstraintViolated = false;
     for(int j = 0; j < numConstraints; j++)
     {
          b[j]   = B.row(j).transpose();                                                            // Transpose the row vector
          bbt[j] = b[j]*b[j].transpose();                                                           // Compute the outer product

          d[j] = z(j) - b[j].dot(x0);                                                               // Distance to constraint
          
          if(d[j] <= 0) initialConstraintViolated = true;                                           // Flag
     }
     
     // Set the start point
     if(initialConstraintViolated)
     {
          Eigen::Vector<DataType,Eigen::Dynamic> dz
          = 1e-03*Eigen::Vector<DataType,Eigen::Dynamic>::Ones(numConstraints);                     // Add a tiny offset so we're not exactly on the constraint      
          
               if(numConstraints > dim) x = (B.transpose()*B).ldlt().solve(B.transpose()*(z - dz)); // Underdetermined system
          else if(numConstraints < dim) x =  B.transpose()*(B*B.transpose()).ldlt().solve(z - dz);  // Overdetermined system
          else                          x =  B.partialPivLu().solve(z - dz);                        // Exact solution
     }
     else x = x0;                                                                                   // Given start point
     
     // Run the interior point algorithm
     for(int i = 0; i < this->maxSteps; i++)
     {
          this->numSteps = i+1;                                                                     // Increment the counter
          
          // (Re)set values for new loop
          g = H*x + f;                                                                              // Gradient vector
          I = H;                                                                                    // Hessian matrix
          
          // Compute distance to every constraint
          for(int j = 0; j < numConstraints; j++)
          {
               d[j] = z(j) - b[j].dot(x);                                                           // Distance to constraint
               
               if(i == 0 and d[j] <= 0)
               {
                    throw std::runtime_error("[ERROR] [QP SOLVER] solve(): "
                                             "Unable to find a solution that satisfies constraints.");
               }
               
               if(d[j] <= 0) d[j] = 1e-03;                                                          // Constraint violated; set a small, but non-zero distance
           
               g += (u/d[j])*b[j];                                                                  // Add up gradient
               I += (u/(d[j]*d[j]))*bbt[j];                                                         // Add up Hessian
          }

          Eigen::Vector<DataType,Eigen::Dynamic> dx = I.ldlt().solve(-g);                           // Compute Newton step
          
          // Compute scalar for step size so that constraint is not violated on next step
          DataType alpha = 1.0;
          for(int j = 0; j < numConstraints; j++)
          {
               DataType dotProd = b[j].dot(dx);                                                     // Makes calcs a little easier
               
               if(d[j] - dotProd <= 0) alpha = min(alpha,0.9*d[j]/dotProd);                         // Shrink scalar if constraint violated
          }
          
          dx *= alpha;                                                                              // Scale the step
          
          this->stepSize = dx.norm();                                                               // Magnitude of the step size
          
          if(this->stepSize <= this->tol) break;                                                    // If smaller than tolerance, break
          
          // Increment values for next loop
          x += dx;                                                                                  // Increment state
          u *= this->barrierReductionRate;                                                          // Reduce barrier
     }
     
     this->lastSolution = x;                                                                        // Save the value
     
     return x;
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
                       "but it must be between 0 and 1." << std::endl;
                       
          return false;
     }
     else
     {
          this->barrierReductionRate = rate;
          
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
                    << "but it must be positive." << std::endl;
            
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
bool QPSolver<DataType>::set_max_steps(const unsigned int &number)
{
     if(number == 0)
     {
          std::cerr << "[ERROR] [QP SOLVER] set_num_steps(): "
                    << "Input argument was 0 but it must be greater than zero." << std::endl;
          
          return false;
     }
     else
     {
          this->maxSteps = number;
          
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
                    << "but it must be positive." << std::endl;
          
          return false;
     }
     else
     {
          this->initialBarrierScalar = scalar;
          
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
