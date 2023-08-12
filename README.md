# SimpleQPSolver
A small, efficient class for solving convex optimisation problems. A generic quadratic programming (QP) problem is of the form:
```math
\begin{align}
	\min_{\mathbf{x}} ~ \frac{1}{2}\mathbf{x^\mathrm{T}Hx + x^\mathrm{T}f} \\
	\text{subject to: } \mathbf{Bx \le z}
\end{align}
```
where:
- $\mathbf{x}\in\mathbb{R}^\mathrm{n}$ is the decision variable,
- $\mathbf{H = H^\mathrm{T}}\in\mathbb{R}^\mathrm{n\times n}$ is a weighting matrix,
- $\mathbf{f}\in\mathbb{R}^\mathrm{n}$ is the linear component of the quadratic equation,
- $\mathbf{B}\in\mathbb{R}^\mathrm{c\times n}$ is a constraint matrix, and
- $\mathbf{z}\in\mathbb{R}^\mathrm{c}$ is a constraint vector.

There are also several functions for handling linear least squares problems with equality and inequality constraints.

`SimpleQPSolver` is **free to use** under the GNU General Public License v3.0. If you find this software useful, [citing it](#citing-this-repository) would be appreciated.

**Jump To:**
- [Installation Instructions](#installation-instructions)
	- [Installing Eigen](#installing-eigen)
 	- [Installing SimpleQPSolver](#installing-simpleqpsolver)
- [Using the QP Solver](#using-the-qp-solver)
  	- [A Generic QP Problem](#a-generic-qp-problem)
   	- [Linear Least Squares](#linear-least-squares-linear-regression)
   	- [Least Squares with Equality Constraints](#least-squares-with-equality-constraints-over-determined-systems)
   	 - [Optimisation with Inequality Constraints](#optimisation-with-inequality-constraints)
   	- [Options for the Interior Point Algorithm](#options-for-the-interior-point-algorithm)
- [Citing this Repository](#citing-this-repository)

## Installation Instructions

### Installing Eigen

The `SimpleQPSolver` requires the `Eigen` libraries. If you're using Linux you can install it from the command line:

```
sudo apt install libeigen3-dev
```
Otherwise, you can head over to the [main page](https://eigen.tuxfamily.org/index.php?title=Main_Page) to see how you can install it.

### Installing SimpleQPSolver

SimpleQPSolver is a template class contained in a single header file. There is no need to clone this repository and build any packages (unless you want to, of course). All you need to do is download the header file `QPSolver.h` and include it in your package:

```
SimpleQPSolver/include/QPSolver.h
```
That is all!

_If you want to build the package for some reason..._ there is a simple `test.cpp` file you can run that demonstrates the use of the `QPSolver` class. First navigate to your working directory:

```
cd ~/MyWorkspace
```
Then clone the repository:
```
git clone https://github.com/Woolfrey/SimpleQPSolver.git
```
Create a build folder and navigate to it:
```
cd SimpleQPSolver/ && mkdir build && cd build
```
Generate the build tools using:
```
cmake ../
```
Then build the package with:
```
make
```
You can then run
```
./test
```
which prints information about the use of different class methods, as well as the accuracy and speed of solutions.

:arrow_backward: [Go Back.](#simpleqpsolver)

## Using the QP Solver

You can use `float` or `double` with this class. Input arguments for the `Eigen` classes must match:
- `Eigen::MatrixXf` and `Eigen::VectorXf` when using `QPSolver<float>`, or
- `Eigen::MatrixXd` and `Eigen::VectorXd` when using `QPSolver<double>`.

For problems _without_ inequality constraints, you can call `static` methods without creating a `QPSolver` object:
- `QPSolver<float>::solve(H,f)`
- `QPSolver<double>::least_squares(y,A,W)`
- `QPSolver<float>::redundant_least_squares(xd,W,A,y)`

Methods with inequality constraints, e.g. $\mathbf{Bx \le z}$, or $\mathbf{x_\mathrm{min} \le x \le x_\mathrm{max}}$ use an interior point algorithm, and thus require you to create an instance of the class.

Some examples are below.

### A Generic QP Problem

```math
\min_{\mathbf{x}} \frac{1}{2}\mathbf{x^\mathrm{T} Hx + x^\mathrm{T}f}
```
Assuming $\mathbf{H}$ and $\mathbf{f}$ are given, then you can call:
```
Eigen::VectorXd x = QPSolver<double>::solve(H,f);
```
You can actually solve this yourself by calling something like `x = H.ldlt().solve(-f)`, but I put this function in for completeness.

### Linear Least Squares (Linear Regression)

```math
\min_{\mathbf{x}} \frac{1}{2}\|\mathbf{y - Ax}\|^2_\mathbf{W} = \frac{1}{2}\mathbf{\left(y - Ax\right)^\mathrm{T} W\left(y - Ax\right)}
```
where:
- $\mathbf{y}\in\mathbb{R}^\mathrm{m}$,
- $\mathbf{A}\in\mathbb{R}^\mathrm{m\times n}$,
- $\mathbf{x}\in\mathbb{R}^\mathrm{n}$,
- $\mathbf{W}\in\mathbb{R}^\mathrm{m\times m}$ weights the error, and
- m $\ge$ n (more outputs than inputs)

Call:
```
Eigen::VectorXf x = QPSolver<float>::least_squares(y,A,W);
```
### Least Squares with Equality Constraints (Over-determined Systems)
```math
\begin{align}
\min_{\mathbf{x}} \frac{1}{2}\|\mathbf{x_\mathrm{d} - x}\|^2_\mathrm{W} = \frac{1}{2}\mathbf{\left(x_\mathrm{d} - x\right)^\mathrm{T} W\left(x_\mathrm{d} - x\right)} \\
\text{subject to: } \mathbf{Ax} = \mathbf{y}
\end{align}
```
where:
- $\mathbf{y}\in\mathbb{R}^\mathrm{m}$,
- $\mathbf{A}\in\mathbb{R}^\mathrm{m\times n}$,
- $\mathbf{x}\in\mathbb{R}^\mathrm{n}$,
- $\mathbf{x}_\mathrm{d}\in\mathbb{R}^\mathrm{n}$ is a _desired_ value for $\mathbf{x}$,
- $\mathbf{W}\in\mathbb{R}^\mathrm{n\times n}$ weights the solution,
- m < n (more inputs than outputs)

In this situation the $\mathbf{x}$ vector has more dimensions than the $\mathbf{y}$ vector. Therefore, infinite solutions exist. You can give a desired value $\mathbf{x}_\mathrm{d}$ that the solver will try to achieve whilst satisfying the relationship $\mathbf{Ax = y}$. Call:
```
Eigen::VectorXd x = QPSolver<double>::redundant_least_squares(xd,W,A,y);
```

### Optimisation with Inequality Constraints
```math
\begin{align}
	\min_{\mathbf{x}} \frac{1}{2}\mathbf{x^\mathrm{T}Hx + x^\mathrm{T}f} \\
	\text{subject to: } \mathbf{Bx} \le \mathbf{z}
\end{align}
```
For problems like this with inequality constraints, the solver uses an interior point algorithm. This uses Newton's method to iteratively minimize the objective function whilst satisfying the inequality. It therefore requires a _start point_ or _initial guess_:

First create an object, then call the function:
```
QPSolver<double> solver;
Eigen::VectorXd x = solver.solve(H,f,B,z,x0);
```
where `x0` is the start point argument. It can have a huge influence on the result, so it is good to give the solver an approximate solution if you can.

If you're repeatedly solving a QP problem you can get the last solution and use that as the input:
```
Eigen::VectorXd x0 = solver.last_solution();
```
There are several functions conveniently written for least squares type problems:

**Linear least squares with upper and lower bounds:**

```math
\begin{align}
	\min_{\mathbf{x}} \frac{1}{2}\mathbf{\left(y - Ax\right)^\mathrm{T} W\left(y - Ax\right)} \\
	\text{subject to: } \mathbf{x_\mathrm{min} \le x \le x_\mathrm{max}}
\end{align}
```
use:
```
Eigen::VectorXf x = solver.constrained_least_squares(y,A,W,xMin,xMax,x0);
```
**Redundant least squares with upper and lower bounds:**

```math
\begin{align}
	\min_{\mathbf{x}} \frac{1}{2}\mathbf{\left(x_\mathrm{d} - x\right)^\mathrm{T} W\left(x_\mathrm{d} - x\right)} \\
	\text{subject to: } \mathbf{Ax} = \mathbf{y} \\
            \mathbf{x_\mathrm{min} \le x \le x_\mathrm{max}}
\end{align}
```
use:
```
Eigen::VectorXd x = solver.constrained_least_squares(xd,W,A,y,xMin,xMax,x0);
```
**Redundant least squares with inequality constraints:**

```math
\begin{align}
	\min_{\mathbf{x}} \frac{1}{2}\mathbf{\left(x_\mathrm{d} - x\right)^\mathrm{T} W\left(x_\mathrm{d} - x\right)} \\
	\text{subject to: } \mathbf{Ax} &= \mathbf{y} \\
                          \mathbf{Bx} &\le \mathbf{z}
\end{align}
```
use:
```
Eigen::VectorXd x = solver.constrained_least_squares(xd,W,A,y,B,z,x0);
```
:warning: When using the dual method for this problem, the desired value $\mathbf{x}_{\mathrm{d}}$ must satisfy constraints when projected on to the null space of $\mathbf{A}$.

:arrow_backward: [Go Back.](#simpleqpsolver)

### Options for the Interior Point Algorithm

There are several parameters that can be set when solving for inequality constaints:
- `use_dual()`: This is the default method for _redundant, constrained least squares_ methods. The solver is sensitive to the start point `x0`.
- `use_primal()`: This an alternative method for _redundant, constrained least squares_. It is slower than the dual method, but a little bit more robust.
- `set_step_size(const DataType &size)`: The parameter $\alpha$ scales the step size $\alpha\cdot\Delta\mathbf{x}$. Default value is 1; a smaller size will mean slower convergence.
- `set_tolerance(const DataType &tolerance)`: The algorithm terminates when $\|\alpha\cdot\Delta\mathbf{x}\|$ is less than this value. A smaller value means a more accurate solution, but slower solution time.
- `set_num_steps(const unsigned int &numer)`: The algorithm terminates if this number of steps is reached. A higher value means a more accurate solution, but it might take longer to solve.
- `set_barrier_scalar(const DataType &scalar)`: The inequality constraints are converted to a log-barrier function. This parameter determines how steep the slope of the barrier is. A smaller value means a faster solution, but you may prematurely run in to the constraint and terminate the algorithm.
- `set_barrier_reduction_rate(const DataType &rate)`: Every loop the barrier slope is decreased. This determines how fast it decreases. A smaller value means the barrier effect will shrink quickly. This will make the algorithm faster, but then it may not find a solution if it hits the constraints prematurely.

:arrow_backward: [Go Back.](#simpleqpsolver)

## Citing this Repository
If you use `SimpleQPSolver` and find it useful, I'd appreciate it if you could cite me. Here is a `BibTeX` format:
```
@software{Woolfrey_SimpleQPSolver_2023,
     author  = {Woolfrey, Jon},
     month   = aug,
     title   = {{SimpleQPSolver}},
     url     = {https://github.com/Woolfrey/SimpleQPSolver},
     version = {1.0.0},
     year    = {2023}
}
```
Here's the automatically generated APA format:
```
Woolfrey, J. (2023). SimpleQPSolver (Version 1.0.0) [Computer software]. https://github.com/Woolfrey/SimpleQPSolver
```
Alternatively, click on `Cite this repository` on the top-right corner of this page.

:arrow_backward: [Go Back.](#simpleqpsolver)
