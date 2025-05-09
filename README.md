# :cyclone: Simple QP Solver
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

There are also several functions for handling linear least squares problems with equality and inequality constraints. It is fast, and effective. It has been used to control the upper body of a humanoid robot subject to grasp constraints and joint limits.

<p align="center">
	<img src="https://github.com/Woolfrey/software_simple_qp/blob/master/assets/bimanual_manipulation.gif" width=200 height="auto"/>
</p>

>[!NOTE]
> Simple QP Solver is **free to use** under the GNU General Public License v3.0. If you find this software useful, [citing it](#citing-this-repository) would be appreciated.

#### :compass: Navigation
- [Installation Instructions](#floppy_disk-installation-instructions)
	- [Installing Eigen](#installing-eigen)
 	- [Installing SimpleQPSolver](#installing-simpleqpsolver)
	- [Automatic Download in Another Package](#automatic-download-in-another-package)
- [Using the QP Solver](#wrench-using-the-qp-solver)
  	- [A Generic QP Problem](#a-generic-qp-problem)
   	- [Linear Least Squares](#linear-least-squares-linear-regression)
   	- [Least Squares with Equality Constraints](#least-squares-with-equality-constraints-over-determined-systems)
   	 - [Optimisation with Inequality Constraints](#optimisation-with-inequality-constraints)
   	- [Options for the Interior Point Algorithm](#options-for-the-interior-point-algorithm)
   	- [Running the Test File](#running-the-test-file)
- [Contributing](#handshake-contributing)
- [Citing this Repository](#bookmark_tabs-citing-this-repository)
- [License](#scroll-license)

## :floppy_disk: Installation Instructions

### Installing Eigen:

> [!NOTE]
> If you're using Ubuntu 20.04, you must install Eigen 3.4 manually. Later versions (22.04 +) will automatically install 3.4 using the `apt install` command.

The `SimpleQPSolver` requires the `Eigen` v3.4 libraries. If you're using Linux you can install it from the command line:

```
sudo apt install libeigen3-dev
```
Otherwise, you can head over to the [main page](https://eigen.tuxfamily.org/index.php?title=Main_Page) to see how you can install it.

### Installing SimpleQPSolver:

SimpleQPSolver is a template class contained in a single header file. There is no need to clone this repository and build any packages (unless you want to, of course). All you need to do is download the header file `QPSolver.h` and include it in your package:

```
software_simple_qp/include/QPSolver.h
```
That is all!

>[!NOTE]
> If you want to build the package for some reason, there is a simple `test.cpp` file [you can run](#running-the-test-file) that demonstrates the use of the `QPSolver` class.

### Automatic Download in Another Package:

It is possible to automatically download the `QPSolver.h` header file as part of another package. In your `CMakeLists.txt` file you can add something like:
```
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/your_package/include/QPSolver.h")
	file(DOWNLOAD
	     https://raw.githubusercontent.com/Woolfrey/software_simple_qp/master/include/QPSolver.h
	     ${CMAKE_SOURCE_DIR}/your_package/include/QPSolver.h)
endif()
```
[:top: Back to top.](#cyclone-simple-qp-solver)

## :wrench: Using the QP Solver

You can use `float` or `double` with this class. Input arguments for the `Eigen` classes must match:
- `Eigen::MatrixXf` and `Eigen::VectorXf` when using `QPSolver<float>`, or
- `Eigen::MatrixXd` and `Eigen::VectorXd` when using `QPSolver<double>`.

For problems _without_ inequality constraints, you can call `static` methods without creating a `QPSolver` object:
- `QPSolver<float>::solve(H,f)`
- `QPSolver<double>::least_squares(y,A,W)`
- `QPSolver<float>::redundant_least_squares(xd,W,A,y)`

Methods with inequality constraints, e.g. $\mathbf{Bx \le z}$, or $\mathbf{x_\mathrm{min} \le x \le x_\mathrm{max}}$ use an interior point algorithm, and thus require you to create an instance of the class.

Some examples are below.

### A Generic QP Problem:

```math
\min_{\mathbf{x}} \frac{1}{2}\mathbf{x^\mathrm{T} Hx + x^\mathrm{T}f}
```
Assuming $\mathbf{H}$ and $\mathbf{f}$ are given, then you can call:
```
Eigen::VectorXd x = QPSolver<double>::solve(H,f);
```
>[!TIP]
> You can actually solve this yourself by calling something like `x = H.ldlt().solve(-f)`, but I put this function in for completeness.

### Linear Least Squares (Linear Regression):

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
### Least Squares with Equality Constraints (Over-determined Systems):
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

### Optimisation with Inequality Constraints:
```math
\begin{align}
	\min_{\mathbf{x}} \frac{1}{2}\mathbf{x^\mathrm{T}Hx + x^\mathrm{T}f} \\
	\text{subject to: } \mathbf{Bx} \le \mathbf{z}
\end{align}
```
>[!NOTE]
> For problems like this with inequality constraints, the solver uses an interior point algorithm. This uses Newton's method to iteratively minimize the objective function whilst satisfying the inequality. It therefore requires a _start point_ or _initial guess_.

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

#### Linear least squares with upper and lower bounds:

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

#### Redundant least squares with upper and lower bounds:

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

#### Redundant least squares with inequality constraints:

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
>[!WARNING]
> When using this particular function the desired value $\mathbf{x}_{\mathrm{d}}$ must satisfy constraints when projected on to the null space of $\mathbf{A}$.

[:top: Back to top.](#cyclone-simple-qp-solver)

### Options for the Interior Point Algorithm

There are several parameters that can be set when solving for inequality constaints:
- `set_step_size(const DataType &size)`: The parameter $\alpha$ scales the step size $\alpha\cdot\Delta\mathbf{x}$. Default value is 1; a smaller size will mean slower convergence.
- `set_tolerance(const DataType &tolerance)`: The algorithm terminates when $\|\alpha\cdot\Delta\mathbf{x}\|$ is less than this value. A smaller value means a more accurate solution, but slower solution time.
- `set_num_steps(const unsigned int &numer)`: The algorithm terminates if this number of steps is reached. A higher value means a more accurate solution, but it might take longer to solve.
- `set_barrier_scalar(const DataType &scalar)`: The inequality constraints are converted to a log-barrier function. This parameter determines how steep the slope of the barrier is. A smaller value means a faster solution, but you may prematurely run in to the constraint and terminate the algorithm.
- `set_barrier_reduction_rate(const DataType &rate)`: Every loop the barrier slope is decreased. This determines how fast it decreases. A smaller value means the barrier effect will shrink quickly. This will make the algorithm faster, but then it may not find a solution if it hits the constraints prematurely.

 [:top: Back to top.](#cyclone-simple-qp-solver)

### Running the Test File:

First navigate to your working directory:
```
cd ~/MyWorkspace
```
Then clone the repository:
```
git clone https://github.com/Woolfrey/software_simple_qp.git
```
Create a build folder and navigate to it:
```
cd software_simple_qp/ && mkdir build && cd build
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

<img src="https://github.com/Woolfrey/software_simple_qp/assets/62581255/8ecb93a6-e45b-4ff9-aebf-1f2f27c62f25" width="600" height="auto">

[:top: Back to top.](#cyclone-simple-qp-solver)

## :handshake: Contributing

Contributions to this project are always welcome! Feel free to:
1. Fork the repository,
2. Implement your changes, then
3. Submit a pull request.

If you're looking for ideas, you can always check the [Issues tab](https://github.com/Woolfrey/software_simple_qp/issues) for those with :raising_hand: [OPEN]. These are things I'd like to implement myself, but don't have time. It'd be much appreciated, and you'll be listed as a contributor!

## :bookmark_tabs: Citing this Repository

If you use `SimpleQPSolver` and find it useful, I'd appreciate it if you could cite me. Click on `Cite this repository` under the **About** section at the top-right corner of this page :arrow_upper_right:

Here is the `BibTeX` reference:
```
@software{woolfrey_simple_qp_2023,
     author  = {Woolfrey, Jon},
     month   = aug,
     title   = {{S}imple {QP} {S}olver},
     url     = {https://github.com/Woolfrey/software_simple_qp},
     version = {1.0.0},
     year    = {2023}
}
```
Here's the automatically generated APA format:
```
Woolfrey, J. (2023). Simple QP Solver (Version 1.0.0) [Computer software]. https://github.com/Woolfrey/software_simple_qp
```

[:top: Back to top.](#cyclone-simple-qp-solver)

## :scroll: License

This software package is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://choosealicense.com/licenses/gpl-3.0/). You are free to use, modify, and distribute this package, provided that any modified versions also comply with the GPL-3.0 license. All modified versions must make the source code available and be licensed under GPL-3.0. The license also ensures that the software remains free and prohibits the use of proprietary restrictions such as Digital Rights Management (DRM) and patent claims. For more details, please refer to the [full license text](LICENSE).

[:top: Back to top.](#cyclone-simple-qp-solver)
