# :cyclone: Simple QP Solver
A small, efficient class for solving convex optimisation problems. A generic quadratic programming (QP) problem is of the form:
```math
\begin{align}
	\min_{\mathbf{x}} ~ \frac{1}{2}\mathbf{x^\mathrm{T}Hx + x^\mathrm{T}f} \\
	\text{subject to: } \mathbf{Ax} &= \mathbf{y} \\
	                    \mathbf{Bx} &\le \mathbf{z}
\end{align}
```
where:
- $\mathbf{x}\in\mathbb{R}^\mathrm{n}$ is the decision variable,
- $\mathbf{H = H^\mathrm{T}}\in\mathbb{R}^\mathrm{n\times n}$ is a positive (semi)definite weighting matrix,
- $\mathbf{f}\in\mathbb{R}^\mathrm{n}$ is the linear component of the quadratic equation,
- $\mathbf{A}\in\mathbb{R}^\mathrm{m\times n}$ is an equality constraint matrix,
- $\mathbf{y}\in\mathbb{R}^\mathrm{m}$ is an equality constraint vector,
- $\mathbf{B}\in\mathbb{R}^\mathrm{c\times n}$ is an inequality constraint matrix, and
- $\mathbf{z}\in\mathbb{R}^\mathrm{c}$ is an inequality constraint vector.

There are also several functions for handling linear least squares problems with equality and inequality constraints. Inequality constrained problems can be solved with either an **interior point** or an **active set** algorithm. It is fast, and effective. It has been used to control the upper body of a humanoid robot subject to grasp constraints and joint limits.

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
  	- [Solver Options](#solver-options)
  	- [Solver Results](#solver-results)
  	- [A Generic QP Problem](#a-generic-qp-problem)
   	- [Linear Least Squares](#linear-least-squares-linear-regression)
   	- [Least Squares with Equality Constraints](#least-squares-with-equality-constraints-under-determined-systems)
   	- [Optimisation with Inequality Constraints](#optimisation-with-inequality-constraints)
   	- [Running the Test File](#running-the-test-file)
- [Applications](#applications)
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

>[!IMPORTANT]
> `QPSolver.h` requires C++17 or later. In your `CMakeLists.txt` you can set:
> ```
> set(CMAKE_CXX_STANDARD 17)
> set(CMAKE_CXX_STANDARD_REQUIRED ON)
> ```

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

First create a solver object. Every method returns a `Result` struct containing the solution, and information about how it was found:
```
QPSolver<double> solver;

QPSolver<double>::Result result = solver.solve(H,f);

Eigen::VectorXd x = result.solution;
```

>[!TIP]
> The solver stores its matrix decompositions internally, so they are reused between calls. If you're solving the same size problem repeatedly (e.g. in a control loop), create the solver once and reuse it rather than creating a new one each time.

### Solver Options:

Options are set when constructing the solver:
```
QPSolver<double>::Options options;
options.method               = QPSolver<double>::Options::Method::interiorPoint;
options.minimumStepSize      = 1e-03;
options.maximumIterations    = 20;
options.maximumSolutionTime  = 1e-02;
options.initialBarrierScalar = 100;
options.barrierReductionRate = 1e-02;

QPSolver<double> solver(options);
```

| Option | Default | Description |
|---|---|---|
| `method` | `interiorPoint` | The algorithm for inequality constrained problems: `activeSet` or `interiorPoint`. |
| `minimumStepSize` | `1e-03` | The solver terminates when the norm of the step $\|\Delta\mathbf{x}\|$ is less than this value. A smaller value means a more accurate solution, but slower solution time. |
| `maximumIterations` | `5` | The solver terminates if this many iterations are reached. A higher value means a more accurate solution, but it might take longer to solve. |
| `maximumSolutionTime` | `1e-02` | The solver terminates if it takes longer than this many seconds. |
| `initialBarrierScalar` | `100` | **Interior point only.** The inequality constraints are converted to a log-barrier function. This determines how steep the slope of the barrier is. A smaller value means a faster solution, but you may prematurely run in to the constraint. |
| `barrierReductionRate` | `1e-02` | **Interior point only.** Every iteration the barrier slope is multiplied by this value, so it must be between 0 and 1. A smaller value means the barrier effect will shrink quickly. This will make the algorithm faster, but then it may not find a solution if it hits the constraints prematurely. |

The constructor will throw an `std::invalid_argument` if any of these are out of range.

### Solver Results:

The `Result` struct contains:

| Field | Description |
|---|---|
| `solution` | The decision variable $\mathbf{x}$ which minimises the objective function. |
| `status` | Why the solver terminated: `converged`, `maxIterationsReached`, or `timeout`. |
| `numberOfIterations` | How many iterations the solver performed. |
| `solutionTime` | How long it took to solve the problem (seconds). |
| `finalStepSize` | Magnitude of the last step before the solver terminated. |

For problems without inequality constraints the status is always `converged`. For problems _with_ inequality constraints, a status of `maxIterationsReached` or `timeout` means the solution is the best found so far: it satisfies the constraints, but may not be optimal. In a real-time control loop this is often still perfectly usable.

[:top: Back to top.](#cyclone-simple-qp-solver)

### A Generic QP Problem:

```math
\min_{\mathbf{x}} \frac{1}{2}\mathbf{x^\mathrm{T} Hx + x^\mathrm{T}f}
```
Assuming $\mathbf{H}$ is positive definite and $\mathbf{f}$ is given, then you can call:
```
Eigen::VectorXd x = solver.solve(H,f).solution;
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
Eigen::VectorXf x = solver.least_squares(y,A,W).solution;
```
### Least Squares with Equality Constraints (Under-determined Systems):
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
- $\mathbf{W}\in\mathbb{R}^\mathrm{n\times n}$ is a positive definite weighting on the solution,
- m < n (more inputs than outputs)

In this situation the $\mathbf{x}$ vector has more dimensions than the $\mathbf{y}$ vector. Therefore, infinite solutions exist. You can give a desired value $\mathbf{x}_\mathrm{d}$ that the solver will try to achieve whilst satisfying the relationship $\mathbf{Ax = y}$. Call:
```
Eigen::VectorXd x = solver.redundant_least_squares(xd,W,A,y).solution;
```

### Optimisation with Inequality Constraints:
```math
\begin{align}
	\min_{\mathbf{x}} \frac{1}{2}\mathbf{x^\mathrm{T}Hx + x^\mathrm{T}f} \\
	\text{subject to: } \mathbf{Bx} \le \mathbf{z}
\end{align}
```
>[!NOTE]
> For problems like this with inequality constraints, the solver uses an iterative algorithm (interior point or active set, depending on the [options](#solver-options)). It therefore requires a _start point_ or _initial guess_.

Call:
```
Eigen::VectorXd x = solver.solve(H,f,B,z,x0).solution;
```
where `x0` is the start point argument. If you also have equality constraints $\mathbf{Ax = y}$, use:
```
Eigen::VectorXd x = solver.solve(H,f,A,y,B,z,x0).solution;
```

If you're repeatedly solving a QP problem (e.g. in a control loop), you can use the last solution as the start point for the next one:
```
QPSolver<double>::Result result = solver.solve(H,f,B,z,x0);

x0 = result.solution;                                                                               // Use for next time
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
Eigen::VectorXf x = solver.constrained_least_squares(y,A,W,xMin,xMax,x0).solution;
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
Eigen::VectorXd x = solver.constrained_least_squares(xd,W,A,y,xMin,xMax,x0).solution;
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
Eigen::VectorXd x = solver.constrained_least_squares(xd,W,A,y,B,z,x0).solution;
```

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
./test interiorPoint
```
or
```
./test activeSet
```
to choose which method is used for the constrained problems (the default is `interiorPoint`). It prints information about the use of different class methods, as well as the accuracy and speed of solutions.

<img src="https://github.com/Woolfrey/software_simple_qp/assets/62581255/8ecb93a6-e45b-4ff9-aebf-1f2f27c62f25" width="600" height="auto">

[:top: Back to top.](#cyclone-simple-qp-solver)

## :rocket: Applications

- Woolfrey, J. (2026). Simultaneous force and motion control in generalised coordinates (arXiv:2610.22813). arXiv. https://doi.org/10.48550/arXiv.2610.22813
- Cheng, C., Li, T., Woolfrey, J., Zhao, L., Huang, S. (2026). MR-MPCC: Model Predictive Contouring Control for Robotic Manipulator Tracking under Moving Reference Frames IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) [Link](https://arxiv.org/pdf/2610.21292)

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
     version = {3.0.0},
     year    = {2026}
}
```
Here's the automatically generated APA format:
```
Woolfrey, J. (2026). Simple QP Solver (Version 3.0.0) [Computer software]. https://github.com/Woolfrey/software_simple_qp
```

[:top: Back to top.](#cyclone-simple-qp-solver)

## :scroll: License

This software package is licensed under the [GNU General Public License v3.0 (GPL-3.0)](https://choosealicense.com/licenses/gpl-3.0/). You are free to use, modify, and distribute this package, provided that any modified versions also comply with the GPL-3.0 license. All modified versions must make the source code available and be licensed under GPL-3.0. The license also ensures that the software remains free and prohibits the use of proprietary restrictions such as Digital Rights Management (DRM) and patent claims. For more details, please refer to the [full license text](LICENSE).

[:top: Back to top.](#cyclone-simple-qp-solver)
