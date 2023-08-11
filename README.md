# SimpleQPSolver
A small, efficient class for solving convex optimisation problems.

A generic quadratic programming (QP) problem is of the form:
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

## Installation Instructions

### Installing Eigen

```
sudo apt install libeigen3-dev
```

### Installing SimpleQPSolver

SimpleQPSolver is a template class contained in a single header file. There is no need to clone this repository and build any packages (unless you want to, of course). All you need to do is download the header file `QPSolver.h` and include it in your package:

```
SimpleQPSolver/include/QPSolver.h
```

There is a simple `test.cpp` file you can run that demonstrates the use of the `QPSolver` class. First navigate to your working directory:

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

## Using the QP Solver

You can use `float` or `double` with this class.

Input arguments for the `Eigen` classes must match:
- `Eigen::MatrixXf` and `Eigen::VectorXf` when using `QPSolver<float>`, or
- `Eigen::MatrixXd` and `Eigen::VectorXd` when using `QPSolver<double>`.

### A Generic QP Problem

### Linear Least Squares (Linear Regression)

### Least Squares with Equality Constraints (Over-determined Systems)

### Optimisation with Inequality Constraints

### Options for the Interior Point Algorithm

