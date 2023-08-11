# SimpleQPSolver
A small, efficient class for solving convex optimisation problems.

A generic quadratic programming (QP) problem is of the form:
```math
\begin{align}
	\min_{\mathbf{x}} ~ \frac{1}{2}\mathbf{x^\mathrm{T}Hx - x^\mathrm{T}x} \\
	\text{subject to: } \mathbf{Bx \le z}
\end{align}

```
where:
- $\mathbf{x}\in\mathbb{R}^\mathrm{n}$ is the decision variable,
- $\mathbf{H = H^\mathrm{T}}\in\mathbb{R}^\mathrm{n\times }$ is a weighting matrix,
- $\mathbf{f}\in\mathbb{R}^\mathrm{n}$ is the linear component of the quadratic equation,
- $\mathbf{B}\in\mathbb{R}^\mathrm{c\times n}$ is a constraint matrix, and
- $\mathbf{z}\in\mathbb{R}^\mathrm{c}$ is a constraint vector.

There are also several functions for handling linear least squares problems with equality and inequality constraints.

## Installing Instructions

## Using the QP Solver
