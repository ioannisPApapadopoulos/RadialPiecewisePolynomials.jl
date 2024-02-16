# RadialPiecewisePolynomials.jl

[![CI](https://github.com/ioannisPApapadopoulos/RadialPiecewisePolynomials.jl/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ioannisPApapadopoulos/RadialPiecewisePolynomials.jl/actions/workflows/ci.yml)
[![codecov.io](http://codecov.io/github/ioannisPApapadopoulos/RadialPiecewisePolynomials.jl/coverage.svg?branch=main)](http://codecov.io/github/ioannisPApapadopoulos/RadialPiecewisePolynomials.jl?branch=main)

A Julia package for a hierarchical hp-finite element basis on disk and annuli. The mesh is an innermost disk (omitted if the domain is an annulus) and concentric annuli. The basis consists of Zernike (annular) polynomials modified into hat (external shape) and bubble (internal shape) functions.

This package utilizes [AnnuliOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/AnnuliOrthogonalPolynomials.jl) and [SemiclassicalOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/SemiclassicalOrthogonalPolynomials.jl) for the construction of the FEM basis.


See the [examples/](https://github.com/ioannisPApapadopoulos/RadialPiecewisePolynomials.jl/tree/main/examples) folder for examples. Some basic usage:

```julia
julia> using RadialPiecewisePolynomials

julia> s = 0.5^(-1/3); points = [0; reverse([s^(-j) for j in 0:3])]
5-element Vector{Float64}:
 0.0
 0.49999999999999994
 0.6299605249474365
 0.7937005259840997
 1.0

julia> Φ = ContinuousZernike(10, points); # H¹ conforming disk FEM basis up to degree 10
ContinuousZernike at degree N=10 and endpoints [0.0, 0.49999999999999994, 0.6299605249474365, 0.7937005259840997, 1.0].

julia> Ψ = ZernikeBasis(10, points, 0, 0); # L² conforming disk FEM basis up to degree 10

julia> f = Ψ \ (xy -> sin(first(xy))+last(xy)).(xy); # Expand sin(x+y)

julia> M = Φ' * Φ; # mass matrix

julia> D = Derivative(axes(Φ,1)); A = (D*Φ)' * (D*Φ); # stiffness matrix

julia> b = (Φ' * Ψ) .* f; # load vector

julia> u = Matrix.(A .+ M) .\ b; # Solve (-Δ + I)u = f with zero Nuemann bcs

```
