using Revise, RadialPiecewisePolynomials, ClassicalOrthogonalPolynomials
using ClassicalOrthogonalPolynomials
using PyPlot, Plots
import RadialPiecewisePolynomials: RadialCoordinate
import ForwardDiff: derivative
"""
In this example we solve <∇u, ∇v> = <1, v> for all v ∈ H^1_0
on the disk with outer radius 1. 

We use 3 annuli elements placed at the radii 0..0.05, 0.05..0.2, and 0.2..1.
"""

# Actual solution to PDE

function ua(x, y)
    r² = x^2+y^2; θ = atan(y, x);
    (0.25*(1-r²) * exp(-r²))
end

function ua_xy(xy)
    x,y = first(xy), last(xy)
    ua(x,y)
end

# Right-hand side
function rhs(xy)
    x,y = first(xy), last(xy)
    -(derivative(x->derivative(x->ua(x,y),x),x) + derivative(y->derivative(y->ua(x,y),y),y))
end


points = [0; 0.05; 0.2; 1]; K = length(points)-1 # Radii of annuli elements
N = 100 # Truncation degree
m = 0 # This problem is independent of the angle (effectively 1D). Only need to consider the m=0 mode.
j = 1 # Corresponding Fourier mode

F = FiniteContinuousZernikeMode(N, points, m, j)

# Expand the right-hand side over the the 3 elements
f = F \ rhs.(axes(F,1))

# Mass matrix
M = F' * F

# Stiffness matrix
D = Derivative(axes(F,1))
Δ = (D*F)' * (D*F)

# Map right-hand side to dual space
Mf = M*f

# Enforce the boundary conditions on the right-hand side and stiffness matrices
zero_dirichlet_bcs!(F, Δ, Mf)

## Solve the PDE!
u = Δ \ Mf

# Use fast transforms to extract values for plotting
(uc,θs,rs,vals) = element_plotvalues(F*u)

include("../src/plots.jl")
# Check inf-norm errors on the grid
_, error = inf_error(F, θs, rs, vals, ua_xy) # Check inf-norm errors on the grid
error
plot(F,θs,rs,vals)
