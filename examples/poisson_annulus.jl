using Revise, RadialPiecewisePolynomials, ClassicalOrthogonalPolynomials
using ClassicalOrthogonalPolynomials
using PyPlot, Plots
import RadialPiecewisePolynomials: RadialCoordinate

"""
In this example we solve <∇u, ∇v> = <1, v> for all v ∈ H^1_0
on the annulus with inradius ρ=0.01 and outer radius 1. 

We use 3 annuli elements placed at the radii 0.01..0.05, 0.05..0.2, and 0.2..1.
"""

# Actual solution to PDE
function ua(x)
    x = RadialCoordinate(x)
    r = x.r; θ = x.θ;
    (0.25*(1-r^2) + (ρ^2-1)/(4*log(ρ)) * log(r))
end

# Right-hand side
function rhs(x)
    return 1.
end


ρ = 0.01; # Incircle radius of annulus domain 
points = [ρ; 0.05; 0.2; 1]; K = length(points)-1 # Radii of annuli elements
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
_, error = inf_error(F, θs, rs, vals, ua) # Check inf-norm errors on the grid
error
# Plot
plot(F,θs,rs,vals)

### Check decay of coefficients vs ChebyshevT
function uat(r)
    (0.25*(1-r^2) + (ρ^2-1)/(4*log(ρ)) * log(r))
end

T = chebyshevt(ρ.. 1); uct = T \ uat.(axes(T,1))

n = 1:50
X = zeros(length(n), K+1)
X[:,1] = uct[n]
for k in 1:K X[:,k+1] = uc[k][n] end
X[findall(x->x==0, X)] .= NaN
Labels = ["ChebyshevT"]
for k in 1:K Labels = hcat(Labels, ["C($(points[k]), $(points[k+1]))"]) end
p = Plots.plot(n, abs.(X),#,#abs.(X[:,1]),
        yscale=:log10,
        label=Labels,
        yticks=[1e-20,1e-15,1e-10,1e-5,1e0],
        xlabel="n",
        ylabel="Coefficient absolute value")