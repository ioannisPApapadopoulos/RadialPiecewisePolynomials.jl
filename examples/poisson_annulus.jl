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

F = FiniteContinuousZernikeAnnulusMode(N, points, m, j)

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

# Check inf-norm errors on the grid
vals_ = []
for k = 1:K
    append!(vals_, [abs.(vals[k] - ua.(RadialCoordinate.(rs[k],θs[k]')))])
end
sum(maximum.(vals_)) # Inf-norm error


# Plotting via PyPlots
fig = plt.figure()
ax = fig.add_subplot(111, projection="polar")
vmin,vmax = minimum(minimum.(vals)), maximum(maximum.(vals))
norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
pc = []
for k=1:K
    pc = pcolormesh(θs[k], rs[k], vals[k], cmap="bwr", shading="gouraud", norm=norm)
end
cbar = plt.colorbar(pc, pad=0.2)#, cax = cbar_ax)
display(gcf())


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