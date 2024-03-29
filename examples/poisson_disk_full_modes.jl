using Revise, RadialPiecewisePolynomials
using PyPlot
import ForwardDiff: derivative
include("../src/plotting/plots.jl") # Helper functions for plotting (do not want to compile PyPlot with RadialPiecewisePolynomials)

c1 = -10; c2 = 0; c3=0.6
function ua(x, y)
    exp(c1*(x^2 + (y-c3)^2)) * (1-(x^2+y^2))
end

function ua_xy(xy)
    x,y = first(xy), last(xy)
    ua(x,y)
end

function rhs_xy(xy)
    x,y = first(xy), last(xy)
    -(derivative(x->derivative(x->ua(x,y),x),x) + derivative(y->derivative(y->ua(x,y),y),y))
end

points = [0;0.3;0.5;0.8;1.0]; K = length(points)-1
N=40; F = ContinuousZernike(N, points); Z = ZernikeBasis(N, points, 0, 0)

x = axes(F,1)

fz = Z \ rhs_xy.(x)

(θs, rs, vals) = finite_plotvalues(Z, fz)
vals_, err = inf_error(Z, θs, rs, vals, rhs_xy) # Check inf-norm errors on the grid
err
plot(Z, θs, rs, vals)

# Solve Poisson equation in weak form
M = F' * F # list of mass matrices for each Fourier mode
D = Derivative(axes(F,1))
Δ = (D*F)' * (D*F) # list of stiffness matrices for each Fourier mode

Mf = (F'*Z) .* fz # right-hand side
zero_dirichlet_bcs!(F, Δ) # bcs
zero_dirichlet_bcs!(F, Mf) # bcs

# Solve over each Fourier mode seperately
u = Δ .\ Mf

(θs, rs, vals) = finite_plotvalues(F, u, N=200)
vals_, err = inf_error(F, θs, rs, vals, ua_xy) # Check inf-norm errors on the grid
err
plot(F, θs, rs, vals) # plot