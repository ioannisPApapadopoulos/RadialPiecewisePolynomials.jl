using Revise, RadialPiecewisePolynomials
using PyPlot
import ForwardDiff: derivative
include("../src/plots.jl") # Helper functions for plotting (do not want to compile PyPlot with RadialPiecewisePolynomials)

c1 = -10; c2 = 0; c3=0.6
function ua(x, y)
    exp(c1*(x^2 + (y-c3)^2)) * (1-(x^2+y^2)) * ((x^2+y^2)-ρ^2)
end

function ua_xy(xy)
    x,y = first(xy), last(xy)
    ua(x,y)
end

function rhs_xy(xy)
    x,y = first(xy), last(xy)
    -(derivative(x->derivative(x->ua(x,y),x),x) + derivative(y->derivative(y->ua(x,y),y),y))
end

ρ = 0.2
points = [ρ;0.3;0.5;0.8;1.0]; K = length(points)-1
N=40; F = FiniteContinuousZernike(N, points)

x = axes(F,1)

f = F \ rhs_xy.(x)
(θs, rs, vals) = finite_plotvalues(F, f)
vals_, err = inf_error(F, θs, rs, vals, rhs_xy) # Check inf-norm errors on the grid
err
plot(F, θs, rs, vals)

# Solve Poisson equation in weak form
M = F' * F # list of mass matrices for each Fourier mode
D = Derivative(x)
Δ = (D*F)' * (D*F) # list of stiffness matrices for each Fourier mode

Mf = M .* f # right-hand side
zero_dirichlet_bcs!(F, Δ, Mf) # bcs

# Solve over each Fourier mode seperately
u = Δ .\ Mf

(θs, rs, vals) = finite_plotvalues(F, u)
vals_, err = inf_error(F, θs, rs, vals, ua_xy) # Check inf-norm errors on the grid
err
plot(F, θs, rs, vals) # plot