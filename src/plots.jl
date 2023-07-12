using PyPlot

#######
### Helper plot functions
#######

function _plot(K::Int, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector)
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
end

function plot(F::FiniteContinuousZernike{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector) where T
    K = lastindex(F.points)-1
    _plot(K, θs, rs, vals)
end

function plot(F::FiniteContinuousZernikeMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector) where T
    K = lastindex(F.points)-1
    _plot(K, θs, rs, vals)
end

function plot(Z::FiniteZernikeBasis{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector) where T
    K = lastindex(Z.points)-1
    _plot(K, θs, rs, vals)
end


#######
### Compute inf-norm errors
#######
function _inf_error(K::Int, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function)
    vals_ = []
    for k = 1:K
        append!(vals_, [abs.(vals[k] - u.(RadialCoordinate.(rs[k],θs[k]')))])
    end
    vals_, sum(maximum.(vals_))
end

function inf_error(F::FiniteContinuousZernike{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function) where T
    K = lastindex(F.points)-1
    _inf_error(K, θs, rs, vals, u)
end

function inf_error(F::FiniteContinuousZernikeMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function) where T
    K = lastindex(F.points)-1
    _inf_error(K, θs, rs, vals, u)
end

function inf_error(Z::FiniteZernikeBasis{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function) where T
    K = lastindex(Z.points)-1
    _inf_error(K, θs, rs, vals, u)
end
