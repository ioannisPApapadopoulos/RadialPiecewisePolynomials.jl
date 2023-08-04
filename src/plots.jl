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

function plot(C::ContinuousZernikeAnnulusElementMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector) where T
    _plot(1, θs, rs, vals)
end

function plot(C::ContinuousZernikeElementMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector) where T
    _plot(1, θs, rs, vals)
end