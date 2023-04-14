using PyPlot

function plot(F::FiniteContinuousZernikeAnnulus{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector) where T
    K = lastindex(F.points)-1
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

function plot(F::FiniteContinuousZernikeMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector) where T
    G = FiniteContinuousZernikeAnnulus{T}(F.N, F.points)
    plot(G, θs, rs, vals)
end

function inf_error(F::FiniteContinuousZernikeAnnulus{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function) where T
    vals_ = []
    K = lastindex(F.points)-1
    for k = 1:K
        append!(vals_, [abs.(vals[k] - u.(RadialCoordinate.(rs[k],θs[k]')))])
    end
    vals_, sum(maximum.(vals_))
end

function inf_error(F::FiniteContinuousZernikeMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function) where T
    G = FiniteContinuousZernikeAnnulus{T}(F.N, F.points)
    inf_error(G, θs, rs, vals, u)
end