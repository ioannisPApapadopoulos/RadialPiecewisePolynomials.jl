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