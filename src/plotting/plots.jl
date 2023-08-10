using PyPlot, DelimitedFiles

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

### This helper functions takes in the coordinates and vals and saves the relevant logs
### to be used with the MATLAB script for plotting solutions on cylinders
function cylinder_plot_save(xy::Matrix{<:RadialCoordinate}, z::AbstractArray, vals::AbstractMatrix, path="src/plotting/")
    writedlm(path*"z.log", z)
    r = [xy[i,1].r for i in 1:size(xy,1)]
    writedlm(path*"r.log", r)
    θ = [xy[1,j].θ for j in 1:size(xy,2)]
    writedlm(path*"theta.log", θ)
    
    writedlm(path*"vals.log", reshape(vals, length(r), length(θ), length(z))) 
end