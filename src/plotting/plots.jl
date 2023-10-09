using PyPlot, DelimitedFiles, LaTeXStrings

#######
### Helper plot functions
#######

function _plot(K::Int, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[]) where T
    PyPlot.rc("font", family="serif", size=14)
    rcParams = PyPlot.PyDict(PyPlot.matplotlib["rcParams"])
    rcParams["text.usetex"] = true
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="polar")
    vmin,vmax = minimum(minimum.(vals)), maximum(maximum.(vals))
    norm = PyPlot.matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    if ρ > 0.0
        ax.set_ylim(ρ,1)
        ax.set_rorigin(0)
        tick_inner_radial = isodd(10*ρ) ? ρ+0.1 : ρ
        ax.set_rticks(tick_inner_radial:0.2:1)
        y_tick_labels = tick_inner_radial:0.2:1
        ax.set_yticklabels(y_tick_labels)
    end

    pc = []
    for k=1:K
        pc = pcolormesh(θs[k], rs[k], vals[k], cmap="bwr", shading="gouraud", norm=norm)
    end

    if ttl != []
        cbar = plt.colorbar(pc, pad=0.1)#, cax = cbar_ax)
        cbar.set_label(ttl)
    end
    display(gcf())
end

function plot(F::FiniteContinuousZernike{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[]) where T
    K = lastindex(F.points)-1
    _plot(K, θs, rs, vals, ρ=ρ, ttl=ttl)
end

function plot(F::FiniteContinuousZernikeMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[]) where T
    K = lastindex(F.points)-1
    _plot(K, θs, rs, vals, ρ=ρ, ttl=ttl)
end

function plot(Z::FiniteZernikeBasis{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[]) where T
    K = lastindex(Z.points)-1
    _plot(K, θs, rs, vals, ρ=ρ, ttl=ttl)
end

function plot(C::ContinuousZernikeAnnulusElementMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[]) where T
    _plot(1, θs, rs, vals, ρ=ρ, ttl=ttl)
end

function plot(C::ContinuousZernikeElementMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector;ρ::T=0.0, ttl=[]) where T
    _plot(1, θs, rs, vals, ρ=ρ, ttl=ttl)
end

function plot(θs::AbstractVector, rs::AbstractVector, vals::AbstractVector; ρ::T=0.0, ttl=[]) where T
    _plot(1, [[θs[1]; 2π]], rs, [hcat(vals[1], vals[1][:,1])], ρ=ρ, ttl=ttl)
    # _plot(1, θs, rs, vals, ρ=ρ, ttl=ttl)
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