struct FiniteZernikeBasis{T} <: Basis{T}
    N::Int
    points::AbstractVector{T}
    a::Int
    b::Int
    Zs::AbstractArray
end

function FiniteZernikeBasis(N::Int, points::AbstractVector{T}, a::Int, b::Int) where T
    @assert length(points) > 1 && points == sort(points)
    ρs = []
    for k = 1:length(points)-1
        α, β = convert(T, first(points[k])), convert(T, last(points[k+1]))
        append!(ρs, [α / β])
    end

    if ρs[1] ≈ 0
        Zs = [Zernike{T}(0,1); ZernikeAnnulus{T}.(ρs[2:end], 1, 1)]
    else
        Zs = ZernikeAnnulus{T}.(ρs, 1, 1)
    end
    FiniteZernikeBasis{T}(N, points, a, b, Zs)
end

function axes(Z::FiniteZernikeBasis{T}) where T
    first(Z.points) ≈ 0 && return (Inclusion(last(Z.points)*UnitDisk{T}()), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
    (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
end
==(P::FiniteZernikeBasis, Q::FiniteZernikeBasis) = P.N == Q.N && P.points == Q.points

function ldiv(Z::FiniteZernikeBasis{T}, f::AbstractQuasiVector) where T

    N, points, Zs = Z.N, Z.points, Z.Zs
    
    K = length(points)-1
    c = []
    for k in 1:K
        # Scale so outer radius 1.
        fc(xy) = _scale_fcn(f.f, points[k], points[k+1], xy)

        x = axes(Zs[k],1)
        f̃ = fc.(x)
        append!(c, [Zs[k][:,Block.(1:N)] \ f̃])
    end
    return c
end