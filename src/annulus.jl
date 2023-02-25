struct ContinuousZernikeAnnulusMode{T, P<:AbstractVector, M<:Int, J<:Int} <: Basis{T}
    points::P
    m::M
    j::J
end

function ContinuousZernikeAnnulusMode{T}(points::AbstractVector, m::Int, j::Int) where {T}
    @assert length(points) > 1 && points == sort(points)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    ContinuousZernikeAnnulusMode{T,typeof(points), Int, Int}(points, m, j)
end

ContinuousZernikeAnnulusMode(points::AbstractVector, m::Int, j::Int) = ContinuousZernikeAnnulusMode{Float64}(points, m, j)

axes(Z::ContinuousZernikeAnnulusMode) = (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(∞))
==(P::ContinuousZernikeAnnulusMode, Q::ContinuousZernikeAnnulusMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j

function getindex(C::ContinuousZernikeAnnulusMode{T}, xy::StaticVector{2}, j::Int)::T where {T}
    K = length(C.points) - 1
    b = ((j-1) % K) + 1
    i = (j + K-1) ÷ K
    
    # FIXME: I need to combine the hat functions
    if b == searchsortedlast(C.points, RadialCoordinate(xy).r)
        ContinuousZernikeAnnulusElementMode{T}([C.points[b]; C.points[b+1]], C.m, C.j)[xy, i]
    else
        return zero(T)
    end
end