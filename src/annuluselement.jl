"""
ContinuousZernikeAnnulusElementMode{T, P<:AbstractVector, M<:Int, J<:Int}

is a quasi-vector representing the hat + bubble (m,j)-mode ZernikeAnnulus on a scaled annulus.

The points [α,β] defines the inner and outer radii
m is the Forier mode, j = 0 corresponds to sin, j = 1 corresponds to cos. 
"""

annulus(ρ::T, r::T) where T = (r*UnitDisk{T}()) \ (ρ*UnitDisk{T}())
# ClassicalOrthogonalPolynomials.checkpoints(d::DomainSets.SetdiffDomain{SVector{2, T}, Tuple{DomainSets.EuclideanUnitBall{2, T, :closed}, DomainSets.GenericBall{SVector{2, T}, :closed, T}}}) where T = [SVector{2,T}(cos(0.1),sin(0.1)), SVector{2,T}(cos(0.2),sin(0.2))]

struct ContinuousZernikeAnnulusElementMode{T, P<:AbstractVector, M<:Int, J<:Int, B<:Int} <: Basis{T}
    points::P
    m::M
    j::J
    b::B # Should remove once adaptive expansion has been figured out.
end

function ContinuousZernikeAnnulusElementMode{T}(points::AbstractVector, m::Int, j::Int, b::Int) where {T}
    @assert length(points) == 2 && zero(T) < points[1] < points[2] ≤ one(T)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    # @assert b ≥ m
    ContinuousZernikeAnnulusElementMode{T,typeof(points), Int, Int, Int}(points, m, j, b)
end

ContinuousZernikeAnnulusElementMode(points::AbstractVector, m::Int, j::Int, b::Int) = ContinuousZernikeAnnulusElementMode{Float64}(points, m, j, b)
ContinuousZernikeAnnulusElementMode(points::AbstractVector, m::Int, j::Int) = ContinuousZernikeAnnulusElementMode(points, m, j, 200)

axes(Z::ContinuousZernikeAnnulusElementMode) = (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(∞))
==(P::ContinuousZernikeAnnulusElementMode, Q::ContinuousZernikeAnnulusElementMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j

function getindex(Z::ContinuousZernikeAnnulusElementMode{T}, xy::StaticVector{2}, j::Int)::T where {T}
    p = Z.points
    α, β = convert(T, p[1]), convert(T, p[2])
    ρ = α / β
    rθ = RadialCoordinate(xy)
    r̃ = affine(α.. β, ρ.. 1)[rθ.r]
    xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
    Z.m == 0 && @assert Z.j == 1

    # The weight for ZernikeAnnulus is (r^2-ρ^2)^a (1-r^2)^b
    if j == 1
        Weighted(ZernikeAnnulus{T}(ρ, 0, 1))[xỹ, Block(1+Z.m)][Z.m+Z.j]
    elseif j == 2
        Weighted(ZernikeAnnulus{T}(ρ, 1, 0))[xỹ, Block(1+Z.m)][Z.m+Z.j]
    else
        Weighted(ZernikeAnnulus{T}(ρ, 1, 1))[xỹ, Block(2j-5+Z.m)][Z.m+Z.j]
    end
end

###
# Transforms
###

function _ann2element(t::T, m::Int) where T

    Q₀₀ = SemiclassicalJacobi{T}(t, 0, 0, m)
    Q₀₁ = SemiclassicalJacobi{T}(t, 0, 1, m)
    Q₁₀ = SemiclassicalJacobi{T}(t, 1, 0, m)
    Q₁₁ = SemiclassicalJacobi{T}(t, 1, 1, m)

    L₁₁ = (Weighted(Q₀₀) \ Weighted(Q₁₁)) / t^2
    L₀₁ = (Weighted(Q₀₀) \ Weighted(Q₀₁)) / t
    L₁₀ = (Weighted(Q₀₀) \ Weighted(Q₁₀)) / t

    (L₁₁, L₀₁, L₁₀)
end

function _scale_fcn(f, α::T, β::T, xy::AbstractArray) where T
    ρ = α / β
    rθ = RadialCoordinate(xy)
    r̃ = affine(ρ.. 1, α.. β)[rθ.r]
    xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
    f(xỹ)
end

# Use memoize to cache the Zernike annulus expansion. This will avoid the analysis for the same function being run
# many times when considering problems with multiple modes.
@memoize Dict function _zernikeannulus_ldiv(Z::ZernikeAnnulus{T}, f̃::QuasiArrays.BroadcastQuasiVector, f::AbstractMatrix, b::Int) where T
    # FIXME! Temporary hack to do with truncation and adaptive ldiv.
    # c = Z\f̃ # ZernikeAnnulus transform
    pad(Z[:, Block.(1:b)]\f̃, axes(Z,2))
end

function ldiv(C::ContinuousZernikeAnnulusElementMode{V}, f::AbstractQuasiVector) where V
    # T = promote_type(V, eltype(f))
    T = V
    α, β = convert(T, first(C.points)), convert(T, last(C.points))
    ρ = α / β
    m, j = C.m, C.j
    Z = ZernikeAnnulus{T}(ρ, 0, 0)
    x = axes(Z,1)
    # # Need to take into account different scalings
    if β ≉  1
        fc(xy) = _scale_fcn(f.f, α, β, xy)
        f̃ = fc.(x)
    else
        f̃ = f.f.(x)
    end

    c = _zernikeannulus_ldiv(Z, f̃, f̃.f.(AlgebraicCurveOrthogonalPolynomials.grid(Z,20)), C.b) # ZernikeAnnulus transform
    c̃ = ModalTrav(paddeddata(c))
    c̃ = c̃.matrix[:, 2*C.m + C.j]
    # Truncate machine error tail
    Ñ = findall(x->abs(x) > 2*eps(T), c̃)
    c̃ = isempty(Ñ) ? c̃[1:3] : c̃[1:Ñ[end]+min(5, length(c̃)-Ñ[end])]
    N = length(c̃) # degree
    
    t = inv(one(T)-ρ^2)
    (L₁₁, L₀₁, L₁₀) = _ann2element(t, m)
    R̃ = [L₁₀[:,1] L₀₁[:,1] L₁₁]

    # convert from ZernikeAnnulus(ρ,0,0) to hats + Bubble
    dat = R̃[1:N,1:N] \ c̃
    cfs = T[]
    pad(append!(cfs, dat), axes(C,2))
end

###
# L2 inner product
###
@simplify function *(A::QuasiAdjoint{<:Any,<:ContinuousZernikeAnnulusElementMode}, B::ContinuousZernikeAnnulusElementMode)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B

    α, β = convert(T, first(B.points)), convert(T, last(B.points))
    ρ = α / β
    m, j = B.m, B.j

    t = inv(one(T)-ρ^2)
    (L₁₁, L₀₁, L₁₀) = _ann2element(t, m)

    # Contribution from the mass matrix <R_m,j^(ρ,0,0,0),R_m,j^(ρ,0,0,0)>_L^2
    jw = sum(SemiclassicalJacobiWeight{T}(t,0,0,m))
    m₀ = convert(T,π) / ( t^(one(T) + m) ) * jw
    m₀ = m == 0 ? m₀ : m₀ / T(2)

    M = L₁₁' *  L₁₁

    a = ((L₁₁)'[1:2,:] * L₁₀[:,1])
    b = ((L₁₁)'[1:2,:] * L₀₁[:,1])

    a11 = ((L₁₀)'[1,:]' * L₁₀[:,1])
    a12 = ((L₁₀)'[1,:]' * L₀₁[:,1])
    a22 = ((L₀₁)'[1,:]' * L₀₁[:,1])
    
    C = [a11' a12' a'; a12' a22' b']

    M = [[C[1:2,3:4]'; Zeros{T}(∞,2)] M]
    β^2*m₀*Vcat([C Zeros{T}(2,∞)], M)
end

###
# Gradient and L2 inner product of gradient
##

"""
GradienContinuousZernikeAnnulusElementMode{T, P<:AbstractVector, M<:Int, J<:Int}

is a quasi-vector representing the gradient of ContinuousZernikeAnnulusElementMode.

The points, m, and j are the same as for ContinuousZernikeAnnulusElementMode. This is
effectively a placeholder for the actual implementation of the gradient of ZernikeAnnulus.
For now we use it as a intermediate to compute the weak Laplacian matrix. 
"""

struct GradientContinuousZernikeAnnulusElementMode{T, P<:AbstractVector, M<:Int, J<:Int}<:Basis{T}
    points::P
    m::M
    j::J
end

GradientContinuousZernikeAnnulusElementMode{T}(points::AbstractVector, m::Int, j::Int) where {T} =  GradientContinuousZernikeAnnulusElementMode{T,typeof(points), Int, Int}(points, m, j)
GradientContinuousZernikeAnnulusElementMode(points::AbstractVector, m::Int, j::Int) =  GradientContinuousZernikeAnnulusElementMode{Float64}(points, m, j)
GradientContinuousZernikeAnnulusElementMode(m::Int, j::Int) =  GradientContinuousZernikeAnnulusElementMode([0.0; 1.0], m, j)

axes(Z:: GradientContinuousZernikeAnnulusElementMode) = (Inclusion(last(Z.points)*UnitDisk{eltype(Z)}()), oneto(∞))
==(P:: GradientContinuousZernikeAnnulusElementMode, Q:: GradientContinuousZernikeAnnulusElementMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j

@simplify function *(D::Derivative, C::ContinuousZernikeAnnulusElementMode)
    GradientContinuousZernikeAnnulusElementMode(C.points, C.m, C.j)
end


# This explicit formula are calculated by noting ∇ = (∂ᵣ, ∂_θ/r),
# W^(0,1)_m,m,1 = (1-r^2) r^m Re[z^m] cos(mθ) and 
# W^(1,0)_m,m,1 = (r^2-ρ^2) r^m Re[z^m] cos(mθ)
# We then plug into Mathematica to extract the formulae.

# <∇ W^(0,1)_m,m,j, ∇ W^(0,1)_m,m,j>
function W010(m::Int, ρ::T) where T 
    (convert(T,π)*(T(2) - ρ^(2m)*(m*(2 + m) - 2m*(2 + m)*ρ^2 + (2 + m*(2 + m))*ρ^4 )))/(2 + m)
end

# <∇ W^(1,0)_m,m,j, ∇ W^(1,0)_m,m,j>
function W100(m::Int, ρ::T) where T
    (convert(T,π)*(T(2) - 2*ρ^(4 + 2m) + m*(2 + m)*(-one(T) + ρ^2)^2))/(2 + m)   
end

# <∇ W^(1,0)_m,m,j, ∇ W^(0,1)_m,m,j>
function W_100_010(m::Int, ρ::T) where T
    (2*convert(T,π)*(-one(T) + ρ^(4 + 2m)))/(2 + m)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:GradientContinuousZernikeAnnulusElementMode}, B::GradientContinuousZernikeAnnulusElementMode)
    T = promote_type(eltype(A), eltype(B))
    α, β = convert(T, first(B.points)), convert(T, last(B.points))
    ρ = α / β
    m = B.m
    # # Scaling if outer radius is β instead of 1.
    # s = inv(( (one(T)- ρ) / (β - ρ) )^2)
    
    # Contribution from the normalisation <w R_m,j^(ρ,1,1,0),R_m,j^(ρ,1,1,0)>_L^2
    t = inv(one(T)-ρ^2)
    jw = sum(SemiclassicalJacobiWeight{T}(t,1,1,m))
    m₀ = convert(T,π) / ( t^(3 + m) ) * jw
    m₀ = m == 0 ? m₀ : m₀ / T(2)
    
    Z = ZernikeAnnulus{T}(ρ,1,1)
    D = Z \ (Laplacian(axes(Z,1))*Weighted(Z))

    Δ = - m₀ * D.ops[m+1]
    if m == 0
        c = convert(T,2π)*(T(1) - ρ^4)
        C = [c -c 4*m₀*(m+1); -c c -4*m₀*(m+1)]
    else
        C = [W010(m, ρ) W_100_010(m, ρ) 4*m₀*(m+1); W_100_010(m, ρ) W100(m,ρ) -4*m₀*(m+1)]
    end

    Δ = [[C[1:2,3]'; Zeros{T}(∞,2)] Δ]
    Vcat([C Zeros{T}(2,∞)], Δ)
end

###
# Plotting
###

function grid(C::ContinuousZernikeAnnulusElementMode{T}, j::Int) where T
    Z = ZernikeAnnulus{T}(C.points[1]/C.points[2], 0, 0)
    AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(j+C.m))
end

function scalegrid(g::Matrix{RadialCoordinate{T}}, α::T, β::T) where T
    ρ = α / β
    rs = x -> affine(ρ.. 1, α.. β)[x.r]
    gs = (x, r) -> RadialCoordinate(SVector(r*cos(x.θ), r*sin(x.θ)))
    r̃ = map(rs, g)
    gs.(g, r̃)
end

function bubble2ann(α::T, β::T, m::Int, c::AbstractVector{T}) where T
    ρ = α / β
    t = inv(one(T)-ρ^2)
    (L₁₁, L₀₁, L₁₀) = _ann2element(t, m)
    R̃ = [L₁₀[:,1] L₀₁[:,1] L₁₁]
    if c isa LazyArray
        c = paddeddata(c)
    end
    R̃[1:length(c), 1:length(c)] * c # coefficients for ZernikeAnnulus(ρ,0,0)
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{ContinuousZernikeAnnulusElementMode, AbstractVector}}) where T
    C,c = u.args
    α, β = convert(T, first(C.points)), convert(T, last(C.points))
    ρ = α / β
    m = C.m

    Z = ZernikeAnnulus{T}(ρ, 0, 0)

    c̃ = bubble2ann(α, β, m, c)
    
    # Massage coeffcients into ModalTrav form.
    N = (isqrt(8*sum(1:2*length(c̃)+C.m-1)+1)-1) ÷ 2
    m = N ÷ 2 + 1
    n = 4(m-1) + 1

    f = zeros(T,m,n)
    f[:, 2C.m + C.j]= [c̃; zeros(size(f,1)-length(c̃))] 
    F = ModalTrav(f)

    # AlgebraicCurveOrthogonalPolynomials.plotvalues(Z*[F; zeros(∞)], x)
    N = size(f,1)

    # Scale the grid
    g = scalegrid(grid(C, N), α, β)

    # Use fast transforms for synthesis
    FT = ZernikeAnnulusITransform{T}(N+C.m, 0, 0, 0, Z.ρ)
    g, FT * pad(F,axes(Z,2))[Block.(OneTo(N+C.m))]
end

function plotannulus(g::Matrix{RadialCoordinate{T}}, vals::Matrix{T}) where T
    p = g -> [g.r, g.θ]
    rθ = map(p, g)
    r = first.(rθ)[:,1]
    θ = last.(rθ)[1,:]

    θ = [θ; 2π]
    vals = hcat(vals, vals[:,1])
    (θ, r, vals)
end