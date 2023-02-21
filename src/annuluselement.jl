annulus(ρ::T, r::T) where T = (r*UnitDisk{T}()) \ (ρ*UnitDisk{T}())
# ClassicalOrthogonalPolynomials.checkpoints(d::DomainSets.SetdiffDomain{SVector{2, T}, Tuple{DomainSets.EuclideanUnitBall{2, T, :closed}, DomainSets.GenericBall{SVector{2, T}, :closed, T}}}) where T = [SVector{2,T}(cos(0.1),sin(0.1)), SVector{2,T}(cos(0.2),sin(0.2))]

struct ContinuousZernikeAnnulusElementMode{T, P<:AbstractVector, M<:Int, J<:Int} <: Basis{T}
    points::P
    m::M
    j::J
end

function ContinuousZernikeAnnulusElementMode{T}(points::AbstractVector, m::Int, j::Int) where {T}
    @assert length(points) == 2 && zero(T) < points[1] < points[2] ≤ one(T)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    ContinuousZernikeAnnulusElementMode{T,typeof(points), Int, Int}(points, m, j)
end

ContinuousZernikeAnnulusElementMode(points::AbstractVector, m::Int, j::Int) = ContinuousZernikeAnnulusElementMode{Float64}(points, m, j)

axes(Z::ContinuousZernikeAnnulusElementMode) = (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(∞))
==(P::ContinuousZernikeAnnulusElementMode, Q::ContinuousZernikeAnnulusElementMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j

function getindex(Z::ContinuousZernikeAnnulusElementMode{T}, xy::StaticVector{2}, j::Int)::T where {T}
    p = Z.points
    ρ, β = convert(T, p[1]), convert(T, p[2])
    rθ = RadialCoordinate(xy)
    r̃ = affine(ρ.. β, ρ.. 1)[rθ.r]
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

function ann2element(t::T, m::Int) where T

    Q₀₀ = SemiclassicalJacobi{T}(t, 0, 0, m)
    Q₀₁ = SemiclassicalJacobi{T}(t, 0, 1, m)
    Q₁₀ = SemiclassicalJacobi{T}(t, 1, 0, m)
    Q₁₁ = SemiclassicalJacobi{T}(t, 1, 1, m)

    L₁₁ = (Weighted(Q₀₀) \ Weighted(Q₁₁)) / t^2
    L₀₁ = (Weighted(Q₀₀) \ Weighted(Q₀₁)) / t
    L₁₀ = (Weighted(Q₀₀) \ Weighted(Q₁₀)) / t

    (L₁₁, L₀₁, L₁₀)
end

function fa_annulus(f, ρ, r, xy)
    rθ = RadialCoordinate(xy)
    r̃ = affine(ρ.. 1, ρ.. r)[rθ.r]
    xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
    f(xỹ)
end

function ldiv(C::ContinuousZernikeAnnulusElementMode{V}, f::AbstractQuasiVector) where V
    # T = promote_type(V, eltype(f))
    T = V
    ρ, β = convert(T, first(C.points)), convert(T, last(C.points))
    m, j = C.m, C.j
    Z = ZernikeAnnulus{T}(ρ, 0, 0)
    x = axes(Z,1)
    # # Need to take into account different scalings
    if β ≉  1
        fc(xy) = fa_annulus(f.f, ρ, β, xy)
        f̃ = fc.(x)
    else
        f̃ = f.f.(x)
    end
    c = Z\f̃ # ZernikeAnnulus transform
    c̃ = ModalTrav(paddeddata(c))
    N = size(c̃.matrix, 1) # degree
    
    t = inv(one(T)-ρ^2)
    (L₁₁, L₀₁, L₁₀) = ann2element(t, m)
    R̃ = [L₁₀[:,1] L₀₁[:,1] L₁₁]

    # convert from ZernikeAnnulus(ρ,0,0) to hats + Bubble
    dat = R̃[1:N,1:N] \ c̃.matrix[:, 2*C.m + C.j]
    cfs = T[]
    pad(append!(cfs, dat), axes(C,2))
end

###
# L2 inner product
###
@simplify function *(A::QuasiAdjoint{<:Any,<:ContinuousZernikeAnnulusElementMode}, B::ContinuousZernikeAnnulusElementMode)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B

    ρ, β = convert(T, first(B.points)), convert(T, last(B.points))
    m, j = B.m, B.j

    t = inv(one(T)-ρ^2)
    (L₁₁, L₀₁, L₁₀) = ann2element(t, m)

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
    m₀*Vcat([C Zeros{T}(2,∞)], M)
end

###
# Gradient and L2 inner product of gradient
##
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
    (convert(T,π)*(T(2) - ρ^(2m)*(m*(T(2) + m) - 2m*(T(2) + m)*ρ^(T(2)) + (T(2) + m*(T(2) + m))*ρ^(T(4)) )))/(T(2) + m)
end

# <∇ W^(1,0)_m,m,j, ∇ W^(1,0)_m,m,j>
function W100(m::Int, ρ::T) where T
    (convert(T,π)*(T(2) - T(2)*ρ^(T(4) + 2m) + m*(T(2) + m)*(-one(T) + ρ^(T(2)))^(T(2))))/(T(2) + m)   
end

# <∇ W^(1,0)_m,m,j, ∇ W^(0,1)_m,m,j>
function W_100_010(m::Int, ρ::T) where T
    (T(2)*convert(T,π)*(-one(T) + ρ^(T(4) + 2m)))/(T(2) + m)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:GradientContinuousZernikeAnnulusElementMode}, B::GradientContinuousZernikeAnnulusElementMode)
    T = promote_type(eltype(A), eltype(B))
    ρ, β = convert(T, first(B.points)), convert(T, last(B.points))
    m = B.m
    # Scaling if outer radius is β instead of 1.
    s = inv(( (one(T)- ρ) / (β - ρ) )^2)
    # Contribution from the normalisation <w R_m,j^(ρ,1,1,0),R_m,j^(ρ,1,1,0)>_L^2
    t = inv(one(T)-ρ^2)
    jw = sum(SemiclassicalJacobiWeight{T}(t,1,1,m))
    m₀ = convert(T,π) / ( t^(T(3) + m) ) * jw
    m₀ = m == 0 ? m₀ : m₀ / T(2)
    
    Z = ZernikeAnnulus{T}(ρ,1,1)
    D = Z \ (Laplacian(axes(Z,1))*Weighted(Z))

    Δ = -s * m₀ * D.ops[m+1]
    if m == 0
        c = convert(T,2π)*(T(1) - ρ^4)
        C = [c -c 4s*m₀*(m+1); -c c -4s*m₀*(m+1)]
    else
        C = [W010(m, ρ) W_100_010(m, ρ) 4s*m₀*(m+1); W_100_010(m, ρ) W100(m,ρ) -4s*m₀*(m+1)]
    end

    Δ = [[C[1:2,3]'; Zeros{T}(∞,2)] Δ]
    Vcat([C Zeros{T}(2,∞)], Δ)
end

###
# Plotting
###

function grid(C::ContinuousZernikeAnnulusElementMode{T}, j::Int) where T
    Z = ZernikeAnnulus{T}(C.points[1], 0, 0)
    AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(j+C.m))
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{ContinuousZernikeAnnulusElementMode, AbstractVector}}) where T
    C,c = u.args
    ρ, β = convert(T, first(C.points)), convert(T, last(C.points))
    m = C.m
    
    β < 1 && error("Currently can only plot if outer radius 1")
    # First scale the grid
    # g = grid(C, N)
    # rθ = RadialCoordinate(xy)
    # r̃ = affine(ρ.. β, ρ.. 1)[rθ.r]
    # xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))

    Z = ZernikeAnnulus{T}(ρ, 0, 0)

    t = inv(one(T)-ρ^2)
    (L₁₁, L₀₁, L₁₀) = ann2element(t, m)
    R̃ = [L₁₀[:,1] L₀₁[:,1] L₁₁]
 
    c = paddeddata(c)
    c̃ = R̃[1:length(c), 1:length(c)] * c # coefficients for ZernikeAnnulus(ρ,0,0)
    
    # Massage coeffcients into ModalTrav form.
    N =  (isqrt(8*sum(1:2*length(c̃))+1)-1) ÷ 2
    m = N ÷ 2 + 1
    n = 4(m-1) + 1

    f = zeros(T,m,n)
    f[:, 2C.m + C.j]= [c̃; zeros(size(f,1)-length(c̃))] 
    F = ModalTrav(f)

    # AlgebraicCurveOrthogonalPolynomials.plotvalues(Z*[F; zeros(∞)], x)

    # Use fast transforms for synthesis
    N = size(f,1)
    FT = ZernikeAnnulusITransform{T}(N+C.m, 0, 0, 0, Z.ρ)
    grid(C, N), FT * pad(F,axes(Z,2))[Block.(OneTo(N+C.m))]
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