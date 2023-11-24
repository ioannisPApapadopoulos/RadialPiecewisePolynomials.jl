"""
ContinuousZernikeAnnulusElementMode{T, P<:AbstractVector, M<:Int, J<:Int}

is a quasi-vector representing the hat + bubble (m,j)-mode ZernikeAnnulus on a scaled annulus.

The points [α,β] defines the inner and outer radii
m is the Forier mode, j = 0 corresponds to sin, j = 1 corresponds to cos. 
"""

annulus(ρ::T, r::T) where T = (r*UnitDisk{T}()) \ (ρ*UnitDisk{T}())
# ClassicalOrthogonalPolynomials.checkpoints(d::DomainSets.SetdiffDomain{SVector{2, T}, Tuple{DomainSets.EuclideanUnitBall{2, T, :closed}, DomainSets.GenericBall{SVector{2, T}, :closed, T}}}) where T = [SVector{2,T}(cos(0.1),sin(0.1)), SVector{2,T}(cos(0.2),sin(0.2))]

# Matrices for lowering to ZernikeAnnulus(1,1) via
# the Jacobi matrix. Stable, but probably higher complexity
# and cannot be used for L2 inner-product of FiniteZernikeBasis
# and FiniteContinuousZernike
function _ann2element_via_Jacobi(t::T, m::Int) where T
    Q₁₁ = SemiclassicalJacobi(t, 1, 1, m)
    
    x = axes(Q₁₁, 1)
    X = jacobimatrix(Q₁₁)
    R₁₁ = (X - X*X)/t^2
    R₀₁ = (I - X)/t
    R₁₀ = X/t

    (R₁₁, R₀₁, R₁₀)
end

# Matrices for lowering to ZernikeAnnulus(0,0) via
# direct lowering. Slower for now, but actually lower complexity.
function _ann2element_via_raising(t::T, m::Int) where T
    Q₀₀ = SemiclassicalJacobi(t, 0, 0, m)
    Q₀₁ = SemiclassicalJacobi(t, 0, 1, m)
    Q₁₀ = SemiclassicalJacobi(t, 1, 0, m)
    Q₁₁ = SemiclassicalJacobi(t, 1, 1, m)

    R₁₁ = (Weighted(Q₀₀) \ Weighted(Q₁₁)) / t^2
    R₀₁ = (Weighted(Q₀₀) \ Weighted(Q₀₁))[1:2,1] / t
    R₁₀ = (Weighted(Q₀₀) \ Weighted(Q₁₀))[1:2,1] / t

    Hcat(Vcat(R₁₀, Zeros{T}(∞)), Vcat(R₀₁, Zeros{T}(∞)), R₁₁)
end

struct ContinuousZernikeAnnulusElementMode{T} <: Basis{T}
    points::AbstractVector{T}
    m::Int
    j::Int
    R::AbstractMatrix{T}
    D::AbstractMatrix{T}
    normalize_constants::AbstractVector{T}
    b::Int # Should remove once adaptive expansion has been figured out.
end

function ContinuousZernikeAnnulusElementMode(points::AbstractVector{T}, m::Int, j::Int, R₁₁::AbstractMatrix, R₀₁::AbstractMatrix, R₁₀::AbstractMatrix, D::AbstractMatrix, normalize_constants::AbstractVector{T}, b::Int) where T
    @assert length(points) == 2 && zero(T) < points[1] < points[2] ≤ one(T)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    @assert length(normalize_constants) ≥ 2
    # @assert b ≥ m
    ContinuousZernikeAnnulusElementMode{T}(points, m, j, R, D, normalize_constants, b)
end

function ContinuousZernikeAnnulusElementMode(points::AbstractVector{T}, m::Int, j::Int) where T
    α, β = convert(T, first(points)), convert(T, last(points))
    ρ = α / β
    t = inv(one(T)-ρ^2)

    normalize_constants = [_sum_semiclassicaljacobiweight(t, a, a, m) for a in 1:-1:0]
    R = _ann2element_via_raising(t, m)

    Z = ZernikeAnnulus{T}(ρ,1,1)
    D = (Z \ (Laplacian(axes(Z,1))*Weighted(Z))).ops[m+1]

    ContinuousZernikeAnnulusElementMode(points, m, j, R, D, normalize_constants, 200)
end

# ContinuousZernikeAnnulusElementMode(points::AbstractVector, m::Int, j::Int, R₁₁::AbstractMatrix, R₀₁::AbstractMatrix, R₁₀::AbstractMatrix, D::AbstractMatrix, b::Int) = ContinuousZernikeAnnulusElementMode{Float64}(points, m, j, R₁₁, R₀₁, R₁₀, D, b)
# ContinuousZernikeAnnulusElementMode(points::AbstractVector, m::Int, j::Int, R₁₁::AbstractMatrix, R₀₁::AbstractMatrix, R₁₀::AbstractMatrix, D::AbstractMatrix) = ContinuousZernikeAnnulusElementMode(points, m, j, R₁₁, R₀₁, R₁₀, D, 200)

axes(Z::ContinuousZernikeAnnulusElementMode) = (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(∞))
==(P::ContinuousZernikeAnnulusElementMode, Q::ContinuousZernikeAnnulusElementMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j

function show(io::IO, C::ContinuousZernikeAnnulusElementMode)
    points = C.points
    print(io, "ContinuousZernikeAnnulusElementMode: $points.")
end

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

function ldiv(C::ContinuousZernikeAnnulusElementMode{T}, f::AbstractQuasiVector) where T
    # T = promote_type(V, eltype(f))

    α, β = convert(T, first(C.points)), convert(T, last(C.points))
    ρ = α / β

    Z = ZernikeAnnulus{T}(ρ, zero(T), zero(T))

    x = axes(Z,1)
    # # Need to take into account different scalings
    if β ≉  1
        fc(xy) = _scale_fcn(f.f, α, β, xy)
        f̃ = fc.(x)
    else
        f̃ = f.f.(x)
    end

    c = _zernikeannulus_ldiv(Z, f̃, f̃.f.(AnnuliOrthogonalPolynomials.grid(Z,20)), C.b) # ZernikeAnnulus transform
    c̃ = ModalTrav(paddeddata(c))
    c̃ = c̃.matrix[:, 2*C.m + C.j]
    # Truncate machine error tail
    Ñ = findall(x->abs(x) > 2*eps(T), c̃)
    c̃ = isempty(Ñ) ? Zeros{T}(3) : c̃[1:Ñ[end]+min(5, length(c̃)-Ñ[end])]
    N = length(c̃) # degree

    R̃ = view(C.R, 1:N, 1:N)

    # convert from ZernikeAnnulus(ρ,w_a,w_a) to hats + Bubble
    dat = R̃[1:N,1:N] \ c̃
    cfs = T[]
    pad(append!(cfs, dat), axes(C,2))
end


###
# L2 inner product
###

function _sum_semiclassicaljacobiweight(t::T, a::Number, b::Number, c::Number) where T
    (t,a,b,c) = map(big, map(float, (t,a,b,c)))
    # return convert(T, t^c * beta(1+a,1+b) * _₂F₁general2(1+a,-c,2+a+b,1/t))
    # Working much better thanks to Timon Gutleb's efforts,
    # see https://github.com/JuliaApproximation/SemiclassicalOrthogonalPolynomials.jl/pull/90
    convert(T, first(sum.(SemiclassicalJacobiWeight.(t,a,b,c:c))))
end

function _mass_m₀(C::ContinuousZernikeAnnulusElementMode, m::Int, t::T) where T
    jw = C.normalize_constants[2] # _sum_semiclassicaljacobiweight(t,0,0,m)
    m₀ = convert(T,π) / ( t^(one(T) + m) ) * jw
    m₀ = m == 0 ? m₀ : m₀ / T(2)
    return m₀
end

@simplify function *(A::QuasiAdjoint{<:Any,<:ContinuousZernikeAnnulusElementMode}, B::ContinuousZernikeAnnulusElementMode)
    @assert A' == B
    mass_matrix(B)
end

function mass_matrix(C::ContinuousZernikeAnnulusElementMode)
    T = eltype(C)

    α, β = convert(T, first(C.points)), convert(T, last(C.points))
    ρ = α / β
    m = C.m

    t = inv(one(T)-ρ^2)

    m₀ = _mass_m₀(C, m, t)
    # TODO fix the excess zeros
    return ApplyArray(*,Diagonal(Fill(β^2*m₀,∞)), ApplyArray(*, C.R', C.R))
end



@simplify function *(A::QuasiAdjoint{<:Any,<:ContinuousZernikeAnnulusElementMode}, B::BroadcastQuasiMatrix{<:Any, typeof(*), <:Tuple{BroadcastQuasiVector, <:ContinuousZernikeAnnulusElementMode}})
    λ, C = B.args
    @assert A' == C
    T = eltype(C)

    α, β = convert(T, first(C.points)), convert(T, last(C.points))
    ρ = α / β
    m = C.m
    t = inv(one(T)-ρ^2)
    # We need to compute the Jacobi matrix multiplier addition due to the
    # variable Helmholtz coefficient λ(r²). We expand λ(r²) in chebyshevt
    # and then use Clenshaw to compute λ(β^2*(I-X/t)) where X is the 
    # correponding Jacobi matrix for this basis.
    Tn = chebyshevt(C.points[1]..C.points[2])
    u = Tn \ λ.f.(axes(Tn,1))
    X = jacobimatrix(SemiclassicalJacobi(t, 0, 0, m))
    Λ = Clenshaw(paddeddata(u), recurrencecoefficients(Tn)..., β^2*(I-X/t), _p0(Tn))

    assembly_matrix(C, Λ)
end

function assembly_matrix(C::ContinuousZernikeAnnulusElementMode, Λ::AbstractMatrix)
    T = eltype(C)
    α, β = convert(T, first(C.points)), convert(T, last(C.points))
    ρ = α / β
    m = C.m

    t = inv(one(T)-ρ^2)
    m₀ = _mass_m₀(C, m, t)

    # TODO fix the excess zeros
    ApplyArray(*,Diagonal(Fill(β^2*m₀,∞)), ApplyArray(*, C.R', ApplyArray(*, Λ, C.R)))
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

struct GradientContinuousZernikeAnnulusElementMode{T}<:Basis{T}
    C::ContinuousZernikeAnnulusElementMode{T}
end

# GradientContinuousZernikeAnnulusElementMode{T}(points::AbstractVector, m::Int, j::Int) where {T} =  GradientContinuousZernikeAnnulusElementMode{T,typeof(points), Int, Int}(points, m, j)
# GradientContinuousZernikeAnnulusElementMode(points::AbstractVector, m::Int, j::Int) =  GradientContinuousZernikeAnnulusElementMode{Float64}(points, m, j)
# GradientContinuousZernikeAnnulusElementMode(m::Int, j::Int) =  GradientContinuousZernikeAnnulusElementMode([0.0; 1.0], m, j)

axes(Z::GradientContinuousZernikeAnnulusElementMode) = (Inclusion(last(Z.C.points)*UnitDisk{eltype(Z)}()), oneto(∞))
==(P::GradientContinuousZernikeAnnulusElementMode, Q::GradientContinuousZernikeAnnulusElementMode) = P.C == Q.C

@simplify function *(D::Derivative, C::ContinuousZernikeAnnulusElementMode)
    GradientContinuousZernikeAnnulusElementMode(C)
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
    @assert A' == B
    C = B.C
    stiffness_matrix(C)
end

function stiffness_matrix(C::ContinuousZernikeAnnulusElementMode)
    T = eltype(C)
    α, β = convert(T, first(C.points)), convert(T, last(C.points))
    ρ = α / β
    m = C.m
    
    # Contribution from the normalisation <w R_m,j^(ρ,1,1,0),R_m,j^(ρ,1,1,0)>_L^2
    t = inv(one(T)-ρ^2)
    jw = C.normalize_constants[1] # _sum_semiclassicaljacobiweight(t,1,1,m)
    m₀ = convert(T,π) / ( t^(3 + m) ) * jw
    m₀ = m == 0 ? m₀ : m₀ / 2

    Δ = - m₀ * C.D
    if m == 0
        c = convert(T,2π)*(one(T) - ρ^4)
        C = [c -c 4*m₀*(m+1); -c c -4*m₀*(m+1)]
    else
        C = [W010(m, ρ) W_100_010(m, ρ) 4*m₀*(m+1); W_100_010(m, ρ) W100(m,ρ) -4*m₀*(m+1)]
    end

    Δ = [[C[1:2,3]'; Zeros{T}(∞,2)] Δ]
    Vcat(Hcat(C, Zeros{T}(2,∞)), Δ)
end

###
# Plotting
###

function grid(C::ContinuousZernikeAnnulusElementMode{T}, j::Int) where T
    Z = ZernikeAnnulus{T}(C.points[1]/C.points[2], zero(T), zero(T))
    AnnuliOrthogonalPolynomials.grid(Z, Block(j+C.m))
end

function scalegrid(g::Matrix{RadialCoordinate{T}}, α::T, β::T) where T
    ρ = α / β
    rs = x -> affine(ρ.. 1, α.. β)[x.r]
    gs = (x, r) -> RadialCoordinate(SVector(r*cos(x.θ), r*sin(x.θ)))
    r̃ = map(rs, g)
    gs.(g, r̃)
end

function bubble2ann(C::ContinuousZernikeAnnulusElementMode, c::AbstractVector{T}) where T
    if c isa LazyArray
        c = paddeddata(c)
    end
    N = length(c)
    R̃ = view(C.R, 1:N, 1:N) #Hcat(view(R₁₀,1:N,1), view(R₀₁,1:N,1), view(R₁₁,1:N, 1:N-2))
    R̃ * c # coefficients for ZernikeAnnulus(ρ,w_a,w_a)
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{ContinuousZernikeAnnulusElementMode, AbstractVector}}) where T
    C,c = u.args
    α, β = convert(T, first(C.points)), convert(T, last(C.points))
    ρ = α / β
    m = C.m

    c̃ = bubble2ann(C, c)
    
    # Massage coeffcients into ModalTrav form.
    N = (isqrt(8*sum(1:2*length(c̃)+C.m-1)+1)-1) ÷ 2
    m = N ÷ 2 + 1
    n = 4(m-1) + 1

    f = zeros(T,m,n)
    f[:, 2C.m + C.j]= [c̃; zeros(size(f,1)-length(c̃))] 
    F = ModalTrav(f)

    # AnnuliOrthogonalPolynomials.plotvalues(Z*[F; zeros(∞)], x)
    N = size(f,1)

    # Scale the grid
    g = scalegrid(grid(C, N), α, β)

    # Use fast transforms for synthesis
    FT = ZernikeAnnulusITransform{T}(N+C.m, 0, 0, 0, ρ)
    g, FT * pad(F,blockedrange(oneto(∞)))[Block.(OneTo(N+C.m))]
end

function plot_helper(g::Matrix{RadialCoordinate{T}}, vals::Matrix{T}) where T
    p = g -> [g.r, g.θ]
    rθ = map(p, g)
    r = first.(rθ)[:,1]
    θ = last.(rθ)[1,:]

    θ = [θ; 2π]
    vals = hcat(vals, vals[:,1])
    (θ, r, vals)
end