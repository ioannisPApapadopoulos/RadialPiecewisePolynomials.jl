ClassicalOrthogonalPolynomials.checkpoints(d::DomainSets.GenericBall{SVector{2, T}, :closed, T}) where T = [SVector{2,T}(cos(0.1),sin(0.1)), SVector{2,T}(cos(0.2),sin(0.2))]

# I do not think it is faster to pass all the matrices for the disk element.
# Much quicker to just repeatedly call Z0 \ Weighted(Z1)

struct ContinuousZernikeElementMode{T, P<:AbstractVector} <: Basis{T}
    points::P
    m::Int
    j::Int
end

function ContinuousZernikeElementMode(points::AbstractVector{T}, m::Int, j::Int) where {T}
    @assert length(points) == 2 && points[1] ≈ 0 && 0 < points[2]
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    ContinuousZernikeElementMode{T,typeof(points)}(points, m, j)
end
# ContinuousZernikeElementMode(points::AbstractVector, m::Int, j::Int) = ContinuousZernikeElementMode{Float64}(points, m, j)
ContinuousZernikeElementMode(m::Int, j::Int) = ContinuousZernikeElementMode([0.0; 1.0], m, j)

axes(Z::ContinuousZernikeElementMode) = (Inclusion(last(Z.points)*UnitDisk{eltype(Z)}()), oneto(∞))
==(P::ContinuousZernikeElementMode, Q::ContinuousZernikeElementMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j

function getindex(Z::ContinuousZernikeElementMode{T}, xy::StaticVector{2}, j::Int)::T where {T}
    p = Z.points
    α, β = convert(T, p[1]), convert(T, p[2])

    rθ = RadialCoordinate(xy)
    r̃ = affine(α.. β, 0.. 1)[rθ.r]
    xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))

    if j == 1
        Zernike{T}(0)[xỹ, Block(1+Z.m)][Z.m+Z.j]
    else
        Weighted(Zernike{T}(1))[xỹ, Block(2j-3+Z.m)][Z.m+Z.j]
    end
end

###
# Transforms
###

function grid(C::ContinuousZernikeElementMode{T}, n::Integer) where T
    N = 2n-1 # degree
    # sinpi.((N .-(0:N-1) .- one(T)/2) ./ (2N))
    RadialCoordinate.(sinpi.((N .-(0:N-1) .- one(T)/2) ./ (2N)), 0.)
end

function _scale_fcn(f::Function, r::T, xy::AbstractArray) where T
    rθ = RadialCoordinate(xy)
    r̃ = affine(0.. 1, 0.. r)[rθ.r]
    xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
    f(xỹ)
end

function ldiv(C::ContinuousZernikeElementMode{T}, f::AbstractQuasiVector) where T
    # T = promote_type(V, eltype(f))
    Z0 = Zernike{T}(0)

    r = convert(T, f.args[1].domain.radius)
    # Need to take into account different scalings
    if r ≉  1
        fc(xy) = _scale_fcn(f.f, r, xy)
        x = axes(Z0,1)
        f̃ = fc.(x)
    else
        f̃ = f
    end

    # Seems to cache on its own, no need to memoize unlike annulus
    c = Z0 \ f̃ # Zernike transform
    c̃ = ModalTrav(paddeddata(c))
    N = size(c̃.matrix, 1) # degree
    
    # Restrict to relevant mode and add a column corresponding to the hat function.
    R = Z0 \ Weighted(Zernike{T}(1)) # Very fast and does not change with width of disk.

    R̃ =  [[one(T); Zeros{T}(∞)] R.ops[C.m+1]]

    # convert from Zernike(0,0) to hat + Bubble
    # Using adaptive ldiv, so possible we terminate before the required truncation
    # in FiniteContinuousZernike. So seperate case for that.
    cfs = T[]
    if 2*C.m + C.j ≤ size(c̃.matrix, 2)
        dat = R̃[1:N,1:N] \ c̃.matrix[:, 2*C.m + C.j]
        return pad(append!(cfs, dat), axes(C,2))
    else
        return pad(append!(cfs, zero(T)), axes(C,2))
    end
end

###
# L2 inner product
###
@simplify function *(A::QuasiAdjoint{<:Any,<:ContinuousZernikeElementMode}, B::ContinuousZernikeElementMode)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B
    ρ = convert(T, last(B.points))

    X = jacobimatrix(Normalized(Jacobi{T}(1,B.m)))
    M = ρ^2*(I-X)/2
    M = [[T[ρ^2/sqrt(B.m+2)]; Zeros{T}(∞)] M]
    Vcat([T[ρ^2]; T[ρ^2/sqrt(B.m+2)]; Zeros{T}(∞)]', M)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:ContinuousZernikeElementMode}, B::BroadcastQuasiMatrix{<:Any, typeof(*), <:Tuple{BroadcastQuasiVector, <:ContinuousZernikeElementMode}})
    λ, C = B.args
    T = promote_type(eltype(A), eltype(C))
    @assert A' == C

    ρ = convert(T, last(C.points))
    m = C.m

    L₁₁ = ρ / sqrt(2*one(T)) * (Weighted(Normalized(Jacobi{T}(0,m))) \ Weighted(Normalized(Jacobi{T}(1,m))))
    L = Hcat(Vcat(ρ*one(T), Zeros{T}(∞)), L₁₁)

    # We need to compute the Jacobi matrix multiplier addition due to the
    # variable Helmholtz coefficient λ(r²). We expand λ(r²) in chebyshevt
    # and then use Clenshaw to compute λ(ρ^2*(X-I)/2) where X is the 
    # correponding Jacobi matrix for this basis.
    Tn = chebyshevt(C.points[1]..C.points[2])
    u = Tn \ λ.f.(axes(Tn,1))
    X = jacobimatrix(Normalized(Jacobi(0, m)))
    W = Clenshaw(paddeddata(u), recurrencecoefficients(Tn)..., ρ^2*(X+I)/2, _p0(Tn))

    # TODO fix the excess zeros
    return ApplyArray(*, L', ApplyArray(*, W, L))
end

###
# Gradient and L2 inner product of gradient
##
struct GradientContinuousZernikeElementMode{T}<:Basis{T}
    C::ContinuousZernikeElementMode{T}
end

# GradientContinuousZernikeElementMode{T}(points::AbstractVector, m::Int, j::Int) where {T} =  GradientContinuousZernikeElementMode{T,typeof(points), Int, Int}(points, m, j)
# GradientContinuousZernikeElementMode(points::AbstractVector, m::Int, j::Int) =  GradientContinuousZernikeElementMode{Float64}(points, m, j)
# GradientContinuousZernikeElementMode(m::Int, j::Int) =  GradientContinuousZernikeElementMode([0.0; 1.0], m, j)

axes(Z:: GradientContinuousZernikeElementMode) = (Inclusion(last(Z.C.points)*UnitDisk{eltype(Z)}()), oneto(∞))
==(P::GradientContinuousZernikeElementMode, Q::GradientContinuousZernikeElementMode) = P.C == Q.C

@simplify function *(D::Derivative, C::ContinuousZernikeElementMode)
    GradientContinuousZernikeElementMode(C)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:GradientContinuousZernikeElementMode}, B::GradientContinuousZernikeElementMode)
    T = promote_type(eltype(A), eltype(B))
    Z = Zernike{T}(1)
    D = Z \ (Laplacian(axes(Z,1))*Weighted(Z))
    m = B.C.m

    Δ = -D.ops[m+1]
    cₘ = π*m*zerniker(m,m,0,0,one(T))^2 # = <∇ Z^(0,1)_{m,m,j},∇ Z^(0,1)_{m,m,j}>_L^2

    Vcat([T[cₘ]; Zeros{T}(∞)]', [Zeros{T}(∞) Δ])
end

###
# Plotting
###

_angle(rθ::RadialCoordinate) = rθ.θ

function grid(C::ContinuousZernikeElementMode{T}, j::Int) where T
    Z = Zernike{T}(zero(T), zero(T))
    MultivariateOrthogonalPolynomials.grid(Z, Block(j+C.m))
    # θ = [map(_angle,g[1,:]); 0]
    # [permutedims(RadialCoordinate.(1,θ)); g g[:,1]; permutedims(RadialCoordinate.(0,θ))][:,1:end-1]
end

function scalegrid(g::Matrix{RadialCoordinate{T}}, ρ::T) where T
    rs = x -> affine(0.. 1, 0.. ρ)[x.r]
    gs = (x, r) -> RadialCoordinate(SVector(r*cos(x.θ), r*sin(x.θ)))
    r̃ = map(rs, g)
    gs.(g, r̃)
end

function bubble2disk(m::Int, c::AbstractVector{T}) where T
    R = Zernike{T}(0) \ Weighted(Zernike{T}(1)) # Very fast and does not change with width of disk.
    R̃ =  [[T[1]; Zeros{T}(∞)] R.ops[m+1]]
    if c isa LazyArray
        c = paddeddata(c)
    end
    R̃[1:length(c), 1:length(c)] * c # coefficients for Zernike(0,0)
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{ContinuousZernikeElementMode, AbstractVector}}) where T
    C, c = u.args
    ρ = convert(T, last(C.points))
    m = C.m

    c̃ = bubble2disk(m, c)

    # Massage coeffcients into ModalTrav form.
    N = (isqrt(8*sum(1:2*length(c̃)+C.m-1)+1)-1) ÷ 2
    m = N ÷ 2 + 1
    n = 4(m-1) + 1

    f = zeros(T,m,n)
    f[:, 2C.m + C.j]= [c̃; zeros(size(f,1)-length(c̃))]
    F = ModalTrav(f)

    N = size(f,1)
    # Scale the grid
    g = scalegrid(grid(C, 2N), ρ)

    # g, (Z*pad(F,axes(Z,2)))[grid(C,N)]

    # Use fast transforms for synthesis
    FT = ZernikeITransform{T}(2N+C.m, 0, 0)
    g, FT * pad(F,blockedrange(oneto(∞)))[Block.(OneTo(2N+C.m))]
end