ClassicalOrthogonalPolynomials.checkpoints(d::DomainSets.GenericBall{SVector{2, T}, :closed, T}) where T = [SVector{2,T}(cos(0.1),sin(0.1)), SVector{2,T}(cos(0.2),sin(0.2))]

struct ContinuousZernikeElementMode{T, P<:AbstractVector, M<:Int, J<:Int} <: Basis{T}
    points::P
    m::M
    j::J
end

function ContinuousZernikeElementMode{T}(points::AbstractVector, m::Int, j::Int) where {T}
    @assert length(points) == 2 && points[1] ≈ 0 && 0 < points[2]
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    ContinuousZernikeElementMode{T,typeof(points), Int, Int}(points, m, j)
end
ContinuousZernikeElementMode(points::AbstractVector, m::Int, j::Int) = ContinuousZernikeElementMode{Float64}(points, m, j)
ContinuousZernikeElementMode(m::Int, j::Int) = ContinuousZernikeElementMode([0.0; 1.0], m, j)

axes(Z::ContinuousZernikeElementMode) = (Inclusion(last(Z.points)*UnitDisk{eltype(Z)}()), oneto(∞))
==(P::ContinuousZernikeElementMode, Q::ContinuousZernikeElementMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j

function getindex(Z::ContinuousZernikeElementMode{T}, xy::StaticVector{2}, j::Int)::T where {T}
    p = Z.points
    α, β = convert(T, p[1]), convert(T, p[2])
    @assert α ≈ 0
    rθ = RadialCoordinate(xy)
    r̃ = affine(α.. β, 0.. 1)[rθ.r]
    xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
    Z.m == 0 && @assert Z.j == 1
    if j == 1
        Zernike{T}(0, 1)[xỹ, Block(1+Z.m)][Z.m+Z.j]
    else
        Weighted(Zernike{T}(0,1))[xỹ, Block(2j-3+Z.m)][Z.m+Z.j]
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

function _scale_fcn(f, r::T, xy::AbstractArray) where T
    rθ = RadialCoordinate(xy)
    r̃ = affine(0.. 1, 0.. r)[rθ.r]
    xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
    f(xỹ)
end

function ldiv(C::ContinuousZernikeElementMode{V}, f::AbstractQuasiVector) where V
    T = promote_type(V, eltype(f))
    Z = Zernike{T}(0,1)
    r = f.args[1].domain.radius
    # Need to take into account different scalings
    if r ≉  1
        fc(xy) = _scale_fcn(f.f, r, xy)
        x = axes(Z,1)
        f̃ = fc.(x)
    else
        f̃ = f
    end

    # Seems to cache on its own, no need to memoize unlike annulus
    c = Z \ f̃ # Zernike transform
    c̃ = paddeddata(c)
    N = size(c̃.matrix, 1) # degree
    
    # Restrict to relevant mode and add a column corresponding to the hat function.
    R = Z \ Weighted(Z) # Very fast and does not change with width of disk.

    # BUG! Random factor of 2, Issue raised: https://github.com/JuliaApproximation/MultivariateOrthogonalPolynomials.jl/issues/141 
    R̃ =  [[T[1]; Zeros{T}(∞)] R.ops[C.m+1]/2]

    # convert from Zernike(0,1) to hat + Bubble
    dat = R̃[1:N,1:N] \ c̃.matrix[:, 2*C.m + C.j]
    cfs = T[]
    pad(append!(cfs, dat), axes(C,2))
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
    M = [[T[ρ^2]; Zeros{T}(∞)] M]
    Vcat([T[ρ^2*(B.m+2)]; T[ρ^2]; Zeros{T}(∞)]', M)
end

###
# Gradient and L2 inner product of gradient
##
struct GradientContinuousZernikeElementMode{T, P<:AbstractVector, M<:Int, J<:Int}<:Basis{T}
    points::P
    m::M
    j::J
end

GradientContinuousZernikeElementMode{T}(points::AbstractVector, m::Int, j::Int) where {T} =  GradientContinuousZernikeElementMode{T,typeof(points), Int, Int}(points, m, j)
GradientContinuousZernikeElementMode(points::AbstractVector, m::Int, j::Int) =  GradientContinuousZernikeElementMode{Float64}(points, m, j)
GradientContinuousZernikeElementMode(m::Int, j::Int) =  GradientContinuousZernikeElementMode([0.0; 1.0], m, j)

axes(Z:: GradientContinuousZernikeElementMode) = (Inclusion(last(Z.points)*UnitDisk{eltype(Z)}()), oneto(∞))
==(P:: GradientContinuousZernikeElementMode, Q:: GradientContinuousZernikeElementMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j

@simplify function *(D::Derivative, C::ContinuousZernikeElementMode)
    GradientContinuousZernikeElementMode(C.points, C.m, C.j)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:GradientContinuousZernikeElementMode}, B::GradientContinuousZernikeElementMode)
    T = promote_type(eltype(A), eltype(B))
    Z = Zernike(0,1)
    D = Z \ (Laplacian(axes(Z,1))*Weighted(Z))

    Δ = -D.ops[B.m+1]
    cₘ = π*B.m*zerniker(B.m,B.m,0,1,one(T))^2 # = <Z^(0,1)_{m,m,j}, Z^(0,1)_{m,m,j}>_L^2

    Vcat([T[cₘ]; Zeros{T}(∞)]', [Zeros{T}(∞) Δ])
end

###
# Plotting
###

_angle(rθ::RadialCoordinate) = rθ.θ

function grid(C::ContinuousZernikeElementMode{T}, j::Int) where T
    Z = Zernike{T}(zero(T), one(T))
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
    Z = Zernike{T}(0,1)
    R = Z \ Weighted(Z) # Very fast and does not change with width of disk.
    R̃ =  [[T[1]; Zeros{T}(∞)] R.ops[m+1]/2]
    if c isa LazyArray
        c = paddeddata(c)
    end
    R̃[1:length(c), 1:length(c)] * c # coefficients for Zernike(0,1)
end

function plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{ContinuousZernikeElementMode, AbstractVector}}) where T
    C, c = u.args
    ρ = convert(T, last(C.points))
    m = C.m

    Z = Zernike{T}(0,1)
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
    FT = ZernikeITransform{T}(2N+C.m, 0, 1)
    g, FT * pad(F,axes(Z,2))[Block.(OneTo(2N+C.m))]
end