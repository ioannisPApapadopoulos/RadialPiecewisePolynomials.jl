"""
The ordering interlaces the functions of the same degree & Fourier mode.
For instance in a 2-element approximation there are two hat function per Fourier mode.
Let _{n,m,j} denote n = degree, (m,j) = Fourier mode.
The ordering is as follows

H^0_{0,0,1}
H^1_{0,0,1}
---------
H^0_{1,1,0}
H^1_{1,1,0}
H^0_{1,1,1}
H^1_{1,1,1}
--------
H^0_{2,2,0}
H^1_{2,2,0}
H^0_{2,2,1}
H^1_{2,2,1}
W^0_{0,0,1}
W^1_{0,0,1}
---------
.
.
"""

###
# Hat functions on the disk
###
struct ContinuousDiskHatPolynomials{T, P<:AbstractVector} <: Basis{T}
    points::P
end

ContinuousDiskHatPolynomials{T}(points::AbstractVector) where {T} =
    ContinuousDiskHatPolynomials{T,typeof(points)}(points)
ContinuousDiskHatPolynomials(points::AbstractVector) =
    ContinuousDiskHatPolynomials{Float64}(points)

axes(B::ContinuousDiskHatPolynomials) = (Inclusion(last(B.points)*UnitDisk{eltype(B)}()), blockedrange(Vcat(length(B.points)-1, Fill(2*(length(B.points) - 1), ∞))))
==(P::ContinuousDiskHatPolynomials, Q::ContinuousDiskHatPolynomials) = P.points == Q.points

function getindex(H::ContinuousDiskHatPolynomials{T}, xy::StaticVector{2}, j::Int)::T where {T}
    Kk =  findblockindex(axes(H, 2), j)
    K, k = block(Kk), blockindex(Kk)
    p = H.points
    rθ = RadialCoordinate(xy)
    b = searchsortedlast(p, rθ.r)
    α, β = convert(T, p[b]), convert(T, p[b+1])  

    if b - k % (length(p)-1) == 1
        b == 1 && return zero(T)
        r̃ = affine(α.. β, α.. one(T))[rθ.r]
        xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
        c = inv((1-α^2)*α^(Int(K)-1))
        if K == Block(1)
            c*Weighted(ZernikeAnnulus{T}(α,0,1))[xỹ, K][end]
        else
            c*Weighted(ZernikeAnnulus{T}(α,0,1))[xỹ, K][end-Int(k==b-1)]
        end
    elseif (b - k) % (length(p)-1) == 0
        if b == 1
            @assert α == 0
            r̃ = affine(α.. β, zero(T).. one(T))[rθ.r]
            xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
            if K == Block(1)
                Zernike{T}(0,1)[xỹ, K][end] / Zernike(0,1)[SVector(1.0,0), K][end]
            else
                Zernike{T}(0,1)[xỹ, K][end-Int(k==b)] / Zernike(0,1)[SVector(1.0,0), K][end]
            end
        else
            r̃ = affine(α.. β, α.. one(T))[rθ.r]
            xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
            c = inv(1-α^2)
            if K == Block(1)
                c*Weighted(ZernikeAnnulus{T}(α,1,0))[xỹ, K][end]
            else
                c*Weighted(ZernikeAnnulus{T}(α,1,0))[xỹ, K][end-Int(k==b)]
            end
        end
    else
        return zero(T)
    end
end

###
# Continuous piecewise OPs on the disk
###
struct ContinuousPiecewiseZernike{T, P<:AbstractVector} <: Basis{T}
    points::P
end

ContinuousPiecewiseZernike{T}(points::AbstractVector) where {T} =
    ContinuousPiecewiseZernike{T,typeof(points)}(points)
ContinuousPiecewiseZernike(points::AbstractVector) =
    ContinuousPiecewiseZernike{Float64}(points)

axes(B::ContinuousPiecewiseZernike) = (Inclusion(last(B.points)*UnitDisk{eltype(B)}()), blockedrange((length(B.points) - 1) .* oneto(∞)))

==(P::ContinuousPiecewiseZernike, Q::ContinuousPiecewiseZernike) = P.points == Q.points


function getindex(Z::ContinuousPiecewiseZernike{T}, xy::StaticVector{2}, j::Int)::T where {T}
    Kk =  findblockindex(axes(Z, 2), j)
    K, k = block(Kk), blockindex(Kk)
    p = Z.points; skip = 2*(length(p)-1);

    # If in first "skip" number of functions, then it is a hat function
    if k ≤ skip
        ContinuousDiskHatPolynomials{T}(Z.points)[xy, K][k]
    else
        k̃ = k - skip
        b = searchsortedlast(p, RadialCoordinate(xy).r)
        if (k̃ - b) % (length(p)-1) == 0
            α, β = convert(T, p[b]), convert(T, p[b+1])
            rθ = RadialCoordinate(xy)
            if b == 1 # First element is Zernike
                @assert α == 0
                r̃ = affine(α.. β, 0.. 1)[rθ.r]
                xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
                Weighted(Zernike(0,1))[xỹ, k̃]
            else # Any other elements are ZernikeAnnulus
                r̃ = affine(α.. β, α.. 1)[rθ.r]
                xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
                Weighted(ZernikeAnnulus(α,1,1))[xỹ, k̃]
            end
        else
            zero(T)
        end
    end
end

function extract_piecewise_coefficients(Z, c)
    p = Z.points
    n = length(p)-1
    c = PseudoBlockVector(c, n .*oneto(∞))
    cw = c[Block.(3:end)]


end


struct ContinuousZernikeElementMode{T, P<:AbstractVector, M<:Int, J<:Int} <: Basis{T}
    points::P
    m::M
    j::J
end

ContinuousZernikeElementMode{T}(points::AbstractVector, m::Int, j::Int) where {T} = ContinuousZernikeElementMode{T,typeof(points), Int, Int}(points, m, j)
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
        Zernike{T}(0, 1)[xỹ, Block(2j-1+Z.m)][Z.m+Z.j]
    else
        Weighted(Zernike{T}(0,1))[xỹ, Block(2j-3+Z.m)][Z.m+Z.j]
    end
end

# ClassicalOrthogonalPolynomials.checkpoints(d::DomainSets.GenericBall{SVector{2, T}, :closed, T}) where T = [SVector{2,T}(cos(0.1),sin(0.1)), SVector{2,T}(cos(0.2),sin(0.2))]

# function grid(C::ContinuousZernikeElementMode{T}, n::Integer) where T
#     N = 2(n-1) # degree
#     RadialCoordinate.(sinpi.((N .-(0:N-1) .- one(T)/2) ./ (2N)), 0.)
# end

function fa(f, r, xy)
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
        fc(xy) = fa(f.f, r, xy)
        x = axes(Z,1)
        f̃ = fc.(x)
    else
        f̃ = f
    end

    c = Z\f̃ # Zernike transform
    c̃ = paddeddata(c)
    N = size(c̃.matrix, 1) # degree
    
    # Restrict to relevant mode and add a column corresponding to the hat function.
    R = Z \ Weighted(Z)
    mode = 2*C.m + C.j

    # BUG! Random factor of 2, Issue raised: https://github.com/JuliaApproximation/MultivariateOrthogonalPolynomials.jl/issues/141 
    R̃ =  [[T[1]; Zeros{T}(∞)] R.ops[mode]/2]

    # convert from Zernike(0,1) to hat + Bubble
    dat = R̃[1:N,1:N] \ c̃.matrix[:, mode]
    cfs = T[]
    pad(append!(cfs, dat), axes(C,2))
end

# function getindex(Z::ContinuousZernikeElementMode{T}, xy::StaticVector{2}, j::Int)::T where {T}
#     Kk =  findblockindex(axes(Z, 2), j)
#     K, k = block(Kk), blockindex(Kk)
#     p = Z.points; 
#     @assert length(p)==2
#     α, β = convert(T, p[1]), convert(T, p[2])
#     @assert α ≈ 0
#     rθ = RadialCoordinate(xy)
#     r̃ = affine(α.. β, 0.. 1)[rθ.r]
#     xỹ = SVector(r̃*cos(rθ.θ), r̃*sin(rθ.θ))
#     # Index of hat function
#     if (Int(K) - 2) - k < 0
#         Zernike{T}(0,1)[xỹ, K][k] / Zernike{T}(0,1)[SVector(one(T),zero(T)), K][end]
#     else
#         Weighted(Zernike{T}(0,1))[xỹ, K-2][k]
#     end
# end

# function \(A::ContinuousZernikeElementMode{V}, B::Zernike{T}) where {T,V}
#     TV = promote_type(T,V)
#     @assert B.a == B.b == 0

#     L₁ = Weighted.(SemiclassicalJacobi{real(TV)}.(t,zero(TV),zero(TV),zero(TV):∞)) .\ Weighted.(SemiclassicalJacobi{real(TV)}.(t,one(TV),one(TV),zero(TV):∞))
#     L₂ = SemiclassicalJacobi{real(TV)}.(t,one(TV),one(TV),zero(TV):∞) .\ SemiclassicalJacobi{real(TV)}.(t,zero(TV),zero(TV),zero(TV):∞)

#     L = (one(TV)-ρ^2)^2 .* (L₂ .* L₁)
#     ModalInterlace{TV}(L, (ℵ₀,ℵ₀), (4, 4))
# end
