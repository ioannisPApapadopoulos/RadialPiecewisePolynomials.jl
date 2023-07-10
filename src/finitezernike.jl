# The Zernike (annular) basis on a single reference element and single Fourier mode
struct ZernikeBasisMode{T, P<:AbstractArray{T}} <: Basis{T}
    points::P
    a::Int
    b::Int
    m::Int
    j::Int
end

function ZernikeBasisMode(points::AbstractVector{T}, a::Int, b::Int, m::Int, j::Int) where T
    @assert length(points) == 2 && zero(T) < points[1] < points[2] ≤ one(T)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    ZernikeBasisMode{T, Vector{T}}(points, a, b, m, j)
end

# The Zernike (annular) basis on multiple elements and single Fourier mode
struct FiniteZernikeBasisMode{T, P<:AbstractArray{T}} <: Basis{T}
    N::Int
    points::P
    a::Int
    b::Int
    m::Int
    j::Int
end
function FiniteZernikeBasisMode(N::Int, points::AbstractVector{T}, a::Int, b::Int, m::Int, j::Int) where T
    @assert length(points) > 1 && points == sort(points)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    FiniteZernikeBasisMode{T, Vector{T}}(N, points, a, b, m, j)
end

# The Zernike (annular) basis on multiple elements and all Fourier modes
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
        Zs = [Zernike{T}(0,1); ZernikeAnnulus{T}.(ρs[2:end], a, b)]
    else
        Zs = ZernikeAnnulus{T}.(ρs, a, b)
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
    
    c = ModalTrav.(c)
    cs = Vector{T}[]
    
    ms = ((0:2N) .÷ 2)[2:end-1]
    Ms = ((N + 1 .- ms) .÷ 2); Ms[Ms .== 1] .= 2

    for i in 1:2N-1
        u = T[]
        for k in 1:K
            append!(u, c[k].matrix[1:Ms[i],i])
        end
        append!(cs, [u])
    end
    return cs
end

####
# L2 inner product matrices
####

@simplify function *(FT::QuasiAdjoint{<:Any,<:ContinuousZernikeAnnulusElementMode}, Z::ZernikeBasisMode)
    T = promote_type(eltype(FT), eltype(Z))
    F = FT.parent

    @assert F.points == Z.points && F.m == Z.m && F.j == Z.j

    points, a, b, m, j = Z.points, Z.a, Z.b, Z.m, Z.j
    α, β = first(points), last(points)
    ρ = α / β
    t = inv(one(T)-ρ^2)

    if a == 1 && b == 1
        # Contribution from the mass matrix of harmonic polynomial
        m₀ = convert(T,π) / ( t^(one(T) + m) )
        m₀ = m == 0 ? m₀ : m₀ / T(2)
        
    else
        error("L²-inner product not implemented for parameters Z.a = $a and Z.b = $b")
    end
end

@simplify function *(FT::QuasiAdjoint{<:Any,<:FiniteContinuousZernike}, Z::FiniteZernikeBasis)
    T = promote_type(eltype(FT), eltype(Z))
    F = FT.parent

    @assert F.N == Z.N && F.points == Z.points

    N, points, a, b = Z.N, Z.points, Z.a, Z.b


    if a == 1 && b == 1
        error("L²-inner product not implemented for parameters Z.a = $a and Z.b = $b")
    elseif a == 0 && b == 0
        error("L²-inner product not implemented for parameters Z.a = $a and Z.b = $b")
    else
        error("L²-inner product not implemented for parameters Z.a = $a and Z.b = $b")
    end
end



###
# Plotting
##
# This helper function takes the list of coefficient values from ldiv and converts them into 
# a 3-tensor of degree × Fourier mode × element.

function _bubble2disk_or_ann_all_modes(Z::FiniteZernikeBasis{T}, us::AbstractVector) where T
    points = T.(Z.points); K = length(points)-1
    N = Z.N;
    ms = ((0:2N) .÷ 2)[2:end-1]
    Ms = ((N + 1 .- ms) .÷ 2); Ms[Ms .== 1] .= 2

    Ñ = isodd(N) ? N : N+1
    Us = zeros(T,(Ñ+1)÷2,2Ñ-1,K)

    for i in 1:2N-1
        for k = 1:K 
            Us[1:Ms[i],i,k] = us[i][(k-1)*Ms[i]+1:k*Ms[i]]
        end
    end
    [], Us
end


function finite_plotvalues(Z::FiniteZernikeBasis{T}, us::AbstractVector) where T
    _, Ũs = _bubble2disk_or_ann_all_modes(Z, us)
    points = T.(Z.points); N = Z.N; K = length(points)-1
    Zs = Z.Zs
    θs=[]; rs=[]; vals = []   
    for k in 1:K
        if k == 1 && first(points) ≈ 0
            ρ = points[2]
            g = scalegrid(AlgebraicCurveOrthogonalPolynomials.grid(Zs[k], Block(2N)), ρ)
            FT = ZernikeITransform{T}(2N, Zs[k].a, Zs[k].b)
            val = FT * pad(ModalTrav(Ũs[:,:,k]),axes(Zs[k],2))[Block.(OneTo(2N))]
        else
            Za = Zs[k]
            α, β = points[k], points[k+1]
            # Scale the grid
            g = scalegrid(AlgebraicCurveOrthogonalPolynomials.grid(Za, Block(N)), α, β)
            # Use fast transforms for synthesis
            FT = ZernikeAnnulusITransform{T}(N, Za.a, Za.b, 0, Za.ρ)
            val = FT * pad(ModalTrav(Ũs[:,:,k]),axes(Za,2))[Block.(OneTo(N))]
        end
        (θ, r, val) = plot_helper(g, val)
        append!(θs, [θ]); append!(rs, [r]); append!(vals, [val])
    end
    return (θs, rs, vals)
end