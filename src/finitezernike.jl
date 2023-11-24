# The Zernike (annular) basis on a single reference element and single Fourier mode
struct ZernikeBasisMode{T, P<:AbstractArray{T}} <: Basis{T}
    points::P
    a::Int
    b::Int
    m::Int
    j::Int
end

function ZernikeBasisMode(points::AbstractVector{T}, a::Int, b::Int, m::Int, j::Int) where T
    @assert length(points) == 2 && zero(T) ≤ points[1] < points[2] ≤ one(T)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    ZernikeBasisMode{T, Vector{T}}(points, a, b, m, j)
end

function axes(Ψ::ZernikeBasisMode)
    α = first(Ψ.points)
    if α ≈ 0
        (Inclusion(last(Ψ.points)*UnitDisk{eltype(Ψ)}()), oneto(∞))
    else
        (Inclusion(annulus(α, last(Ψ.points))), oneto(∞))
    end
end

==(P::ZernikeBasisMode, Q::ZernikeBasisMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j && P.a == Q.a && P.b == Q.b

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

function axes(Ψ::FiniteZernikeBasisMode{T}) where T
    first(Ψ.points) ≈ 0 && return (Inclusion(last(Ψ.points)*UnitDisk{T}()), blockedrange(Fill(length(Ψ.points) - 1, Ψ.N-1)))
    (Inclusion(annulus(first(Ψ.points), last(Ψ.points))), blockedrange(Fill(length(Ψ.points) - 1, Ψ.N-1)))
end
==(P::FiniteZernikeBasisMode, Q::FiniteZernikeBasisMode) = P.N == Q.N && P.points == Q.points && P.m == Q.m && P.j == Q.j && P.a == Q.a && P.b == Q.b


function _getZs(points::AbstractVector{T}, a::Int, b::Int, m::Int, j::Int) where T
    K = length(points)-1
    return [ZernikeBasisMode([points[k]; points[k+1]], a, b, m, j) for k in 1:K]
end

function _getZs(F::FiniteZernikeBasisMode)
    points, a, b, m, j = F.points, F.a, F.b, F.m, F.j
    _getZs(points, a, b, m, j)
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
        Zs = [Zernike{T}(a,b); ZernikeAnnulus{T}.(ρs[2:end], a, b)]
    else
        Zs = ZernikeAnnulus{T}.(ρs, a, b)
    end
    FiniteZernikeBasis{T}(N, points, a, b, Zs)
end

function axes(Ψ::FiniteZernikeBasis{T}) where T
    first(Ψ.points) ≈ 0 && return (Inclusion(last(Ψ.points)*UnitDisk{T}()), oneto(Ψ.N*(length(Ψ.points)-1)))
    (Inclusion(annulus(first(Ψ.points), last(Ψ.points))), oneto(Ψ.N*(length(Ψ.points)-1)))
end
==(P::FiniteZernikeBasis, Q::FiniteZernikeBasis) = P.N == Q.N && P.points == Q.points

function _getFZs(N::Int, points::AbstractArray{T}, a::Int, b::Int) where T
    # Ordered list of Fourier modes (ms, js) and correct length for each Fourier mode Ms.
    Ms, ms, js = _getMs_ms_js(N)

    # Loop over the Fourier modes
    Fs = []
    for (M, m, j) in zip(Ms, ms, js)
        # Construct the structs for each Fourier mode seperately
        append!(Fs, [FiniteZernikeBasisMode(M, points, a, b, m, j)])
    end
    return Fs
end

_getFZs(Ψ::FiniteZernikeBasis{T}) where T = _getFZs(Ψ.N, Ψ.points, Ψ.a, Ψ.b)

####
# Transforms (analysis)
####
function ldiv(Ψ::FiniteZernikeBasis{T}, f::AbstractQuasiVector) where T

    N, points, Zs = Ψ.N, Ψ.points, Ψ.Zs
    
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
    cs = []
    
    Ms, ms, js = _getMs_ms_js(N)

    for i in 1:2N-1
        u = zeros(T, Ms[i], K)
        for k in 1:K
            u[:, k] = c[k].matrix[1:Ms[i],i]
        end
        append!(cs, [pad(vec(u'), blockedrange(Fill(K, Ms[i])))])
    end
    return cs
end

####
# L2 inner product matrices
####

@simplify function *(FT::QuasiAdjoint{<:Any,<:ContinuousZernikeAnnulusElementMode}, Ψ::ZernikeBasisMode)
    gram_matrix(FT.parent, Ψ)
end
function gram_matrix(C::ContinuousZernikeAnnulusElementMode, Ψ::ZernikeBasisMode)
    T = promote_type(eltype(C), eltype(Ψ))

    @assert C.points == Ψ.points && C.m == Ψ.m && C.j == Ψ.j

    points, a, b, m = Ψ.points, Ψ.a, Ψ.b, Ψ.m
    α, β = first(points), last(points)
    ρ = α / β
    t = inv(one(T)-ρ^2)

    if a == 0 && b == 0
        m₀ = _mass_m₀(C,m,t)
        ApplyArray(*,Diagonal(Fill(β^2*m₀,∞)), C.R')
    else
        error("L²-inner product between ContinuousZernikeAnnulusElementMode and ZernikeBasisMode not implemented for parameters Ψ.a = $a and Ψ.b = $b")
    end
end

@simplify function *(FT::QuasiAdjoint{<:Any,<:ContinuousZernikeElementMode}, Ψ::ZernikeBasisMode)
    gram_matrix(FT.parent, Ψ)
end
function gram_matrix(C::ContinuousZernikeElementMode, Ψ::ZernikeBasisMode)
    T = promote_type(eltype(C), eltype(Ψ))

    @assert C.points == Ψ.points && C.m == Ψ.m && C.j == Ψ.j

    points, a, b, m = Ψ.points, Ψ.a, Ψ.b, Ψ.m
    α, β = first(points), last(points)
    @assert α ≈ 0

    if a == 0 && b == 0
        R = Zernike(0) \ Weighted(Zernike(1))
        Vcat(Hcat(β^2, Zeros{T}(1,∞)), β^2*R.ops[m+1]')
    else
        error("L²-inner product between ContinuousZernikeElementMode and ZernikeBasisMode not implemented for parameters Ψ.a = $a and Ψ.b = $b")
    end
end

### Helper functions for building the ArrowheadMatrix
# Interaction of hats with lowest order Zernike
function _build_top_left_block(F::FiniteContinuousZernikeMode{T}, Ψ::FiniteZernikeBasisMode{T}, Ms, γs::AbstractArray{T}, p::T) where T
    K = length(Ms)
    if p ≈ 0
        dv, ev = zeros(T, K), zeros(T,K-1)
        γs = vcat(γs, one(T))

        dv[1] = Ms[1][1,1] * γs[1]
        for i in 2:K dv[i] = Ms[i][2,1] * γs[i] end
        for i in 2:K ev[i-1] = Ms[i][1,1] end
        return BandedMatrix{T}(0=>dv, 1=>ev)
    else

        dv, ev = zeros(T, K), zeros(T,K)
        γs = vcat(γs, one(T))

        for i in 1:K dv[i] = Ms[i][1,1] end
        for i in 1:K ev[i] = Ms[i][2,1] * γs[i] end
        return BandedMatrix{T}((0=>dv, -1=>ev), (K+1, K))
    end

end

# Interaction of hats with next lower order Zernike
function _build_second_block(F::FiniteContinuousZernikeMode{T}, Ψ::FiniteZernikeBasisMode{T}, Ms, γs::AbstractArray{T}, p::T) where T
    K = length(Ms)
    if p ≈ 0
        dv, ev = zeros(T, K), zeros(T,K-1)
        γs = vcat(γs, one(T))

        dv[1] = Ms[1][1,2] * γs[1]
        for i in 2:K dv[i] = Ms[i][2,2] * γs[i] end
        for i in 2:K ev[i-1] = Ms[i][1,2] end

        bdv = zeros(T, K)
        bdv[1] = Ms[1][2,1]
        for i in 2:K bdv[i] = Ms[i][3,1] end

        H = [BandedMatrix{T}(0=>dv, 1=>ev), Zeros{T}(K, K)]
        B = [BandedMatrix{T}(0=>bdv), Zeros{T}(K, K)]
        return (H, B)
    else
        dv, ev = zeros(T, K), zeros(T,K)
        γs = vcat(γs, one(T))

        for i in 1:K dv[i] = Ms[i][1,2] end
        for i in 1:K ev[i] = Ms[i][2,2] * γs[i] end

        bdv = zeros(T, K)
        for i in 1:K bdv[i] = Ms[i][3,1] end

        H = [BandedMatrix{T}((0=>dv, -1=>ev), (K+1, K)), Zeros{T}(K+1, K)]
        B = [BandedMatrix{T}(0=>bdv), Zeros{T}(K, K)]
        return (H, B)
    end

end

# Interaction of the bubbles with Zernike
function _build_trailing_bubbles(F::FiniteContinuousZernikeMode{T}, Ψ::FiniteZernikeBasisMode{T}, Ms, N::Int, p::T) where T
    K = length(Ms)
    if p ≈ 0
        # Mn = vcat([Ms[1][2:N-1,2:N]], [Ms[i][3:N, 2:N] for i in 2:K])
        Mn = vcat([[i=>view(Ms[1], band(i))[2:N-1-j] for (i, j) in zip(-1:1, [1,0,0])]], 
                [[-1=>view(Ms[k], band(-2))[2:N-2], 0=>view(Ms[k], band(-1))[2:N-1], 1=>view(Ms[k], band(0))[3:N]] for k in 2:K])
    else
        # Mn = [Ms[i][3:N, 2:N] for i in 1:K]
        # Mn = [[i=>view(Ms[k], band(i))[3:N-j] for (i, j) in zip(-1:1, [1,0,0])] for k in 1:K]
        Mn = [[-1=>view(Ms[k], band(-2))[2:N-2], 0=>view(Ms[k], band(-1))[2:N-1], 1=>view(Ms[k], band(0))[3:N]] for k in 1:K]
    end
    # return [BandedMatrix{T}(-1=>view(M, band(-1)), 0=>view(M, band(0)), 1=>view(M, band(1))) for M in Mn]
    return [BandedMatrix{T}(M...) for M in Mn]
end

function _arrow_head_matrix(F::FiniteContinuousZernikeMode, Ψ::FiniteZernikeBasisMode, Ms, γs::AbstractArray{T}, N::Int, p::T) where T
    A = _build_top_left_block(F,Ψ,Ms, γs, p)
    B, C = _build_second_block(F,Ψ,Ms, γs,  p)
    D = _build_trailing_bubbles(F,Ψ,Ms, N,  p)
    ArrowheadMatrix{T}(A, B, C, D)
end

@simplify function *(FT::QuasiAdjoint{<:Any,<:FiniteContinuousZernikeMode}, Ψ::FiniteZernikeBasisMode)
    gram_matrix(FT.parent, Ψ)
end
function gram_matrix(F::FiniteContinuousZernikeMode, Ψ::FiniteZernikeBasisMode)
    T = promote_type(eltype(F), eltype(Ψ))

    @assert F.N == Ψ.N && F.points == Ψ.points && F.m == Ψ.m && F.j == Ψ.j

    N, points, m = Ψ.N, T.(Ψ.points), Ψ.m
    K = length(points)-1
    Cs = _getCs(F)
    Zs = _getZs(Ψ)
    γs = _getγs(F)
    Ms = [gram_matrix(C, Z̃) for (C, Z̃) in zip(Cs, Zs)]

    _arrow_head_matrix(F, Ψ, Ms, γs, F.N, first(F.points))[Block.(1:N-1), :]

end

@simplify function *(FT::QuasiAdjoint{<:Any,<:FiniteContinuousZernike}, Ψ::FiniteZernikeBasis)
        T = promote_type(eltype(FT), eltype(Ψ))
        F = FT.parent
        @assert F.N == Ψ.N && F.points == Ψ.points

        N, points = Ψ.N, T.(Ψ.points);
        Fs = _getFs(N, points)
        Zs = _getFZs(N, points, Ψ.a, Ψ.b)
        [gram_matrix(F̃, Z̃) for (F̃, Z̃) in zip(Fs, Zs)]
end


###
# Plotting
##
# This helper function takes the list of coefficient values from ldiv and converts them into 
# a 3-tensor of degree × Fourier mode × element.

function _bubble2disk_or_ann_all_modes(Ψ::FiniteZernikeBasis{T}, us::AbstractVector) where T
    points = T.(Ψ.points); K = length(points)-1
    N = Ψ.N;
    Ms, _, _ = _getMs_ms_js(N)

    Ñ = isodd(N) ? N : N+1
    Us = zeros(T,(Ñ+1)÷2,2Ñ-1,K)

    for i in 1:2N-1
        for k = 1:K 
            Us[1:Ms[i],i,k] = us[i][k:K:end]
        end
    end
    [], Us
end


function finite_plotvalues(Ψ::FiniteZernikeBasis{T}, us::AbstractVector; N=0, K=0) where T
    _, Ũs = _bubble2disk_or_ann_all_modes(Ψ, us)
    points= T.(Ψ.points)
    K = K==0 ? lastindex(Ψ.points)-1 : K
    N = N == 0 ? Ψ.N : N
    Zs = Ψ.Zs
    θs=[]; rs=[]; vals = []   
    for k in 1:K
        if k == 1 && first(points) ≈ 0
            ρ = points[2]
            g = scalegrid(AnnuliOrthogonalPolynomials.grid(Zs[k], Block(N)), ρ)
            FT = ZernikeITransform{T}(N, Zs[k].a, Zs[k].b)
            val = FT * pad(ModalTrav(Ũs[:,:,k]),axes(Zs[k],2))[Block.(OneTo(N))]
        else
            Za = Zs[k]
            α, β = points[k], points[k+1]
            # Scale the grid
            g = scalegrid(AnnuliOrthogonalPolynomials.grid(Za, Block(N)), α, β)
            # Use fast transforms for synthesis
            FT = ZernikeAnnulusITransform{T}(N, Za.a, Za.b, 0, Za.ρ)
            val = FT * pad(ModalTrav(Ũs[:,:,k]),axes(Za,2))[Block.(OneTo(N))]
        end
        (θ, r, val) = plot_helper(g, val)
        append!(θs, [θ]); append!(rs, [r]); append!(vals, [val])
    end
    return (θs, rs, vals)
end

### Error collection
function inf_error(Ψ::FiniteZernikeBasis{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function; K=0) where T
    K = K==0 ? lastindex(Ψ.points)-1 : K
    _inf_error(K, θs, rs, vals, u)
end