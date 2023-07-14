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

function axes(Z::ZernikeBasisMode) 
    α = first(Z.points)
    if α ≈ 0
        (Inclusion(last(Z.points)*UnitDisk{eltype(Z)}()), oneto(∞))
    else
        (Inclusion(annulus(α, last(Z.points))), oneto(∞))
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

function axes(Z::FiniteZernikeBasisMode{T}) where T
    first(Z.points) ≈ 0 && return (Inclusion(last(Z.points)*UnitDisk{T}()), blockedrange(Fill(length(Z.points) - 1, Z.N-1)))
    (Inclusion(annulus(first(Z.points), last(Z.points))), blockedrange(Fill(length(Z.points) - 1, Z.N-1)))
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

function axes(Z::FiniteZernikeBasis{T}) where T
    first(Z.points) ≈ 0 && return (Inclusion(last(Z.points)*UnitDisk{T}()), oneto(Z.N*(length(Z.points)-1)))
    (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)))
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

_getFZs(Z::FiniteZernikeBasis{T}) where T = _getFZs(Z.N, Z.points, Z.a, Z.b)

####
# Transforms (analysis)
####
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

function _piece_element_matrix_zernike_basis(Ms, N::Int, m::Int, points::AbstractVector{T}) where T
    K = length(points)-1
    M = Hcat(Matrix{T}(Ms[1][1:N, 1:N]), zeros(T, N,(K-1)*N))

    if K > 1
        γs = _getγs(points, m)
        append!(γs, one(T))
        for k in 2:K
            M = Matrix(Vcat(M, Hcat(zeros(T, N-1, N+(k-2)*(N)), Ms[k][2:N, 1:N], zeros(T, N-1, (K-k)*(N)))))
        end

        i = first(points) ≈ 0 ? 1 : 2 # disk or annulus?
        M[i, 1:i] *= γs[1] # Convert the left-side hat function coefficients for continuity

        # Right-side of hat function interaction with basis in next element
        M[i, N+1:N+2] = Ms[2][1,1:2]

        b = min(N, 2)
        for k in 2:K-1
            # Convert left-side of hat function coefficients for continuity
            M[N+(k-2)*(N-1)+1, (k-1)*(N)+1:(k-1)*(N)+b] *= γs[k]
            # Right-side of hat function interaction with basis in next element
            M[N+(k-2)*(N-1)+1, (k)*(N)+1:(k)*(N)+b] = Ms[k+1][1,1:2]
        end
    end
    return M
end

@simplify function *(FT::QuasiAdjoint{<:Any,<:ContinuousZernikeAnnulusElementMode}, Z::ZernikeBasisMode)
    T = promote_type(eltype(FT), eltype(Z))
    F = FT.parent

    @assert F.points == Z.points && F.m == Z.m && F.j == Z.j

    points, a, b, m, j = Z.points, Z.a, Z.b, Z.m, Z.j
    α, β = first(points), last(points)
    ρ = α / β
    t = inv(one(T)-ρ^2)
    L₁₁, L₀₁, L₁₀ = F.L₁₁, F.L₀₁, F.L₁₀

    if a == 0 && b == 0
        # Contribution from the mass matrix of harmonic polynomial
        jw = _sum_semiclassicaljacobiweight(t,0,0,m)
        m₀ = convert(T,π) / ( t^(one(T) + m) ) * jw
        m₀ = m == 0 ? m₀ : m₀ / T(2)
        Vcat(Hcat(β^2*m₀*L₁₀[:,1], β^2*m₀*L₀₁[:,1])', β^2*m₀*L₁₁')
    else
        error("L²-inner product between ContinuousZernikeAnnulusElementMode and ZernikeBasisMode not implemented for parameters Z.a = $a and Z.b = $b")
    end
end

@simplify function *(FT::QuasiAdjoint{<:Any,<:ContinuousZernikeElementMode}, Z::ZernikeBasisMode)
    T = promote_type(eltype(FT), eltype(Z))
    F = FT.parent

    @assert F.points == Z.points && F.m == Z.m && F.j == Z.j

    points, a, b, m = Z.points, Z.a, Z.b, Z.m
    α, β = first(points), last(points)
    @assert α ≈ 0

    if a == 0 && b == 0
        L = Zernike(0) \ Weighted(Zernike(1))
        Vcat(Hcat(β^2, Zeros{T}(1,∞)), β^2*L.ops[m+1]')
    else
        error("L²-inner product between ContinuousZernikeElementMode and ZernikeBasisMode not implemented for parameters Z.a = $a and Z.b = $b")
    end
end

### Helper functions for building the ArrowheadMatrix
# Interaction of hats with lowest order Zernike
function _build_top_left_block(F::FiniteContinuousZernikeMode{T}, Z::FiniteZernikeBasisMode{T}, Ms, γs::AbstractArray{T}, p::T) where T
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
function _build_second_block(F::FiniteContinuousZernikeMode{T}, Z::FiniteZernikeBasisMode{T}, Ms, γs::AbstractArray{T}, p::T) where T
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
function _build_trailing_bubbles(F::FiniteContinuousZernikeMode{T}, Z::FiniteZernikeBasisMode{T}, Ms, N::Int, p::T) where T
    K = length(Ms)
    if p ≈ 0
        Mn = vcat([Ms[1][2:N-1,2:N]], [Ms[i][3:N, 2:N] for i in 2:K])
    else
        Mn = [Ms[i][3:N, 2:N] for i in 1:K]
    end
    return [BandedMatrix{T}(-1=>view(M, band(-1)), 0=>view(M, band(0)), 1=>view(M, band(1))) for M in Mn]
end

function _arrow_head_matrix(F::FiniteContinuousZernikeMode, Z::FiniteZernikeBasisMode, Ms, γs::AbstractArray{T}, N::Int, p::T) where T
    A = _build_top_left_block(F,Z,Ms, γs, p)
    B, C = _build_second_block(F,Z,Ms, γs,  p)
    D = _build_trailing_bubbles(F,Z,Ms, N,  p)
    ArrowheadMatrix{T}(A, B, C, D)
end

@simplify function *(FT::QuasiAdjoint{<:Any,<:FiniteContinuousZernikeMode}, Z::FiniteZernikeBasisMode)
    T = promote_type(eltype(FT), eltype(Z))
    F = FT.parent

    @assert F.N == Z.N && F.points == Z.points && F.m == Z.m && F.j == Z.j

    N, points, m = Z.N, T.(Z.points), Z.m
    K = length(points)-1
    Cs = _getCs(F)
    Zs = _getZs(Z)
    γs = _getγs(F)
    Ms = [C' * Z̃ for (C, Z̃) in zip(Cs, Zs)]

    # _piece_element_matrix_zernike_basis(Ms, N, m, points)
    _arrow_head_matrix(F, Z, Ms, γs, F.N, first(F.points))[Block.(1:N-1), :]

end

@simplify function *(FT::QuasiAdjoint{<:Any,<:FiniteContinuousZernike}, Z::FiniteZernikeBasis)
        T = promote_type(eltype(FT), eltype(Z))
        F = FT.parent
        @assert F.N == Z.N && F.points == Z.points

        N, points = Z.N, T.(Z.points);
        Fs = _getFs(N, points)
        Zs = _getFZs(N, points, Z.a, Z.b)

        [F̃' * Z̃ for (F̃, Z̃) in zip(Fs, Zs)]
end


###
# Plotting
##
# This helper function takes the list of coefficient values from ldiv and converts them into 
# a 3-tensor of degree × Fourier mode × element.

function _bubble2disk_or_ann_all_modes(Z::FiniteZernikeBasis{T}, us::AbstractVector) where T
    points = T.(Z.points); K = length(points)-1
    N = Z.N;
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

### Error collection
function inf_error(Z::FiniteZernikeBasis{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function) where T
    K = lastindex(Z.points)-1
    _inf_error(K, θs, rs, vals, u)
end