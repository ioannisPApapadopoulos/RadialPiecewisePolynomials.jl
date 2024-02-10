struct FiniteContinuousZernikeMode{T} <: Basis{T}
    N::Int
    points::AbstractVector{T}
    m::Int
    j::Int
    R::Tuple{Vararg{AbstractMatrix{T}}}
    D::Tuple{Vararg{AbstractMatrix{T}}}
    normalize_constants::AbstractVector{<:AbstractVector{<:T}}
    same_ρs::Bool
    b::Int # Should remove once adaptive expansion has been figured out.
end

function FiniteContinuousZernikeMode(N::Int, points::AbstractVector{T}, m::Int, j::Int, R, D, normalize_constants::AbstractVector{<:AbstractVector{<:T}}; same_ρs::Bool) where T
    @assert points == sort(points)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    K = first(points) ≈ 0 ? length(points)-2 : length(points) - 1
    @assert length(R) == length(D) == (same_ρs ? 1 : K)
    # @assert length(normalize_constants) ≥ 2
    FiniteContinuousZernikeMode{T}(N, points, m, j, R, D, normalize_constants, same_ρs, m+2N)
end

function FiniteContinuousZernikeMode(N::Int, points::AbstractVector{T}, m::Int, j::Int; same_ρs::Bool=false) where {T}
    K = length(points)-1
    κ = first(points[1]) ≈ 0 ? 2 : 1
    ρs = []
    for k = κ:length(points)-1
        α, β = convert(T, first(points[k])), convert(T, last(points[k+1]))
        append!(ρs, [α / β])
    end

    ts = inv.(one(T) .- ρs.^2)
    R = NTuple{K+1-κ,AbstractMatrix}(_ann2element_via_raising.(ts, m))
    normalize_constants = [[_sum_semiclassicaljacobiweight(t, a, a, m) for a in 1:-1:0] for t in ts]

    Z = ZernikeAnnulus{T}.(ρs,1,1)
    D = (Z .\ (Laplacian.(axes.(Z,1)).*Weighted.(Z)))
    D = NTuple{K+1-κ, AbstractMatrix}([Ds.ops[m+1] for Ds in D])

    FiniteContinuousZernikeMode(N, points, m, j, R, D, normalize_constants, same_ρs=same_ρs)
end

function axes(Z::FiniteContinuousZernikeMode{T}) where T
    first(Z.points) ≈ 0 && return (Inclusion(last(Z.points)*UnitDisk{T}()), blockedrange(Vcat(length(Z.points)-1, Fill(length(Z.points) - 1, Z.N-2))))
    # (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
    (Inclusion(annulus(first(Z.points), last(Z.points))), blockedrange(Vcat(length(Z.points), Fill(length(Z.points) - 1, Z.N-2))))
end
==(P::FiniteContinuousZernikeMode, Q::FiniteContinuousZernikeMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j && P.b == Q.b

function show(io::IO, F::FiniteContinuousZernikeMode)
    N, points, m, j = F.N, F.points, F.m, F.j
    print(io, "FiniteContinuousZernikeMode: N=$N, points=$points, m=$m, j=$j.")
end

function _getCs(points::AbstractVector{T}, m::Int, j::Int, b::Int, R, D, normalize_constants, same_ρs::Bool) where T
    K = length(points)-1
    if same_ρs
        first(points) > 0 && return [ContinuousZernikeAnnulusElementMode([points[k]; points[k+1]], m, j, R[1], D[1], normalize_constants[1],  b) for k in 1:K]
        return append!(Any[ContinuousZernikeElementMode([points[1]; points[2]], m, j)], [ContinuousZernikeAnnulusElementMode([points[k]; points[k+1]], m, j, R[1], D[1], normalize_constants[1], b) for k in 2:K])
    else
        first(points) > 0 && return [ContinuousZernikeAnnulusElementMode([points[k]; points[k+1]], m, j, R[k], D[k], normalize_constants[k], b) for k in 1:K]
        return append!(Any[ContinuousZernikeElementMode([points[1]; points[2]], m, j)], [ContinuousZernikeAnnulusElementMode([points[k]; points[k+1]], m, j, R[k-1], D[k-1], normalize_constants[k-1], b) for k in 2:K])
    end
end

function _getCs(F::FiniteContinuousZernikeMode)
    points, m, j, b = F.points, F.m, F.j, F.b
    R, D, normalize_constants, same_ρs = F.R, F.D, F.normalize_constants, F.same_ρs
    _getCs(points, m, j, b, R, D, normalize_constants, same_ρs)
end

function _getγs(points::AbstractArray{T}, m::Int) where T
    K = length(points)-1
    first(points) > 0 && return [(one(T)-(points[k+1]/points[k+2])^2)*(points[k+1]/points[k+2])^m / (one(T)-(points[k]/points[k+1])^2) for k in 1:K-1]
    K == 1 && return T[]
    γ = [(one(T)-(points[k+1]/points[k+2])^2)*(points[k+1]/points[k+2])^m / (one(T)-(points[k]/points[k+1])^2) for k in 2:K-1]
    return append!([(one(T)-(points[2]/points[3])^2)*(points[2]/points[3])^m / (sqrt(convert(T,2)^(m+2-iszero(m))/π) * normalizedjacobip(0, 0, m, 1.0))],γ)
end
_getγs(F::FiniteContinuousZernikeMode{T}) where T = _getγs(F.points, F.m)

# function getindex(F::FiniteContinuousZernikeMode{T}, xy::StaticVector{2}, j::Int)::T where {T}
#     points = T.(F.points); K = length(points)-1
#     N = F.N; m = F.m; j = F.j;
#     Cs = _getCs(points, m, j, F.b)
#     γs = _getγs(points, m)
#     Ns = append!([N], [N+k*(N-1) for k in 2:K])

#     rθ = RadialCoordinate(xy)
#     b = searchsortedlast(points, rθ.r)
#     k = searchsortedlast(j, Ns)
#     J = j - Ns[k]

#     if J == 1
#     if b == k


#     else
#         return zero(T)
#     end
# end


function ldiv(F::FiniteContinuousZernikeMode{T}, f::AbstractQuasiVector) where T
    N = F.N
    points = T.(F.points); K = length(points)-1
    Cs = _getCs(F)
    fs = [C \ f.f.(axes(C, 1)) for C in Cs]

    bubbles = zeros(T, N-2, K)
    if first(points) ≈ 0
        if K == 1
            hats = [fs[1][1]]
        else
            hats = vcat([fs[i][1] for i in 2:K-1], fs[end][1:2])
        end
        bubbles[:,1] = fs[1][2:N-1]
        for i in 2:K bubbles[:,i] = fs[i][3:N] end
    else
        hats = vcat([fs[i][1] for i in 1:K-1], fs[end][1:2])
        for i in 1:K bubbles[:,i] = fs[i][3:N] end
    end

    pad(append!(hats, vec(bubbles')), axes(F,2))
end


###
# L2 inner product
###

# Interaction of hats with themselves and other hats
function _build_top_left_block(F::FiniteContinuousZernikeMode{T}, Ms, γs::AbstractArray{T}, p::T) where T
    K = length(Ms)
    if p ≈ 0
        if K > 1
            a = [Ms[1][1,1]]
            for i in 2:K
                append!(a, [Ms[i][1,1]; Ms[i][2,2]])
            end

            dv = zeros(T, K)
            dv[1] = a[1]*γs[1]^2 + a[2];
            dv[end] = a[end];

            for i in 1:K-2 dv[i+1] = a[2i+1]*γs[i+1]^2 + a[2i+2] end

            ev = zeros(T, K-1)
            γs = vcat(γs, one(T))

            for i in 2:K ev[i-1] = Ms[i][1,2] * γs[i] end
        else
            dv = [Ms[1][1,1]]
        end
    else
        a = []
        for i in 1:K
            append!(a, [Ms[i][1,1]; Ms[i][2,2]])
        end

        dv = zeros(T, K+1)
        dv[1] = a[1]; dv[end] = a[end];
        ev = zeros(T, K)
        γs = vcat(γs, one(T))

        for i in 1:K-1 dv[i+1] = a[2i]*γs[i]^2 + a[2i+1] end
        for i in 1:K ev[i] = Ms[i][1,2] * γs[i] end

    end

    K == 1 && return Symmetric(BandedMatrix{T}(0=>dv))
    return Symmetric(BandedMatrix{T}(0=>dv, 1=>ev))
end

# Interaction of the hats with the bubbles
function _build_second_block(F::FiniteContinuousZernikeMode{T}, Ms, γs::AbstractArray{T}, bs::Int, p::T) where T
    K = length(Ms)
    γs = vcat(γs, one(T))
    dv, ev = [], []

    for j in 1:bs
        append!(dv, [zeros(T, K)])
        if p ≈ 0
            append!(ev, [zeros(T, K-1)])
            dv[j][1] = Ms[1][1,j+1] * γs[1]
            if K > 1
                ev[j][1] = Ms[2][1,j+2]
                for i in 2:K-1
                    dv[j][i] = Ms[i][2,j+2] * γs[i]
                    ev[j][i] = Ms[i+1][1,j+2]
                end
                dv[j][K] = Ms[K][2,j+2]
            end
        else
            append!(ev, [zeros(T, K)])
            for i in 1:K
                dv[j][i] = Ms[i][1,j+2]
                ev[j][i] = Ms[i][2,j+2] * γs[i]
            end
        end
    end
    if p ≈ 0
        K == 1 && return Tuple([BandedMatrix{T}(0=>dv[j]) for j in 1:bs])
        return Tuple([BandedMatrix{T}(0=>dv[j], 1=>ev[j]) for j in 1:bs])
    else
        return Tuple([BandedMatrix{T}((0=>dv[j], -1=>ev[j]), (K+1, K)) for j in 1:bs])
    end
end

# Interaction of the bubbles with themselves and other bubbles
function _build_trailing_bubbles(F::FiniteContinuousZernikeMode{T}, Ms, N::Int, bs::Int, p::T) where T
    K = length(Ms)
    if p ≈ 0
        # Mn = vcat([Ms[1][2:N-1,2:N-1]], [Ms[i][3:N, 3:N] for i in 2:K])
        # Mn = vcat([Ms[1][2:N-1,2:N-1]], [reshape(view(Ms[i],3:N, 3:N)[:], N-2, N-2) for i in 2:K])
        Mn = vcat([[i=>view(Ms[1], band(i))[2:N-1-i] for i in 0:bs]], [[i=>view(Ms[k], band(i))[3:N-i] for i in 0:bs] for k in 2:K])
    else
        # Mn = [Ms[i][3:N, 3:N] for i in 1:K]
        # Mn = [reshape(view(Ms[i],3:N, 3:N)[:], N-2, N-2) for i in 1:K]
        Mn = [[i=>view(Ms[k], band(i))[3:N-i] for i in 0:bs] for k in 1:K]
    end
    # return [Symmetric(BandedMatrix{T}([i=>view(M, band(i))[:] for i in 0:bs]...)) for M in Mn]
    return [Symmetric(BandedMatrix{T}(M...)) for M in Mn]
end

function _arrow_head_matrix(F::FiniteContinuousZernikeMode, Ms, γs::AbstractArray{T}, N::Int, bs::Int, p::T) where T
    A = _build_top_left_block(F,Ms, γs, p)
    B = _build_second_block(F,Ms, γs, bs, p)
    D = _build_trailing_bubbles(F,Ms, N, bs, p)
    Symmetric(BBBArrowheadMatrix{T}(A, B, (), D))
end


@simplify function *(A::QuasiAdjoint{<:Any,<:FiniteContinuousZernikeMode}, B::FiniteContinuousZernikeMode)
    @assert A' == B
    mass_matrix(B)
end

function mass_matrix(B::FiniteContinuousZernikeMode)
    Cs = _getCs(B)

    if B.same_ρs
        if first(B.points) ≈ 0
            Md = mass_matrix(Cs[1])
            M = mass_matrix(Cs[2])
            Ms = append!(Any[Md], [ApplyArray(*,Diagonal(Fill((B.points[k]/B.points[3])^2 ,∞)),M) for k = 3:length(B.points)])
        else
            M = mass_matrix(Cs[1])
            Ms = [ApplyArray(*,Diagonal(Fill((B.points[k]/B.points[2])^2 ,∞)),M) for k = 2:length(B.points)]
        end
    else
        Ms = mass_matrix.(Cs)
    end

    γs = _getγs(B)

    B.N < 4 ? _arrow_head_matrix(B, Ms, γs, B.N, 1, first(B.points)) : _arrow_head_matrix(B, Ms, γs, B.N, 2, first(B.points))
end

###
# Assembly
###
@simplify function *(A::QuasiAdjoint{<:Any,<:FiniteContinuousZernikeMode}, B::BroadcastQuasiMatrix{<:Any, typeof(*), <:Tuple{BroadcastQuasiVector, FiniteContinuousZernikeMode}})
    λ, F = B.args
    T = promote_type(eltype(A), eltype(F))
    @assert A' == F

    K, points = lastindex(F.points)-1, F.points
    ρs = []
    for j = 1:K
        append!(ρs, [points[j] / points[j+1]])
    end
    ts = inv.(one(T) .- ρs.^2)

    Λs = AbstractMatrix{T}[]
    for j in 1:K
        Tn = chebyshevt(points[j]..points[j+1])
        u = Tn \ λ.f.(axes(Tn,1))
        if points[j] ≈ 0
            X = jacobimatrix(Normalized(Jacobi(0, F.m)))
            append!(Λs, [Clenshaw(paddeddata(u), recurrencecoefficients(Tn)..., points[j+1]^2*(X+I)/2, _p0(Tn))])
        else
            X = jacobimatrix(SemiclassicalJacobi(ts[j], 0, 0, F.m))
            append!(Λs, [Clenshaw(paddeddata(u), recurrencecoefficients(Tn)..., points[j+1]^2*(I-X/ts[j]), _p0(Tn))])
        end
    end

    assembly_matrix(F, Λs)
end
function assembly_matrix(F::FiniteContinuousZernikeMode, Λs::Vector{<:AbstractMatrix})
    T = eltype(F)
    Cs = _getCs(F)
    Ms = [assembly_matrix(C, Λ) for (C, Λ) in zip(Cs, Λs)]

    # figure out necessary bandwidth
    # bs for number of bubbles
    N = F.N
    if first(F.points) ≈ 0
        bs = min(N-2, maximum(vcat([last(findall(x->abs(x) > 10*eps(T), Ms[1][1:N+3,1]))],[last(colsupport(view(Ms[i], :, 1)[:]))-2 for i in 2:lastindex(Ms)])))
    else
        bs = min(N-2, maximum([last(colsupport(view(Ms[i], :, 1)[:]))-2 for i in 1:lastindex(Ms)]))
    end
    γs = _getγs(F)
    _arrow_head_matrix(F, Ms, γs, N, bs, first(F.points))
end

###
# Gradient for constructing weak Laplacian.
###

struct GradientFiniteContinuousZernikeAnnulusMode{T}<:Basis{T}
    F::FiniteContinuousZernikeMode{T}
end

# GradientFiniteContinuousZernikeAnnulusMode(F::FiniteContinuousZernikeMode{T}) where T =  GradientFiniteContinuousZernikeAnnulusMode{T}(F)

axes(Z:: GradientFiniteContinuousZernikeAnnulusMode) = axes(Z.F)
==(P::GradientFiniteContinuousZernikeAnnulusMode, Q::GradientFiniteContinuousZernikeAnnulusMode) = P.F == Q.F

@simplify function *(D::Derivative, F::FiniteContinuousZernikeMode)
    GradientFiniteContinuousZernikeAnnulusMode(F)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:GradientFiniteContinuousZernikeAnnulusMode}, B::GradientFiniteContinuousZernikeAnnulusMode)
    @assert A' == B
    stiffness_matrix(B.F)
end
function stiffness_matrix(F::FiniteContinuousZernikeMode)

    N = F.N
    Cs = _getCs(F)

    if F.same_ρs
        if first(F.points) ≈ 0
            Δ = stiffness_matrix(Cs[2])
            Δs = [Δ for i in 1:length(F.points)-2]
            Δs = append!(Any[stiffness_matrix(Cs[1])], Δs)
        else
            Δ = stiffness_matrix(Cs[1])
            Δs = [Δ for i in 1:length(F.points)-1]
        end
    else
        Δs = stiffness_matrix.(Cs)
    end

    γs = _getγs(F)

    _arrow_head_matrix(F, Δs, γs, N, 1, first(F.points))
end

function zero_dirichlet_bcs!(F::FiniteContinuousZernikeMode{T}, Δ::LinearAlgebra.Symmetric{T,<:AbstractMatrix{T}}) where T
    points = F.points
    A, B = Δ.data.A.data, Δ.data.B[1]

    if !(first(points) ≈  0)
        A[1,:] .= zero(T); A[:,end] .= zero(T); A[1,1] = one(T); A[end,end] = one(T)
        A = Symmetric(A)
        B[1,:] .= zero(T); B[end,:] .= zero(T);
    else
        A[:,end] .= zero(T); A[end,end] = one(T)
        A = Symmetric(A)
        B[end,:] .= zero(T);
    end
end

function zero_dirichlet_bcs!(F::FiniteContinuousZernikeMode{T}, A::Matrix) where T
    points = F.points
    K = length(points)-1
    if first(points) > 0
        A[1,:] .= zero(T); A[:,1] .= zero(T)
        A[K+1, :] .= zero(T); A[:, K+1] .= zero(T)
        A[1,1] = one(T); A[K+1, K+1] = one(T)
    else
        A[K,:] .= zero(T); A[:,K] .= zero(T)
        A[K,K] = one(T)
    end
end

function zero_dirichlet_bcs!(F::FiniteContinuousZernikeMode{T}, Mf::PseudoBlockVector) where T
    points = F.points
    K = length(points)-1
    if !(first(points) ≈  0)
        Mf[1] = zero(T)
        Mf[K+1] = zero(T)
    else
        Mf[K] = zero(T)
    end
end

###
# Plotting
###

function element_plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{FiniteContinuousZernikeMode, AbstractVector}}) where T
    C, u = u.args 
    points = T.(C.points); K = length(points)-1
    N = C.N; m = C.m; j = C.j
    Cs = _getCs(C)

    γs = _getγs(points, m)
    append!(γs, one(T))

    if first(points) ≈ 0 && K > 1
        uc = [pad([u[1]*γs[1];u[K+1:K:end]], axes(Cs[1],2))]
        for k = 1:K-1 append!(uc, [pad([u[k];u[k+1]*γs[k+1];u[(K+k+1):K:end]], axes(Cs[k],2))]) end
    else
        uc = []
        for k = 1:K append!(uc, [pad([u[k];u[k+1]*γs[k];u[(K+1+k):K:end]], axes(Cs[k],2))]) end
    end

    θs=[]; rs=[]; valss=[];
    for k in 1:K
        (x, vals) = plotvalues(Cs[k]*uc[k])
        (θ, r, vals) =  plot_helper(x, vals)
        append!(θs,[θ]); append!(rs, [r]); append!(valss, [vals])
    end
    
    return (uc, θs, rs, valss)
end

### Error collection
function inf_error(F::FiniteContinuousZernikeMode{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function) where T
    K = lastindex(F.points)-1
    _inf_error(K, θs, rs, vals, u)
end