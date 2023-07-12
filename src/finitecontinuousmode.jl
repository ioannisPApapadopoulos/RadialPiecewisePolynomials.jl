struct FiniteContinuousZernikeMode{T} <: Basis{T}
    N::Int
    points::AbstractVector{T}
    m::Int
    j::Int
    L₁₁::Tuple{Vararg{AbstractMatrix{T}}}
    L₀₁::Tuple{Vararg{AbstractMatrix{T}}}
    L₁₀::Tuple{Vararg{AbstractMatrix{T}}}
    D::Tuple{Vararg{AbstractMatrix{T}}}
    b::Int # Should remove once adaptive expansion has been figured out.
end

function FiniteContinuousZernikeMode(N::Int, points::AbstractVector{T}, m::Int, j::Int, L₁₁, L₀₁, L₁₀, D, b::Int) where T
    @assert length(points) > 1 && points == sort(points)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    K = first(points) ≈ 0 ? length(points)-2 : length(points) - 1
    @assert length(L₁₁) == length(L₀₁) == length(L₁₀) == length(D) == K
    FiniteContinuousZernikeMode{T}(N, points, m, j, L₁₁, L₀₁, L₁₀, D, b)
end

function FiniteContinuousZernikeMode(N::Int, points::AbstractVector{T}, m::Int, j::Int) where {T}
    K = length(points)-1
    κ = first(points[1]) ≈ 0 ? 2 : 1
    ρs = []
    for k = κ:length(points)-1
        α, β = convert(T, first(points[k])), convert(T, last(points[k+1]))
        append!(ρs, [α / β])
    end

    ts = inv.(one(T) .- ρs.^2)
    Ls = _ann2element_via_lowering.(ts, m)
    L₁₁ = NTuple{K+1-κ, AbstractMatrix}(first.(Ls))
    L₀₁ = NTuple{K+1-κ, AbstractMatrix}([Ls[k][2] for k in 1:K+1-κ])
    L₁₀ = NTuple{K+1-κ, AbstractMatrix}(last.(Ls))

    Z = ZernikeAnnulus{T}.(ρs,1,1)
    D = (Z .\ (Laplacian.(axes.(Z,1)).*Weighted.(Z)))
    D = NTuple{K+1-κ, AbstractMatrix}([Ds.ops[m+1] for Ds in D])

    FiniteContinuousZernikeMode(N, points, m, j, L₁₁, L₀₁, L₁₀, D, m+2N) 
end

# FiniteContinuousZernikeMode(points::AbstractVector, m::Int, j::Int, L₁₁, L₀₁, L₁₀, D, b::Int) = FiniteContinuousZernikeMode{Float64}(points, m, j, L₁₁, L₀₁, L₁₀, D, b)
# FiniteContinuousZernikeMode(points::AbstractVector, m::Int, j::Int, L₁₁, L₀₁, L₁₀, D) = FiniteContinuousZernikeMode{Float64}(points, m, j, L₁₁, L₀₁, L₁₀, D, m+2N)

function axes(Z::FiniteContinuousZernikeMode{T}) where T
    first(Z.points) ≈ 0 && return (Inclusion(last(Z.points)*UnitDisk{T}()), blockedrange(Vcat(length(Z.points), Fill(length(Z.points) - 1, Z.N-2))))
    # (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
    (Inclusion(annulus(first(Z.points), last(Z.points))), blockedrange(Vcat(length(Z.points), Fill(length(Z.points) - 1, Z.N-2))))
end
==(P::FiniteContinuousZernikeMode, Q::FiniteContinuousZernikeMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j && P.b == Q.b

function _getCs(points::AbstractVector{T}, m::Int, j::Int, b::Int, L₁₁, L₀₁, L₁₀, D) where T
    K = length(points)-1
    first(points) > 0 && return [ContinuousZernikeAnnulusElementMode([points[k]; points[k+1]], m, j, L₁₁[k], L₀₁[k], L₁₀[k], D[k], b) for k in 1:K]

    append!(Any[ContinuousZernikeElementMode([points[1]; points[2]], m, j)], [ContinuousZernikeAnnulusElementMode([points[k]; points[k+1]], m, j, L₁₁[k-1], L₀₁[k-1], L₁₀[k-1], D[k-1], b) for k in 2:K])
end

function _getCs(F::FiniteContinuousZernikeMode)
    points, m, j, b, = F.points, F.m, F.j, F.b
    L₁₁, L₀₁, L₁₀, D = F.L₁₁, F.L₀₁, F.L₁₀, F.D
    _getCs(points, m, j, b, L₁₁, L₀₁, L₁₀, D)
end

function _getγs(points::AbstractArray{T}, m::Int) where T
    K = length(points)-1
    first(points) > 0 && return [(one(T)-(points[k+1]/points[k+2])^2)*(points[k+1]/points[k+2])^m / (one(T)-(points[k]/points[k+1])^2) for k in 1:K-1]
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

    hats = vcat([fs[i][1] for i in 1:K-1], fs[end][1:2])

    bubbles = zeros(T, N-2, K)
    for i in 1:K bubbles[:,i] = fs[i][3:N] end

    pad(append!(hats, vec(bubbles')), axes(F,2))
end


###
# L2 inner product
###
function _build_top_left_block(Ms, γs::AbstractArray{T}) where T
    K = length(Ms)

    a = []
    for i in 1:K
        append!(a, diag(Ms[i][1:2,1:2]))
    end

    dv = zeros(T, K+1)
    dv[1] = a[1]; dv[end] = a[end];
    ev = zeros(T, K)
    γs = vcat(γs, one(T))

    for i in 1:K-1 dv[i+1] = a[2i]*γs[i]^2 + a[2i+1] end
    for i in 1:K ev[i] = Ms[i][1,2] * γs[i] end

    Symmetric(BandedMatrix{T}(0=>dv, 1=>ev))
end

function _build_second_block(Ms, γs::AbstractArray{T}, bs::Int) where T
    K = length(Ms)
    γs = vcat(γs, one(T))
    dv, ev = [], []
    for j in 1:bs
        append!(dv, [zeros(T, K)])
        append!(ev, [zeros(T, K)])
        for i in 1:K
            dv[j][i] = Ms[i][1,j+2]
            ev[j][i] = Ms[i][2,j+2] * γs[i]
        end
    end
    [BandedMatrix{T}((0=>dv[j], -1=>ev[j]), (K+1, K)) for j in 1:bs]
end

function _build_trailing_bubbles(Ms, γs::AbstractArray{T}, N::Int, bs::Int) where T
    K = length(Ms)

    Mn = [Ms[i][3:N, 3:N] for i in 1:K]
    if bs ==  2
        return [Symmetric(BandedMatrix{T}(0=>view(M, band(0)), 1=>view(M, band(1)), 2=>view(M, band(2)))) for M in Mn]
    elseif bs == 1
        return [Symmetric(BandedMatrix{T}(0=>view(M, band(0)), 1=>view(M, band(1)))) for M in Mn]
    else
        error("Are you using _build_mass_trailing_bubbles correctly?")
    end
end

function _arrow_head_matrix(Ms, γs::AbstractArray{T}, N::Int, bs::Int) where T
    A = _build_top_left_block(Ms, γs)
    B = _build_second_block(Ms, γs, bs)
    C = BandedMatrix{T, Matrix{T}, Base.OneTo{Int64}}[]
    D = _build_trailing_bubbles(Ms, γs, N, bs)
    Symmetric(ArrowheadMatrix{T}(A, B, C, D))
end

# FIXME: Need to make type-safe
@simplify function *(A::QuasiAdjoint{<:Any,<:FiniteContinuousZernikeMode}, B::FiniteContinuousZernikeMode)
    # T = promote_type(eltype(A), eltype(B))
    @assert A' == B

    Cs = _getCs(B)

    Ms = [C' * C for C in Cs]
    γs = _getγs(B)

    _arrow_head_matrix(Ms, γs, B.N, 2)
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
    F = B.F
    N = F.N
    Cs = _getCs(F)
    
    xs = [axes(C,1) for C in Cs]
    Ds = [Derivative(x) for x in xs]
    Δs = [(D*C)' * (D*C) for (C, D) in zip(Cs, Ds)]

    γs = _getγs(F)

    _arrow_head_matrix(Δs, γs, N, 1)
end

# function zero_dirichlet_bcs!(F::FiniteContinuousZernikeMode{T}, Δ::AbstractMatrix{T}, Mf::AbstractVector{T}) where T
#     N, points = F.N, F.points
#     K = length(points)-1
    
#     if !(first(points) ≈  0)
#         Δ[:,1].=0
#         Δ[1,:].=0
#         Δ[1,1]=1.
#         Mf[1]=0
#     end
#     Δ[N+(K-2)*(N-1)+1,:].=0; Δ[:,N+(K-2)*(N-1)+1].=0; Δ[N+(K-2)*(N-1)+1,N+(K-2)*(N-1)+1]=1.;
#     Mf[N+(K-2)*(N-1)+1]=0;
# end

function zero_dirichlet_bcs!(F::FiniteContinuousZernikeMode{T}, Δ::AbstractMatrix{T}) where T
    points = F.points
    A, B = Δ.data.A.data, Δ.data.B[1]

    if !(first(points) ≈  0)
        A[1,:] .= zero(T); A[:,end] .= zero(T)
        A = Symmetric(A)
        B[1,:] .= zero(T); B[end,:] .= zero(T)
    else
        return 0.0
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
        uc = [pad([u[1]*γs[1];u[2:N]], axes(Cs[1],2))]
        k=2; append!(uc, [pad([u[1]; u[N+(k-2)*(N-1)+1]*γs[k]; u[N+(k-2)*(N-1)+2:N+(k-1)*(N-1)]], axes(Cs[k],2))]) 
        for k = 3:K append!(uc, [pad([u[N+(k-3)*(N-1)+1];u[N+(k-2)*(N-1)+1]*γs[k];u[N+(k-2)*(N-1)+2:N+(k-1)*(N-1)]], axes(Cs[k],2))]) end
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