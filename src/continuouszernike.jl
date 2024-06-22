struct ContinuousZernike{T, N<:Int, P<:AbstractVector} <: Basis{T}
    N::N
    points::P
    Fs::Tuple{Vararg{ContinuousZernikeMode}}
end

function ContinuousZernike{T}(N::Int, points::AbstractVector, Fs::Tuple{Vararg{ContinuousZernikeMode}}) where {T}
    @assert length(points) > 1 && points == sort(points)
    @assert length(Fs) == 2N-1
    ContinuousZernike{T, Int, typeof(points)}(N, points, Fs)
end
ContinuousZernike(N::Int, points::AbstractVector) = ContinuousZernike{Float64}(N, points, Tuple(_getFs(N, points)))

function axes(Z::ContinuousZernike{T}) where T
    first(Z.points) ≈ 0 && return (Inclusion(last(Z.points)*UnitDisk{T}()), blockedrange(Vcat(length(Z.points)-1, Fill(length(Z.points) - 1, Z.N-2))))
    # (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
    (Inclusion(annulus(first(Z.points), last(Z.points))), blockedrange(Vcat(length(Z.points), Fill(length(Z.points) - 1, Z.N-2))))
end
==(P::ContinuousZernike, Q::ContinuousZernike) = P.N == Q.N && P.points == Q.points

function show(io::IO, Φ::ContinuousZernike)
    N, points = Φ.N, Φ.points
    print(io, "ContinuousZernike at degree N=$N and endpoints $points.")
end

# Matrices for lowering to ZernikeAnnulus(0,0) via
# direct lowering. Less stable, but probably lower complexity.
function _ann2element_via_raising(t::T) where T
    # {T} did not change the speed.
    Q₀₀ = SemiclassicalJacobi{T}.(t, 0, 0, 0:∞)
    Q₀₁ = SemiclassicalJacobi{T}.(t, 0, 1, 0:∞)
    Q₁₀ = SemiclassicalJacobi{T}.(t, 1, 0, 0:∞)
    Q₁₁ = SemiclassicalJacobi{T}.(t, 1, 1, 0:∞)

    R₁₁ = (Weighted.(Q₀₀) .\ Weighted.(Q₁₁)) / t^2
    R₀₁ = BroadcastVector{AbstractVector}((Q, P) -> (Weighted(Q) \ Weighted(P))[1:2,1] / t, Q₀₀, Q₀₁)
    R₁₀ = BroadcastVector{AbstractVector}((Q, P) -> (Weighted(Q) \ Weighted(P))[1:2,1] / t, Q₀₀, Q₁₀)

    BroadcastVector{AbstractMatrix}((R11, R01, R10)->Hcat(Vcat(R10, Zeros{T}(∞)), Vcat(R01, Zeros{T}(∞)), R11), R₁₁, R₀₁, R₁₀)
end

function _getMs_ms_js(N::Int)
    ms = ((0:2N) .÷ 2)[2:end-1]
    js = repeat([0; 1], N)[2:end]
    Ms = ((N + 1 .- ms) .÷ 2); Ms[Ms .<= 2] .= 3
    (Ms, ms, js)
end

function _getFs(N::Int, points::AbstractVector{T}) where T
    # Ordered list of Fourier modes (ms, js) and correct length for each Fourier mode Ms.
    same_ρs = false
    Ms, ms, js = _getMs_ms_js(N)
    K = length(points)-1

    κ = first(points[1]) ≈ 0 ? 2 : 1

    # List of radii of the annuli
    ρs = []
    for k = κ:length(points)-1
        α, β = convert(T, points[k]), convert(T, points[k+1])
        append!(ρs, [α / β])
    end

    if !isempty(ρs)
        # If all ρs are the same, then we can reduce the computational
        # overhead by only considering one hierarchy of semiclassical
        # Jacobi polynomials for all the cells.
        if all(ρs .≈ ρs[1])
            ρs = [ρs[1]]
            κ = K
            same_ρs = true
        end

        # Semiclassical Jacobi parameter t
        ts = inv.(one(T) .- ρs.^2)

        # Use broadcast notation to compute all the lowering matrices across all
        # intervals and Fourier modes simultaneously.


        Rs = _ann2element_via_raising.(ts)
        cst = [[sum.(SemiclassicalJacobiWeight.(t,a,a,0:ms[end])) for t in ts] for a in 1:-1:0]

        # Use broadcast notation to compute all the derivative matrices across
        # all the intervals and Fourier modes simultaneously
        Z = ZernikeAnnulus{T}.(ρs,1,1)
        Ds = (Z .\ (Laplacian.(axes.(Z,1)).*Weighted.(Z)))
        Ds = [Ds[i].ops for i in 1:K+1-κ];
    else
        ts = []
        cst = [[]]
    end


    # Loop over the Fourier modes
    Fs = []

    # Pre-calling (sometimes) gives a big speedup.
    # After some tests, this is not the case here.
    # i is the element, j is the type of lowering
    # [[Ls[i][j][N+1] for j in 1:3] for i in 1:K+1-κ]
    # [Ds[i][N+1] for i in 1:K+1-κ]

    for (M, m, j) in zip(Ms, ms, js)
        # Extract the lowering and differentiation matrices associated
        # with each Fourier mode and store in the Tuples
        R = NTuple{K+1-κ, AbstractMatrix}([Rs[i][m+1] for i in 1:K+1-κ])
        D = NTuple{K+1-κ, AbstractMatrix}([Ds[i][m+1] for i in 1:K+1-κ])

        normalize_constants = Vector{T}[T[cst[k][i][m+1] for k in 1:lastindex(cst)] for i in 1:K+1-κ]
        Cs = Tuple(_getCs(points, m, j, N, R, D, normalize_constants, same_ρs))

        # Construct the structs for each Fourier mode seperately
        append!(Fs, [ContinuousZernikeMode(M, points, m, j, Cs, normalize_constants, same_ρs, N)])
    end
    return Fs
end

_getFs(Φ::ContinuousZernike{T}) where T = _getFs(Φ.N, Φ.points)

function ldiv(Φ::ContinuousZernike{V}, f::AbstractQuasiVector) where V
    @warn "Expanding via ContinuousZernike is ill-conditioned, please use ZernikeBasis."
    Fs = Φ.Fs
    [Φ \ f.f.(axes(Φ, 1)) for Φ in Fs]
end

###
# L2 inner products
# Gives out list of mass matrices of correct size
###
@simplify function *(A::QuasiAdjoint{<:Any,<:ContinuousZernike}, B::ContinuousZernike)
    @assert A' == B
    mass_matrix(B)
end
function mass_matrix(A::ContinuousZernike)
    Fs = A.Fs
    mass_matrix.(Fs)
end

###
# Weighted L2 inner products
# Gives out list of the assembly matrices of correct size
###
@simplify function *(A::QuasiAdjoint{<:Any,<:ContinuousZernike}, λB::BroadcastQuasiMatrix{<:Any, typeof(*), <:Tuple{BroadcastQuasiVector, ContinuousZernike}})
    λ, B = λB.args
    @assert A' == B
    Fs = B.Fs
    T = promote_type(eltype(A), eltype(B))

    K, points = lastindex(B.points)-1, B.points
    ρs = []
    for j = 1:K
        append!(ρs, [points[j] / points[j+1]])
    end
    ts = inv.(one(T) .- ρs.^2)

    Λs = []
    for j in 1:K
        Tn = chebyshevt(points[j]..points[j+1])
        u = Tn \ λ.f.(axes(Tn,1))
        if points[j] ≈ 0
            X = jacobimatrix.(Normalized.(Jacobi.(0, 0:B.N-1)))
            append!(Λs, [BroadcastArray{AbstractMatrix}(Y->Clenshaw(paddeddata(u), recurrencecoefficients(Tn)..., points[j+1]^2/2 * (Y+I), _p0(Tn)), X)])
        else
            X = jacobimatrix.(SemiclassicalJacobi.(ts[j], 0, 0, 0:B.N-1))
            append!(Λs, [BroadcastArray{AbstractMatrix}(Y->Clenshaw(paddeddata(u), recurrencecoefficients(Tn)..., points[j+1]^2 * (I-Y/ts[j]), _p0(Tn)), X)])
        end
    end

    # Rearrange the list to match with Fs
    _, ms, _ = _getMs_ms_js(B.N)
    Λs = [[Λs[k][m+1] for k in 1:lastindex(Λs)] for m in ms]

    [assembly_matrix(F, Λ) for (F, Λ) in zip(Fs, Λs)]
end

function piecewise_constant_assembly_matrix(Φ::ContinuousZernike, λ::Function)
    Fs = Φ.Fs
    K, points = lastindex(Φ.points)-1, Φ.points
    λs = λ.((points[1:end-1] + points[2:end] ) / 2)
    Λ = [Diagonal(Fill(λs[k],∞)) for k in 1:K]
    [assembly_matrix(F, Λ) for F in Fs]
end

###
# Gradient for constructing weak Laplacian.
###

struct GradientContinuousZernike{T}<:Basis{T}
    Φ::ContinuousZernike{T}
end

# GradientContinuousZernike{T}(N::Int, points::AbstractVector) where {T} =  GradientContinuousZernike{T,Int, typeof(points)}(N, points)
# GradientContinuousZernike(N::Int, points::AbstractVector) =  GradientContinuousZernike{Float64}(N, points)

axes(Z:: GradientContinuousZernike) = (Inclusion(last(Z.Φ.points)*UnitDisk{eltype(Z)}()), oneto(Z.Φ.N*(length(Z.Φ.points)-1)-(length(Z.Φ.points)-2)))
==(P::GradientContinuousZernike, Q::GradientContinuousZernike) = P.Φ.points == Q.Φ.points

@simplify function *(D::Derivative, Φ::ContinuousZernike)
    GradientContinuousZernike(Φ)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:GradientContinuousZernike}, B::GradientContinuousZernike)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B
    # points = T.(B.Φ.points);
    # N = B.Φ.N;
    stiffness_matrix(B)
end
function stiffness_matrix(B::GradientContinuousZernike)
    Fs = B.Φ.Fs
    stiffness_matrix.(Fs)
end


# function zero_dirichlet_bcs!(Φ::ContinuousZernike{T}, Δ::AbstractVector{<:LinearAlgebra.Symmetric{T,<:BBBArrowheadMatrix{T}}}) where T
#     @assert length(Δ) == 2*Φ.N-1
#     Fs = _getFs(Φ.N, Φ.points)
#     zero_dirichlet_bcs!.(Fs, Δ)
# end

function zero_dirichlet_bcs!(Φ::ContinuousZernike{T}, Δ::AbstractVector{<:AbstractMatrix}) where T
    @assert length(Δ) == 2*Φ.N-1
    # @assert Δ[1] isa LinearAlgebra.Symmetric{T, <:BBBArrowheadMatrix{T}}
    Fs = Φ.Fs #_getFs(Φ.N, Φ.points)
    zero_dirichlet_bcs!.(Fs, Δ)
end

function zero_dirichlet_bcs!(Φ::ContinuousZernike{T}, Mf::AbstractVector{<:PseudoBlockVector}) where T
    @assert length(Mf) == 2*Φ.N-1
    Fs = Φ.Fs #_getFs(Φ.N, Φ.points)
    zero_dirichlet_bcs!.(Fs, Mf)
end

function zero_dirichlet_bcs!(Φ::ContinuousZernike{T}, Mf::AbstractVector{<:AbstractVector}) where T
    @assert length(Mf) == 2*Φ.N-1
    Fs = Φ.Fs #_getFs(Φ.N, Φ.points)
    zero_dirichlet_bcs!.(Fs, Mf)
end

function zero_dirichlet_bcs!(points::AbstractVector{T}, A::Matrix) where T
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

function zero_dirichlet_bcs!(points::AbstractVector{T}, Mf::PseudoBlockVector) where T
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
##
# This helper function takes the list of coefficient values from ldiv and converts them into 
# a 3-tensor of degree × Fourier mode × element. Us is the hat/bubble coeffiecients
# and Ũs are the corresponding ZernikeAnnulus(ρ,1,1) coefficients.
function _bubble2disk_or_ann_all_modes(Φ::ContinuousZernike, us::AbstractVector)
    T = eltype(Φ)
    points = T.(Φ.points); K = length(points)-1
    N = Int((length(us)+1)/2)
    @assert N ≤ Φ.N

    Ms, ms, _ = _getMs_ms_js(N)

    Ñ = isodd(N) ? N : N+1
    Us = zeros(T,(Ñ+1)÷2,2Ñ-1,K)

    for i in 1:2N-1
        m = ms[i]
        γs = _getγs(points, m)
        append!(γs, one(T))
        if first(points) ≈ 0
            Us[1:Ms[i]-1,i,1] = [us[i][1]*γs[1];us[i][K+1:K:end]]
            if K > 1
                for k = 1:K-1
                    Us[1:Ms[i],i,k+1] = [us[i][k];us[i][k+1]*γs[k+1];us[i][(K+k+1):K:end]] 
                end
            end
        else
            for k = 1:K 
                Us[1:Ms[i],i,k] = [us[i][k];us[i][k+1]*γs[k];us[i][(K+1+k):K:end]] 
            end
        end
    end

    Ũs = zeros(T, (Ñ+1)÷2,2Ñ-1,K)
    Fs = Φ.Fs[1:length(us)] #_getFs(N, points)

    for k in 1:K
        if k == 1 && first(points) ≈ 0
            for (m, i) in zip(ms, 1:2N-1)
                Ũs[1:Ms[i],i,k] = bubble2disk(m, Us[1:Ms[i],i,k])
            end
        else
            for (Fm, i) in zip(Fs, 1:2N-1)
                C = Fm.Cs[k] #_getCs(Fm)[k]
                Ũs[1:Ms[i],i,k] = bubble2ann(C, Us[1:Ms[i],i,k])
            end
        end
    end
    Us, Ũs
end


function finite_plotvalues(Φ::ContinuousZernike, us::AbstractVector; N=0, K=0)
    T = eltype(Φ)
    _, Ũs = _bubble2disk_or_ann_all_modes(Φ, us)
    points = T.(Φ.points)
    K = K==0 ? length(Φ.points)-1 : K
    N = N == 0 ? Φ.N : N
    θs, rs, vals = [], [], []
    for k in 1:K
        if k == 1 && first(points) ≈ 0
            ρ = points[2]
            Z = Zernike{T}(0)
            g = scalegrid(AnnuliOrthogonalPolynomials.grid(Z, Block(N)), ρ)
            FT = ZernikeITransform{T}(N, 0, 0)
            val = FT * pad(ModalTrav(Ũs[:,:,k]),axes(Z,2))[Block.(OneTo(N))]
        else
            α, β = points[k], points[k+1]
            ρ = α / β
            w_a = zero(T)
            Z = ZernikeAnnulus{T}(ρ, w_a, w_a)
            # Scale the grid
            g = scalegrid(AnnuliOrthogonalPolynomials.grid(Z, Block(N)), α, β)
            # Use fast transforms for synthesis
            FT = ZernikeAnnulusITransform{T}(N, w_a, w_a, 0, ρ)
            val = FT * pad(ModalTrav(Ũs[:,:,k]),axes(Z,2))[Block.(OneTo(N))]
        end
        (θ, r, val) = plot_helper(g, val)
        append!(θs, [θ]); append!(rs, [r]); append!(vals, [val])
    end
    return (θs, rs, vals)
end


## Error collection
function _inf_error(K::Int, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function)
    vals_ = []
    for k = 1:K
        append!(vals_, [abs.(vals[k] - u.(RadialCoordinate.(rs[k],θs[k]')))])
    end
    vals_, norm((norm.(vals_, Inf)), Inf)
end

function inf_error(Φ::ContinuousZernike{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function;K=0) where T
    K = K==0 ? lastindex(Φ.points)-1 : K
    _inf_error(K, θs, rs, vals, u)
end