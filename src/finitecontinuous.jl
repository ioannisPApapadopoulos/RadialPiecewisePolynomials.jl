struct FiniteContinuousZernike{T, N<:Int, P<:AbstractVector} <: Basis{T}
    N::N
    points::P
end

function FiniteContinuousZernike{T}(N::Int, points::AbstractVector) where {T}
    @assert length(points) > 1 && points == sort(points)
    FiniteContinuousZernike{T, Int, typeof(points)}(N, points)
end
FiniteContinuousZernike(N::Int, points::AbstractVector) = FiniteContinuousZernike{Float64}(N, points)

function axes(Z::FiniteContinuousZernike{T}) where T
    first(Z.points) ≈ 0 && return (Inclusion(last(Z.points)*UnitDisk{T}()), blockedrange(Vcat(length(Z.points)-1, Fill(length(Z.points) - 1, Z.N-2))))
    # (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
    (Inclusion(annulus(first(Z.points), last(Z.points))), blockedrange(Vcat(length(Z.points), Fill(length(Z.points) - 1, Z.N-2))))
end
==(P::FiniteContinuousZernike, Q::FiniteContinuousZernike) = P.N == Q.N && P.points == Q.points


# Matrices for lowering to ZernikeAnnulus(1,1) via
# the Jacobi matrix. Stable, but probably higher complexity
# and cannot be used for L2 inner-product of FiniteZernikeBasis
# and FiniteContinuousZernike
function _ann2element_via_Jacobi(t::T) where T
    Q₁₁ = SemiclassicalJacobi{T}.(t, 1, 1, 0:∞)
    X = jacobimatrix.(Q₁₁)

    L₁₁ = (X .- X .* X)/t^2
    L₀₁ = (Fill(I, ∞) .- X)/t
    L₁₀ = X/t

    (L₁₁, L₀₁, L₁₀)
end

# Matrices for lowering to ZernikeAnnulus(0,0) via
# direct lowering. Less stable, but probably lower complexity.
function _ann2element_via_lowering(t::T) where T
    Q₀₀ = SemiclassicalJacobi{T}.(t, 0, 0, 0:∞)
    Q₀₁ = SemiclassicalJacobi{T}.(t, 0, 1, 0:∞)
    Q₁₀ = SemiclassicalJacobi{T}.(t, 1, 0, 0:∞)
    Q₁₁ = SemiclassicalJacobi{T}.(t, 1, 1, 0:∞)

    L₁₁ = (Weighted.(Q₀₀) .\ Weighted.(Q₁₁)) / t^2
    L₀₁ = (Weighted.(Q₀₀) .\ Weighted.(Q₀₁)) / t
    L₁₀ = (Weighted.(Q₀₀) .\ Weighted.(Q₁₀)) / t

    (L₁₁, L₀₁, L₁₀)
end

function _getMs_ms_js(N::Int)
    ms = ((0:2N) .÷ 2)[2:end-1]
    js = repeat([0; 1], N)[2:end]
    Ms = ((N + 1 .- ms) .÷ 2); Ms[Ms .<= 2] .= 3
    (Ms, ms, js)
end

function _getFs(N::Int, points::AbstractVector{T}) where T
    # Ordered list of Fourier modes (ms, js) and correct length for each Fourier mode Ms.
    Ms, ms, js = _getMs_ms_js(N)
    K = length(points)-1

    κ = first(points[1]) ≈ 0 ? 2 : 1

    # List of radii of the annuli
    ρs = []
    for k = κ:length(points)-1
        α, β = convert(T, points[k]), convert(T, points[k+1])
        append!(ρs, [α / β])
    end
    # Semiclassical Jacobi parameter t
    ts = inv.(one(T) .- ρs.^2)

    # Use broadcast notation to compute all the lowering matrices across all
    # intervals and Fourier modes simultaneously.
    Ls = _ann2element_via_lowering.(ts)

    # Use broadcast notation to compute all the derivative matrices across
    # all the intervals and Fourier modes simultaneously
    Z = ZernikeAnnulus{T}.(ρs,1,1)
    Ds = (Z .\ (Laplacian.(axes.(Z,1)).*Weighted.(Z)))

    # Loop over the Fourier modes
    Fs = []
    for (M, m, j) in zip(Ms, ms, js)
        # Extract the lowering and differentiation matrices associated
        # with each Fourier mode and store in the Tuples
        L₁₁ = NTuple{K+1-κ, AbstractMatrix}([Ls[i][1][m+1] for i in 1:K+1-κ])
        L₀₁ = NTuple{K+1-κ, AbstractMatrix}([Ls[i][2][m+1] for i in 1:K+1-κ])
        L₁₀ = NTuple{K+1-κ, AbstractMatrix}([Ls[i][3][m+1] for i in 1:K+1-κ])
        
        D = NTuple{K+1-κ, AbstractMatrix}([(Ds[i]).ops[m+1] for i in 1:K+1-κ])

        # Construct the structs for each Fourier mode seperately
        append!(Fs, [FiniteContinuousZernikeMode(M, points, m, j, L₁₁, L₀₁, L₁₀, D, N)])
    end
    return Fs
end

_getFs(F::FiniteContinuousZernike{T}) where T = _getFs(F.N, F.points)

function ldiv(F::FiniteContinuousZernike{V}, f::AbstractQuasiVector) where V
    # T = promote_type(V, eltype(f))
    @warn "Expanding via FiniteContinuousZernike is ill-conditioned, please use FiniteZernikeBasis."
    T = V
    N = F.N; points = T.(F.points)

    Fs = _getFs(N, points)
    [F \ f.f.(axes(F, 1)) for F in Fs]
end

###
# L2 inner products
# Gives out list of mass matrices of correct size
###
@simplify function *(A::QuasiAdjoint{<:Any,<:FiniteContinuousZernike}, B::FiniteContinuousZernike)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B
    points = T.(B.points);
    N = B.N;
    Fs = _getFs(N, points)
    [F' * F for F in Fs]
end

###
# Gradient for constructing weak Laplacian.
###

struct GradientFiniteContinuousZernike{T}<:Basis{T}
    F::FiniteContinuousZernike{T}
end

# GradientFiniteContinuousZernike{T}(N::Int, points::AbstractVector) where {T} =  GradientFiniteContinuousZernike{T,Int, typeof(points)}(N, points)
# GradientFiniteContinuousZernike(N::Int, points::AbstractVector) =  GradientFiniteContinuousZernike{Float64}(N, points)

axes(Z:: GradientFiniteContinuousZernike) = (Inclusion(last(Z.F.points)*UnitDisk{eltype(Z)}()), oneto(Z.F.N*(length(Z.F.points)-1)-(length(Z.F.points)-2)))
==(P::GradientFiniteContinuousZernike, Q::GradientFiniteContinuousZernike) = P.F.points == Q.F.points

@simplify function *(D::Derivative, F::FiniteContinuousZernike)
    GradientFiniteContinuousZernike(F)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:GradientFiniteContinuousZernike}, B::GradientFiniteContinuousZernike)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B
    points = T.(B.F.points);
    N = B.F.N;
    Fs = _getFs(N, points)
    ∇ = Derivative(axes(Fs[1],1))
    [(∇*F)' * (∇*F) for F in Fs]
end

function zero_dirichlet_bcs!(F::FiniteContinuousZernike{T}, Δ::AbstractVector{<:LinearAlgebra.Symmetric{T,<:ArrowheadMatrix{T}}}) where T
    @assert length(Δ) == 2*F.N-1
    Fs = _getFs(F.N, F.points)
    zero_dirichlet_bcs!.(Fs, Δ)
end

function zero_dirichlet_bcs!(F::FiniteContinuousZernike{T}, Mf::AbstractVector{<:PseudoBlockVector{T}}) where T
    @assert length(Mf) == 2*F.N-1
    Fs = _getFs(F.N, F.points)
    zero_dirichlet_bcs!.(Fs, Mf)
end

###
# Plotting
##
# This helper function takes the list of coefficient values from ldiv and converts them into 
# a 3-tensor of degree × Fourier mode × element. Us is the hat/bubble coeffiecients
# and Ũs are the corresponding ZernikeAnnulus(ρ,1,1) coefficients.
function _bubble2disk_or_ann_all_modes(F::FiniteContinuousZernike, us::AbstractVector)
    T = eltype(F)
    points = T.(F.points); K = length(points)-1
    N = F.N;
    Ms, ms, _ = _getMs_ms_js(N)

    Ñ = isodd(N) ? N : N+1
    Us = zeros(T,(Ñ+1)÷2,2Ñ-1,K)

    for i in 1:2N-1
        m = ms[i]
        γs = _getγs(points, m)
        append!(γs, one(T))
        if first(points) ≈ 0 && K > 1
            Us[1:Ms[i]-1,i,1] = [us[i][1]*γs[1];us[i][K+1:K:end]]
            for k = 1:K-1 
                Us[1:Ms[i],i,k+1] = [us[i][k];us[i][k+1]*γs[k+1];us[i][(K+k+1):K:end]] 
            end
        else
            for k = 1:K 
                Us[1:Ms[i],i,k] = [us[i][k];us[i][k+1]*γs[k];us[i][(K+1+k):K:end]] 
            end
        end
    end

    Ũs = zeros(T, (Ñ+1)÷2,2Ñ-1,K)
    Fs = _getFs(N, points)

    for k in 1:K
        if k == 1 && first(points) ≈ 0
            for (m, i) in zip(ms, 1:2N-1)
                Ũs[1:Ms[i],i,k] = bubble2disk(m, Us[1:Ms[i],i,k])
            end
        else
            α, β = points[k], points[k+1]
            for (Fm, i) in zip(Fs, 1:2N-1)
                C = _getCs(Fm)[k]
                Ũs[1:Ms[i],i,k] = bubble2ann(C, Us[1:Ms[i],i,k])
            end
        end
    end
    Us, Ũs
end


function finite_plotvalues(F::FiniteContinuousZernike, us::AbstractVector)
    T = eltype(F)
    _, Ũs = _bubble2disk_or_ann_all_modes(F, us)
    points = T.(F.points); N = F.N; K = length(points)-1
    θs=[]; rs=[]; vals = []   
    for k in 1:K
        if k == 1 && first(points) ≈ 0
            ρ = points[2]
            Z = Zernike{T}(0)
            g = scalegrid(AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(2N)), ρ)
            FT = ZernikeITransform{T}(2N, 0, 0)
            val = FT * pad(ModalTrav(Ũs[:,:,k]),axes(Z,2))[Block.(OneTo(2N))]
        else
            α, β = points[k], points[k+1]
            ρ = α / β
            Z = ZernikeAnnulus{T}(ρ, zero(T), zero(T))
            # Scale the grid
            g = scalegrid(AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(N)), α, β)
            # Use fast transforms for synthesis
            FT = ZernikeAnnulusITransform{T}(N, 0, 0, 0, ρ)
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
    vals_, sum(maximum.(vals_))
end

function inf_error(F::FiniteContinuousZernike{T}, θs::AbstractVector, rs::AbstractVector, vals::AbstractVector, u::Function) where T
    K = lastindex(F.points)-1
    _inf_error(K, θs, rs, vals, u)
end