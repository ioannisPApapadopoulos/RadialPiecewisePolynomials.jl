struct FiniteContinuousZernikeAnnulus{T, N<:Int, P<:AbstractVector} <: Basis{T}
    N::N
    points::P
end

function FiniteContinuousZernikeAnnulus{T}(N::Int, points::AbstractVector) where {T}
    @assert length(points) > 1 && points == sort(points)
    FiniteContinuousZernikeAnnulus{T, Int, typeof(points)}(N, points)
end
FiniteContinuousZernikeAnnulus(N::Int, points::AbstractVector) = FiniteContinuousZernikeAnnulus{Float64}(N, points)

axes(Z::FiniteContinuousZernikeAnnulus) = (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
==(P::FiniteContinuousZernikeAnnulus, Q::FiniteContinuousZernikeAnnulus) = P.N == Q.N && P.points == Q.points

function _getFs(N::Int, points::AbstractVector{T}) where T
    ms = ((0:2N) .÷ 2)[2:end-1]
    js = repeat([0; 1], N)[2:end]
    Ms = ((N + 1 .- ms) .÷ 2); Ms[Ms .== 1] .= 2
    [FiniteContinuousZernikeAnnulusMode{T}(M, points, m, j, N) for (M, m, j) in zip(Ms, ms, js)]
end

function ldiv(F::FiniteContinuousZernikeAnnulus{V}, f::AbstractQuasiVector) where V
    # T = promote_type(V, eltype(f))
    T = V
    N = F.N; points = T.(F.points)

    Fs = _getFs(N, points)
    [F \ f.f.(axes(F, 1)) for F in Fs]
end

###
# L2 inner product
# Gives out list of mass matrices of correct size
###
@simplify function *(A::QuasiAdjoint{<:Any,<:FiniteContinuousZernikeAnnulus}, B::FiniteContinuousZernikeAnnulus)
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

struct GradientFiniteContinuousZernikeAnnulus{T, N<:Int, P<:AbstractVector}<:Basis{T}
    N::N
    points::P
end

GradientFiniteContinuousZernikeAnnulus{T}(N::Int, points::AbstractVector) where {T} =  GradientFiniteContinuousZernikeAnnulus{T,Int, typeof(points)}(N, points)
GradientFiniteContinuousZernikeAnnulus(N::Int, points::AbstractVector) =  GradientFiniteContinuousZernikeAnnulus{Float64}(N, points)

axes(Z:: GradientFiniteContinuousZernikeAnnulus) = (Inclusion(last(Z.points)*UnitDisk{eltype(Z)}()), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
==(P:: GradientFiniteContinuousZernikeAnnulus, Q:: GradientFiniteContinuousZernikeAnnulus) = P.points == Q.points

@simplify function *(D::Derivative, C::FiniteContinuousZernikeAnnulus)
    GradientFiniteContinuousZernikeAnnulus(C.N, C.points)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:GradientFiniteContinuousZernikeAnnulus}, B::GradientFiniteContinuousZernikeAnnulus)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B
    points = T.(B.points);
    N = B.N;
    Fs = _getFs(N, points)
    ∇ = Derivative(axes(Fs[1],1))
    [(∇*F)' * (∇*F) for F in Fs]
end

function zero_dirichlet_bcs!(F::FiniteContinuousZernikeAnnulus{T}, Δ::Vector{Matrix{T}}, Mf::Vector{Vector{T}}) where T
    Fs = _getFs(F.N, F.points)
    zero_dirichlet_bcs!.(Fs, Δ, Mf)
end

###
# Plotting
##
# This helper function takes the list of coefficient values from ldiv and converts them into 
# a 3-tensor of degree × Fourier mode × element. Us is the hat/bubble coeffiecients
# and Ũs are the corresponding ZernikeAnnulus(ρ,1,1) coefficients.
function _bubble2ann_all_modes(F::FiniteContinuousZernikeAnnulus{T}, us::AbstractVector) where T
    points = T.(F.points); K = length(points)-1
    N = F.N;
    ms = ((0:2N) .÷ 2)[2:end-1]
    Ms = ((N + 1 .- ms) .÷ 2); Ms[Ms .== 1] .= 2

    Ñ = isodd(N) ? N : N+1
    Us = zeros(T,(Ñ+1)÷2,2Ñ-1,K)

    for i in 1:2N-1
        m = ms[i]
        γs = [(one(T)-(points[k+1]/points[k+2])^2)*(points[k+1]/points[k+2])^m / (one(T)-(points[k]/points[k+1])^2) for k in 1:K-1]
        Us[1:Ms[i],i,1] = us[i][1:Ms[i]]
        for k = 2:K Us[1:Ms[i],i,k] = [Us[2,i,k-1]/γs[k-1]; us[i][Ms[i]+(k-2)*(Ms[i]-1)+1:Ms[i]+(k-1)*(Ms[i]-1)]] end
    end

    Ũs = zeros(T, (Ñ+1)÷2,2Ñ-1,K)
    for k in 1:K
        α, β = points[k], points[k+1]
        ρ = α / β
        for (m, i) in zip(ms, 1:2N-1)
            Ũs[1:Ms[i],i,k] = bubble2ann(α, β, m, Us[1:Ms[i],i,k])
        end
    end
    Us, Ũs
end


function finite_plotvalues(F::FiniteContinuousZernikeAnnulus{T}, us::AbstractVector) where T
    _, Ũs = _bubble2ann_all_modes(F, us)
    points = T.(F.points); N = F.N; K = length(points)-1
    θs=[]; rs=[]; vals = []   
    for k in 1:K
        α, β = points[k], points[k+1]
        ρ = α / β
        Z = ZernikeAnnulus{T}(ρ, one(T), one(T))
        # Scale the grid
        g = scalegrid(AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(N)), α, β)
        # Use fast transforms for synthesis
        FT = ZernikeAnnulusITransform{T}(N, 1, 1, 0, ρ)
        val = FT * pad(ModalTrav(Ũs[:,:,k]),axes(Z,2))[Block.(OneTo(N))]

        (θ, r, val) = plot_helper(g, val)
        append!(θs, [θ]); append!(rs, [r]); append!(vals, [val])
    end
    return (θs, rs, vals)
end