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
    first(Z.points) ≈ 0 && return (Inclusion(last(Z.points)*UnitDisk{T}()), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
    (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
end
==(P::FiniteContinuousZernike, Q::FiniteContinuousZernike) = P.N == Q.N && P.points == Q.points


function _ann2element_via_Jacobi(t::T) where T
    Q₁₁ = SemiclassicalJacobi{T}.(t, 1, 1, 0:∞)
    X = jacobimatrix.(Q₁₁)

    L₁₁ = (X .- X .* X)/t^2
    L₀₁ = (Fill(I, ∞) .- X)/t
    L₁₀ = X/t

    (L₁₁, L₀₁, L₁₀)
end

function _getFs(N::Int, points::AbstractVector{T}) where T
    # Ordered list of Fourier modes (ms, js) and correct length for each Fourier mode Ms.
    ms = ((0:2N) .÷ 2)[2:end-1]
    js = repeat([0; 1], N)[2:end]
    Ms = ((N + 1 .- ms) .÷ 2); Ms[Ms .== 1] .= 2

    K = length(points)-1

    # List of radii of the annuli
    ρs = []
    for k = 1:length(points)-1
        α, β = convert(T, first(points[k])), convert(T, last(points[k+1]))
        append!(ρs, [α / β])
    end
    # Semiclassical Jacobi parameter t
    ts = inv.(one(T) .- ρs.^2)

    # Use broadcast notation to compute all the lowering matrices across all
    # intervals and Fourier modes simultaneously.
    Ls = _ann2element_via_Jacobi.(ts)

    # Use broadcast notation to compute all the derivative matrices across
    # all the intervals and Fourier modes simultaneously
    Z = ZernikeAnnulus{T}.(ρs,1,1)
    Ds = (Z .\ (Laplacian.(axes.(Z,1)).*Weighted.(Z)))

    # Loop over the Fourier modes
    Fs = []
    for (M, m, j) in zip(Ms, ms, js)
        # Extract the lowering and differentiation matrices associated
        # with each Fourier mode and store in the Tuples
        L₁₁ = NTuple{K, AbstractMatrix}([Ls[i][1][m+1] for i in 1:K])
        L₀₁ = NTuple{K, AbstractMatrix}([Ls[i][2][m+1] for i in 1:K])
        L₁₀ = NTuple{K, AbstractMatrix}([Ls[i][3][m+1] for i in 1:K])
        
        D = NTuple{K, AbstractMatrix}([(Ds[i]).ops[m+1] for i in 1:K])

        # Construct the structs for each Fourier mode seperately
        append!(Fs, [FiniteContinuousZernikeMode(M, points, m, j, L₁₁, L₀₁, L₁₀, D, N)])
    end
    return Fs
end

_getFs(F::FiniteContinuousZernike{T}) where T = _getFs(F.N, F.points)

function ldiv(F::FiniteContinuousZernike{V}, f::AbstractQuasiVector) where V
    # T = promote_type(V, eltype(f))
    @warn "Expanding via FiniteContinuousZernike is ill-conditioned, are you sure you want to expand your function in this basis?"
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


@simplify function *(A::QuasiAdjoint{<:Any,<:ZernikeAnnulus}, B::FiniteContinuousZernike)
    return 2.0
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

function zero_dirichlet_bcs!(F::FiniteContinuousZernike{T}, Δ::Vector{Matrix{T}}, Mf::Vector{Vector{T}}) where T
    Fs = _getFs(F.N, F.points)
    zero_dirichlet_bcs!.(Fs, Δ, Mf)
end

###
# Plotting
##
# This helper function takes the list of coefficient values from ldiv and converts them into 
# a 3-tensor of degree × Fourier mode × element. Us is the hat/bubble coeffiecients
# and Ũs are the corresponding ZernikeAnnulus(ρ,1,1) coefficients.
function _bubble2disk_or_ann_all_modes(F::FiniteContinuousZernike{T}, us::AbstractVector) where T
    points = T.(F.points); K = length(points)-1
    N = F.N;
    ms = ((0:2N) .÷ 2)[2:end-1]
    Ms = ((N + 1 .- ms) .÷ 2); Ms[Ms .== 1] .= 2

    Ñ = isodd(N) ? N : N+1
    Us = zeros(T,(Ñ+1)÷2,2Ñ-1,K)

    for i in 1:2N-1
        m = ms[i]
        γs = _getγs(points, m)
        append!(γs, one(T))
        if first(points) ≈ 0 && K > 1
            Us[1:Ms[i],i,1] = [us[i][1]*γs[1]; us[i][2:Ms[i]]]
            k=2; Us[1:Ms[i],i,k] = [us[i][1]; us[i][Ms[i]+(k-2)*(Ms[i]-1)+1]*γs[k]; us[i][Ms[i]+(k-2)*(Ms[i]-1)+2:Ms[i]+(k-1)*(Ms[i]-1)]]
            for k = 3:K Us[1:Ms[i],i,k] = [us[i][Ms[i]+(k-3)*(Ms[i]-1)+1]; us[i][Ms[i]+(k-2)*(Ms[i]-1)+1]*γs[k];us[i][Ms[i]+(k-2)*(Ms[i]-1)+2:Ms[i]+(k-1)*(Ms[i]-1)]] end
        else
            Us[1:Ms[i],i,1] = [us[i][1]; us[i][2]*γs[1]; us[i][3:Ms[i]]]
            for k = 2:K Us[1:Ms[i],i,k] = [us[i][Ms[i]+(k-3)*(Ms[i]-1)+1]; us[i][Ms[i]+(k-2)*(Ms[i]-1)+1]*γs[k];us[i][Ms[i]+(k-2)*(Ms[i]-1)+2:Ms[i]+(k-1)*(Ms[i]-1)]] end
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


function finite_plotvalues(F::FiniteContinuousZernike{T}, us::AbstractVector) where T
    _, Ũs = _bubble2disk_or_ann_all_modes(F, us)
    points = T.(F.points); N = F.N; K = length(points)-1
    θs=[]; rs=[]; vals = []   
    for k in 1:K
        if k == 1 && first(points) ≈ 0
            ρ = points[2]
            Z = Zernike{T}(0,1)
            g = scalegrid(AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(2N)), ρ)
            FT = ZernikeITransform{T}(2N, 0, 1)
            val = FT * pad(ModalTrav(Ũs[:,:,k]),axes(Z,2))[Block.(OneTo(2N))]
        else
            α, β = points[k], points[k+1]
            ρ = α / β
            Z = ZernikeAnnulus{T}(ρ, one(T), one(T))
            # Scale the grid
            g = scalegrid(AlgebraicCurveOrthogonalPolynomials.grid(Z, Block(N)), α, β)
            # Use fast transforms for synthesis
            FT = ZernikeAnnulusITransform{T}(N, 1, 1, 0, ρ)
            val = FT * pad(ModalTrav(Ũs[:,:,k]),axes(Z,2))[Block.(OneTo(N))]
        end
        (θ, r, val) = plot_helper(g, val)
        append!(θs, [θ]); append!(rs, [r]); append!(vals, [val])
    end
    return (θs, rs, vals)
end