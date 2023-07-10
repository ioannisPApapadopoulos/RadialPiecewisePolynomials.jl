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
    @assert length(L₁₁) == length(L₀₁) == length(L₁₀) == length(D) == length(points)-1
    FiniteContinuousZernikeMode{T}(N, points, m, j, L₁₁, L₀₁, L₁₀, D, b)
end

function FiniteContinuousZernikeMode(N::Int, points::AbstractVector{T}, m::Int, j::Int) where {T}
    K = length(points)-1
    ρs = []
    for k = 1:length(points)-1
        α, β = convert(T, first(points[k])), convert(T, last(points[k+1]))
        append!(ρs, [α / β])
    end

    ts = inv.(one(T) .- ρs.^2)
    Ls = _ann2element_via_lowering.(ts, m)
    L₁₁ = NTuple{K, AbstractMatrix}(first.(Ls))
    L₀₁ = NTuple{K, AbstractMatrix}([Ls[k][2] for k in 1:K])
    L₁₀ = NTuple{K, AbstractMatrix}(last.(Ls))

    Z = ZernikeAnnulus{T}.(ρs,1,1)
    D = (Z .\ (Laplacian.(axes.(Z,1)).*Weighted.(Z)))
    D = NTuple{K, AbstractMatrix}([Ds.ops[m+1] for Ds in D])

    FiniteContinuousZernikeMode(N, points, m, j, L₁₁, L₀₁, L₁₀, D, m+2N) 
end

# FiniteContinuousZernikeMode(N::Int, points::AbstractVector, m::Int, j::Int, L₁₁, L₀₁, L₁₀, D, b::Int) = FiniteContinuousZernikeMode{Float64}(N, points, m, j, L₁₁, L₀₁, L₁₀, D, b)
# FiniteContinuousZernikeMode(N::Int, points::AbstractVector, m::Int, j::Int, L₁₁, L₀₁, L₁₀, D) = FiniteContinuousZernikeMode{Float64}(N, points, m, j, L₁₁, L₀₁, L₁₀, D, m+2N)

function axes(Z::FiniteContinuousZernikeMode{T}) where T
    first(Z.points) ≈ 0 && return (Inclusion(last(Z.points)*UnitDisk{T}()), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
    (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
end
==(P::FiniteContinuousZernikeMode, Q::FiniteContinuousZernikeMode) = P.N == Q.N && P.points == Q.points && P.m == Q.m && P.j == Q.j && P.b == Q.b

function _getCs(points::AbstractVector{T}, m::Int, j::Int, b::Int, L₁₁, L₀₁, L₁₀, D) where T
    K = length(points)-1
    first(points) > 0 && return [ContinuousZernikeAnnulusElementMode([points[k]; points[k+1]], m, j, L₁₁[k], L₀₁[k], L₁₀[k], D[k], b) for k in 1:K]
    
    error("Fix disk element.")
    append!(Any[ContinuousZernikeElementMode{T}([points[1]; points[2]], m, j)], [ContinuousZernikeAnnulusElementMode{T}([points[k]; points[k+1]], m, j, b) for k in 2:K])
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
    return append!([(one(T)-(points[2]/points[3])^2)*(points[2]/points[3])^m / (sqrt(convert(T,2)^(m+3-iszero(m))/π) * normalizedjacobip(0, 1, m, 1.0))],γ)
end

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


function ldiv(F::FiniteContinuousZernikeMode{V}, f::AbstractQuasiVector) where V
    # T = promote_type(V, eltype(f))

    T = V
    points = T.(F.points); K = length(points)-1
    N = F.N; m = F.m; j = F.j;
    Cs = _getCs(F)
    fs = [C \ f.f.(axes(C, 1)) for C in Cs]

    f = fs[K][2:N]; for k in K-1:-1:2  f = vcat(fs[k+1][1], fs[k][3:N], f) end

    if first(points) ≈ 0
        f = vcat(fs[2][1], fs[1][2:N], f)
    else
        f = vcat(fs[1][1], fs[2][1], fs[1][3:N], f)
    end

    return f
end


###
# L2 inner product
###
function _piece_element_matrix(Ms, N::Int, K::Int, m::Int, points::AbstractVector{T}) where T
    M = Hcat(Matrix(Ms[1][1:N, 1:N]), zeros(N,(K-1)*(N-1)))

    if K > 1
        γs = _getγs(points, m)
        append!(γs, one(T))
        for k in 2:K
            M = Matrix(Vcat(M, Hcat(zeros(N-1, N+(k-2)*(N-1)), Ms[k][2:N, 2:N], zeros(N-1, (K-k)*(N-1)))))
        end

        i = first(points) ≈ 0 ? 1 : 2 # disk or annulus?
        M[i, 1:i+2] *= γs[1] # Convert the left-side hat function coefficients for continuity
        M[1:i+2, i] *= γs[1]
        M[i, i] += Ms[2][1,1] # Add the contribution from the right-side of the hat function

        # Right-side of hat function with left-side of hat function in next element
        M[i, N+1] = Ms[2][1,2]*γs[2]
        M[N+1, i] = Ms[2][2,1]*γs[2]

        # Right-side of hat function interaction with bubble functions
        M[i, N+2:N+3] = Ms[2][1,3:4]
        M[N+2:N+3,i] = Ms[2][3:4,1]

        b = min(N-1, 3)
        for k in 2:K-1
            # Convert left-side of hat function coefficients for continuity
            M[N+(k-2)*(N-1)+1, N+(k-2)*(N-1)+1:N+(k-2)*(N-1)+b] *= γs[k]
            M[N+(k-2)*(N-1)+1:N+(k-2)*(N-1)+b, N+(k-2)*(N-1)+1] *= γs[k]
            M[N+(k-2)*(N-1)+1, N+(k-2)*(N-1)+1] += Ms[k+1][1,1] # add contribution of right-side of hat function

            # Right-side of hat function with left-side of hat function in next element
            M[N+(k-2)*(N-1)+1, N+(k-1)*(N-1)+1] = Ms[k+1][1,2]*γs[k+1]
            M[N+(k-1)*(N-1)+1, N+(k-2)*(N-1)+1] = Ms[k+1][2,1]*γs[k+1]

            # Right-side of hat function interaction with bubble functions
            M[N+(k-2)*(N-1)+1, N+(k-1)*(N-1)+2:N+(k-1)*(N-1)+b] = Ms[k+1][1,3:b+1]
            M[N+(k-1)*(N-1)+2:N+(k-1)*(N-1)+b, N+(k-2)*(N-1)+1] = Ms[k+1][3:b+1,1]
        end
    end
    return M
end

# FIXME: Need to make type-safe
@simplify function *(A::QuasiAdjoint{<:Any,<:FiniteContinuousZernikeMode}, B::FiniteContinuousZernikeMode)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B

    points = T.(B.points); K = length(points)-1
    N = B.N; m = B.m;
    Cs = _getCs(B)

    Ms = [C' * C for C in Cs]

    _piece_element_matrix(Ms, N, K, m, points)
end

###
# Gradient for constructing weak Laplacian.
###

struct GradientFiniteContinuousZernikeAnnulusMode{T}<:Basis{T}
    F::FiniteContinuousZernikeMode{T}
end

# GradientFiniteContinuousZernikeAnnulusMode(F::FiniteContinuousZernikeMode{T}) where T =  GradientFiniteContinuousZernikeAnnulusMode{T}(F)

axes(Z:: GradientFiniteContinuousZernikeAnnulusMode) = (Inclusion(last(Z.F.points)*UnitDisk{eltype(Z)}()), oneto(Z.F.N*(length(Z.F.points)-1)-(length(Z.F.points)-2)))
==(P::GradientFiniteContinuousZernikeAnnulusMode, Q::GradientFiniteContinuousZernikeAnnulusMode) = P.F.points == Q.F.points && P.F.m == Q.F.m && P.F.j == Q.F.j && P.F.b == Q.F.b

@simplify function *(D::Derivative, F::FiniteContinuousZernikeMode)
    GradientFiniteContinuousZernikeAnnulusMode(F)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:GradientFiniteContinuousZernikeAnnulusMode}, B::GradientFiniteContinuousZernikeAnnulusMode)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B
    F = B.F

    points = T.(F.points); K = length(points)-1
    N = F.N; m = F.m; j = F.j;
    Cs = _getCs(F)
    
    xs = [axes(C,1) for C in Cs]
    Ds = [Derivative(x) for x in xs]
    Δs = [(D*C)' * (D*C) for (C, D) in zip(Cs, Ds)]

    return _piece_element_matrix(Δs, N, K, m, points)
end

function zero_dirichlet_bcs!(F::FiniteContinuousZernikeMode{T}, Δ::AbstractMatrix{T}, Mf::AbstractVector{T}) where T
    N = F.N; points = F.points; K = length(points)-1;
    if !(first(points) ≈  0)
        Δ[:,1].=0
        Δ[1,:].=0
        Δ[1,1]=1.
        Mf[1]=0
    end
    Δ[N+(K-2)*(N-1)+1,:].=0; Δ[:,N+(K-2)*(N-1)+1].=0; Δ[N+(K-2)*(N-1)+1,N+(K-2)*(N-1)+1]=1.;
    Mf[N+(K-2)*(N-1)+1]=0;
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
        uc = [pad([u[1];u[2]*γs[1];u[3:N]], axes(Cs[1],2))]
        for k = 2:K append!(uc, [pad([u[N+(k-3)*(N-1)+1];u[N+(k-2)*(N-1)+1]*γs[k];u[N+(k-2)*(N-1)+2:N+(k-1)*(N-1)]], axes(Cs[k],2))]) end
    end
    
    θs=[]; rs=[]; valss=[];
    for k in 1:K
        (x, vals) = plotvalues(Cs[k]*uc[k])
        (θ, r, vals) =  plot_helper(x, vals)
        append!(θs,[θ]); append!(rs, [r]); append!(valss, [vals])
    end
    
    return (uc, θs, rs, valss)
end