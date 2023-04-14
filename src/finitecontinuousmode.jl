struct FiniteContinuousZernikeMode{T, N<:Int, P<:AbstractVector, M<:Int, J<:Int, B<:Int} <: Basis{T}
    N::N
    points::P
    m::M
    j::J
    b::B
end

function FiniteContinuousZernikeMode{T}(N::Int, points::AbstractVector, m::Int, j::Int, b::Int) where {T}
    @assert length(points) > 1 && points == sort(points)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    FiniteContinuousZernikeMode{T, Int, typeof(points), Int, Int, Int}(N, points, m, j, b)
end
FiniteContinuousZernikeMode(N::Int, points::AbstractVector, m::Int, j::Int, b::Int) = FiniteContinuousZernikeMode{Float64}(N, points, m, j, b)
FiniteContinuousZernikeMode(N::Int, points::AbstractVector, m::Int, j::Int) = FiniteContinuousZernikeMode{Float64}(N, points, m, j, m+2N)

function axes(Z::FiniteContinuousZernikeMode{T}) where T
    first(Z.points) ≈ 0 && return (Inclusion(last(Z.points)*UnitDisk{T}()), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
    (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
end
==(P::FiniteContinuousZernikeMode, Q::FiniteContinuousZernikeMode) = P.N == Q.N && P.points == Q.points && P.m == Q.m && P.j == Q.j && P.b == Q.b

function _getCs(points::AbstractVector{T}, m::Int, j::Int, b::Int) where T
    K = length(points)-1
    first(points) > 0 && return [ContinuousZernikeAnnulusElementMode{T}([points[k]; points[k+1]], m, j, b) for k in 1:K]
    append!(Any[ContinuousZernikeElementMode{T}([points[1]; points[2]], m, j)], [ContinuousZernikeAnnulusElementMode{T}([points[k]; points[k+1]], m, j, b) for k in 2:K])
end

function _getγs(points::AbstractArray{T}, m::Int) where T
    K = length(points)-1
    first(points) > 0 && return [(one(T)-(points[k+1]/points[k+2])^2)*(points[k+1]/points[k+2])^m / (one(T)-(points[k]/points[k+1])^2) for k in 1:K-1]
    γ = [(one(T)-(points[k+1]/points[k+2])^2)*(points[k+1]/points[k+2])^m / (one(T)-(points[k]/points[k+1])^2) for k in 2:K-1]
    return append!([(one(T)-(points[2]/points[3])^2)*(points[2]/points[3])^m / sqrt(T(2)/convert(T,π))],γ)
end

function ldiv(F::FiniteContinuousZernikeMode{V}, f::AbstractQuasiVector) where V
    # T = promote_type(V, eltype(f))
    T = V
    points = T.(F.points); K = length(points)-1
    N = F.N; m = F.m; j = F.j;
    Cs = _getCs(points, m, j, F.b)
    fs = [C \ f.f.(axes(C, 1)) for C in Cs]
    f = fs[1][1:N]; for k in 2:K  f = vcat(f, fs[k][2:N]) end
    return f
end


###
# L2 inner product
###

# FIXME: Need to make type-safe
@simplify function *(A::QuasiAdjoint{<:Any,<:FiniteContinuousZernikeMode}, B::FiniteContinuousZernikeMode)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B

    points = T.(B.points); K = length(points)-1
    N = B.N; m = B.m; j = B.j;
    Cs = _getCs(points, m, j, B.b)

    Ms = [C' * C for C in Cs]
    M = Hcat(Matrix(Ms[1][1:N, 1:N]), zeros(N,(K-1)*(N-1)))
    
    if K > 1
        γs = _getγs(points, m)
        for k in 2:K
            M = Matrix(Vcat(M, Hcat(zeros(N-1, N+(k-2)*(N-1)), Ms[k][2:N, 2:N], zeros(N-1, (K-k)*(N-1)))))
        end

        i = first(points) ≈ 0 ? 1 : 2
        M[i, i] = M[i,i] + Ms[2][1,1] / γs[1]^2
        M[i, N+1:N+3] = Ms[2][1,2:4] / γs[1]
        M[N+1:N+3,i] = Ms[2][2:4,1] / γs[1]

        b = min(N-1, 3)
        for k in 2:K-1
            M[N+(k-2)*(N-1)+1, N+(k-2)*(N-1)+1] += Ms[k+1][1,1] / γs[k]^2
            M[N+(k-2)*(N-1)+1, N+(k-1)*(N-1)+1:N+(k-1)*(N-1)+b] = Ms[k+1][1,2:b+1] / γs[k]
            M[N+(k-1)*(N-1)+1:N+(k-1)*(N-1)+b, N+(k-2)*(N-1)+1] = Ms[k+1][2:b+1,1] / γs[k]
        end
    end
    return M
end

###
# Gradient for constructing weak Laplacian.
###

struct GradientFiniteContinuousZernikeAnnulusMode{T, N<:Int, P<:AbstractVector, M<:Int, J<:Int, B<:Int}<:Basis{T}
    N::N
    points::P
    m::M
    j::J
    b::B
end

GradientFiniteContinuousZernikeAnnulusMode{T}(N::Int, points::AbstractVector, m::Int, j::Int, b::Int) where {T} =  GradientFiniteContinuousZernikeAnnulusMode{T,Int, typeof(points), Int, Int, Int}(N, points, m, j, b)
GradientFiniteContinuousZernikeAnnulusMode(N::Int, points::AbstractVector, m::Int, j::Int, b::Int) =  GradientFiniteContinuousZernikeAnnulusMode{Float64}(N, points, m, j, b)
GradientFiniteContinuousZernikeAnnulusMode(N::Int, points::AbstractVector, m::Int, j::Int) = GradientFiniteContinuousZernikeAnnulusMode{Float64}(N, points, m, j, m+2N)

axes(Z:: GradientFiniteContinuousZernikeAnnulusMode) = (Inclusion(last(Z.points)*UnitDisk{eltype(Z)}()), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
==(P:: GradientFiniteContinuousZernikeAnnulusMode, Q:: GradientFiniteContinuousZernikeAnnulusMode) = P.points == Q.points && P.m == Q.m && P.j == Q.j && P.b == Q.b

@simplify function *(D::Derivative, C::FiniteContinuousZernikeMode)
    GradientFiniteContinuousZernikeAnnulusMode(C.N, C.points, C.m, C.j, C.b)
end

@simplify function *(A::QuasiAdjoint{<:Any,<:GradientFiniteContinuousZernikeAnnulusMode}, B::GradientFiniteContinuousZernikeAnnulusMode)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B

    points = T.(B.points); K = length(points)-1
    N = B.N; m = B.m; j = B.j;
    Cs = _getCs(points, m, j, B.b)
    
    xs = [axes(C,1) for C in Cs]
    Ds = [Derivative(x) for x in xs]
    Δs = [(D*C)' * (D*C) for (C, D) in zip(Cs, Ds)]


    Δ = Hcat(Matrix(Δs[1][1:N, 1:N]), zeros(N,(K-1)*(N-1)))
    if K > 1
        γs = _getγs(points, m)
        for k in 2:K
            Δ = Matrix(Vcat(Δ, Hcat(zeros(N-1, N+(k-2)*(N-1)), Δs[k][2:N, 2:N], zeros(N-1, (K-k)*(N-1)))))
        end

        i = first(points) ≈ 0 ? 1 : 2
        Δ[i, i] = Δ[i,i] + Δs[2][1,1] / γs[1]^2
        Δ[i, N+1:N+2] = Δs[2][1,2:3] / γs[1]
        Δ[N+1:N+2,i] = Δs[2][2:3,1] / γs[1]

        b = min(N-1, 2)
        for k in 2:K-1  
            Δ[N+(k-2)*(N-1)+1, N+(k-2)*(N-1)+1] += Δs[k+1][1,1] / γs[k]^2
            Δ[N+(k-2)*(N-1)+1,N+(k-1)*(N-1)+1:N+(k-1)*(N-1)+b] = Δs[k+1][1,2:b+1] / γs[k]
            Δ[N+(k-1)*(N-1)+1:N+(k-1)*(N-1)+b, N+(k-2)*(N-1)+1] = Δs[k+1][2:b+1,1] / γs[k]
        end
    end

    return Δ
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
    Cs = _getCs(points, m, j, C.b)

    γs = _getγs(points, m)

    uc = [pad(u[1:N], axes(Cs[1],2))]
    if first(points) ≈ 0 && K > 1
        k=2; append!(uc, [pad([uc[k-1][1]/γs[k-1]; u[N+(k-2)*(N-1)+1:N+(k-1)*(N-1)]], axes(Cs[k],2))]) 
        for k = 3:K append!(uc, [pad([uc[k-1][2]/γs[k-1]; u[N+(k-2)*(N-1)+1:N+(k-1)*(N-1)]], axes(Cs[k],2))]) end
    else
        for k = 2:K append!(uc, [pad([uc[k-1][2]/γs[k-1]; u[N+(k-2)*(N-1)+1:N+(k-1)*(N-1)]], axes(Cs[k],2))]) end
    end
    
    θs=[]; rs=[]; valss=[];
    for k in 1:K
        (x, vals) = plotvalues(Cs[k]*uc[k])
        (θ, r, vals) =  plot_helper(x, vals)
        append!(θs,[θ]); append!(rs, [r]); append!(valss, [vals])
    end
    
    return (uc, θs, rs, valss)
end