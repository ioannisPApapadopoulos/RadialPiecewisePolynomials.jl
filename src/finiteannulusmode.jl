struct FiniteContinuousZernikeAnnulusMode{T, N<:Int, P<:AbstractVector, M<:Int, J<:Int, B<:Int} <: Basis{T}
    N::N
    points::P
    m::M
    j::J
    b::B
end

function FiniteContinuousZernikeAnnulusMode{T}(N::Int, points::AbstractVector, m::Int, j::Int, b::Int) where {T}
    @assert length(points) > 1 && points == sort(points)
    @assert m ≥ 0
    @assert m == 0 ? j == 1 : 0 ≤ j ≤ 1
    FiniteContinuousZernikeAnnulusMode{T, Int, typeof(points), Int, Int, Int}(N, points, m, j, b)
end
FiniteContinuousZernikeAnnulusMode(N::Int, points::AbstractVector, m::Int, j::Int, b::Int) = FiniteContinuousZernikeAnnulusMode{Float64}(N, points, m, j, b)
FiniteContinuousZernikeAnnulusMode(N::Int, points::AbstractVector, m::Int, j::Int) = FiniteContinuousZernikeAnnulusMode{Float64}(N, points, m, j, m+2N)

axes(Z::FiniteContinuousZernikeAnnulusMode) = (Inclusion(annulus(first(Z.points), last(Z.points))), oneto(Z.N*(length(Z.points)-1)-(length(Z.points)-2)))
==(P::FiniteContinuousZernikeAnnulusMode, Q::FiniteContinuousZernikeAnnulusMode) = P.N == Q.N && P.points == Q.points && P.m == Q.m && P.j == Q.j && P.b == Q.b

function _getCs(points::AbstractVector{T}, m::Int, j::Int, b::Int) where T
    K = length(points)-1
    [ContinuousZernikeAnnulusElementMode{T}([points[k]; points[k+1]], m, j, b) for k in 1:K]
end

function ldiv(C::FiniteContinuousZernikeAnnulusMode{V}, f::AbstractQuasiVector) where V
    # T = promote_type(V, eltype(f))
    T = V
    points = T.(C.points); K = length(points)-1
    N = C.N; m = C.m; j = C.j;
    Cs = _getCs(points, m, j, C.b)
    fs = [C \ f.f.(axes(C, 1)) for C in Cs]
    f = fs[1][1:N]; for k in 2:K  f = vcat(f, fs[k][2:N]) end
    return f
end


###
# L2 inner product
###

# FIXME: Need to make type-safe
@simplify function *(A::QuasiAdjoint{<:Any,<:FiniteContinuousZernikeAnnulusMode}, B::FiniteContinuousZernikeAnnulusMode)
    T = promote_type(eltype(A), eltype(B))
    @assert A' == B

    points = T.(B.points); K = length(points)-1
    N = B.N; m = B.m; j = B.j;
    Cs = _getCs(points, m, j, B.b)

    Ms = [C' * C for C in Cs]
    M = Hcat(Matrix(Ms[1][1:N, 1:N]), zeros(N,(K-1)*(N-1)))
    
    if K > 1
        # γs = [fs[i][2] / fs[i+1][1] for i in 1:K-1]
        γs = [(one(T)-(points[k+1]/points[k+2])^2)*(points[k+1]/points[k+2])^m / (one(T)-(points[k]/points[k+1])^2) for k in 1:K-1]
        for k in 2:K
            M = Matrix(Vcat(M, Hcat(zeros(N-1, N+(k-2)*(N-1)), Ms[k][2:N, 2:N], zeros(N-1, (K-k)*(N-1)))))
        end

        M[2, 2] = M[2,2] + Ms[2][1,1] / γs[1]^2
        M[2, N+1:N+3] = Ms[2][1,2:4] / γs[1]
        M[N+1:N+3,2] = Ms[2][2:4,1] / γs[1]

        for k in 2:K-1
            M[N+(k-2)*(N-1)+1, N+(k-2)*(N-1)+1] += Ms[k+1][1,1] / γs[k]^2
            M[N+(k-2)*(N-1)+1, N+(k-1)*(N-1)+1:N+(k-1)*(N-1)+3] = Ms[k+1][1,2:4] / γs[k]
            M[N+(k-1)*(N-1)+1:N+(k-1)*(N-1)+3, N+(k-2)*(N-1)+1] = Ms[k+1][2:4,1] / γs[k]
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

@simplify function *(D::Derivative, C::FiniteContinuousZernikeAnnulusMode)
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
        γs = [(one(T)-(points[k+1]/points[k+2])^2)*(points[k+1]/points[k+2])^m / (one(T)-(points[k]/points[k+1])^2) for k in 1:K-1]
        for k in 2:K
            Δ = Matrix(Vcat(Δ, Hcat(zeros(N-1, N+(k-2)*(N-1)), Δs[k][2:N, 2:N], zeros(N-1, (K-k)*(N-1)))))
        end

        Δ[2, 2] = Δ[2,2] + Δs[2][1,1] / γs[1]^2
        Δ[2, N+1:N+3] = Δs[2][1,2:4] / γs[1]
        Δ[N+1:N+3,2] = Δs[2][2:4,1] / γs[1]

        for k in 2:K-1  
            Δ[N+(k-2)*(N-1)+1, N+(k-2)*(N-1)+1] += Δs[k+1][1,1] / γs[k]^2
            Δ[N+(k-2)*(N-1)+1,N+(k-1)*(N-1)+1:N+(k-1)*(N-1)+3] = Δs[k+1][1,2:4] / γs[k]
            Δ[N+(k-1)*(N-1)+1:N+(k-1)*(N-1)+3, N+(k-2)*(N-1)+1] = Δs[k+1][2:4,1] / γs[k]
        end
    end

    return Δ
end

function zero_dirichlet_bcs!(F::FiniteContinuousZernikeAnnulusMode{T}, Δ::AbstractMatrix{T}, Mf::AbstractVector{T}) where T
    N = F.N; K = length(F.points)-1
    Δ[:,1].=0; Δ[1,:].=0; Δ[1,1]=1.; 
    Δ[N+(K-2)*(N-1)+1,:].=0; Δ[:,N+(K-2)*(N-1)+1].=0; Δ[N+(K-2)*(N-1)+1,N+(K-2)*(N-1)+1]=1.;
    Mf[1]=0; Mf[N+(K-2)*(N-1)+1]=0;
end

###
# Plotting
###

function element_plotvalues(u::ApplyQuasiVector{T,typeof(*),<:Tuple{FiniteContinuousZernikeAnnulusMode, AbstractVector}}) where T
    C, u = u.args 
    points = T.(C.points); K = length(points)-1
    N = C.N; m = C.m; j = C.j
    Cs = _getCs(points, m, j, C.b)

    γs = [(one(T)-(points[k+1]/points[k+2])^2)*(points[k+1]/points[k+2])^m / (one(T)-(points[k]/points[k+1])^2) for k in 1:K-1]

    uc = [pad(u[1:N], axes(Cs[1],2))]
    for k = 2:K append!(uc, [pad([uc[k-1][2]/γs[k-1]; u[N+(k-2)*(N-1)+1:N+(k-1)*(N-1)]], axes(Cs[k],2))]) end
    
    θs=[]; rs=[]; valss=[];
    for k in 1:K
        (x, vals) = plotvalues(Cs[k]*uc[k])
        (θ, r, vals) =  plotannulus(x, vals)
        append!(θs,[θ]); append!(rs, [r]); append!(valss, [vals])
    end
    
    return (uc, θs, rs, valss)
end