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
    Ms = ((N + 1 .- ms) .÷ 2)
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
# Plotting
##

function finite_plotvalues(F::FiniteContinuousZernikeAnnulus{T}, us::AbstractVector) where T
    points = T.(F.points); N = F.N; K = length(points)-1
    # Fs = _getFs(N, points)

    # # (uc, θs, rs, valss)
    # γs = zeros(K-1, N)
    # for k in 1:K-1
    #     for m in 1:N
    #         γs[k, m] = (one(T)-(points[k+1]/points[k+2])^2)*(points[k+1]/points[k+2])^m / (one(T)-(points[k]/points[k+1])^2)
    #     end
    # end

    # # for k in 1:K

    # θs=[]; rs=[]; valss=[];
    # for k in 1:K
    #     (x, vals) = plotvalues(Cs[k]*uc[k])
    #     (θ, r, vals) =  plotannulus(x, vals)
    #     append!(θs,[θ]); append!(rs, [r]); append!(valss, [vals])
    # end
    
    # return (uc, θs, rs, valss)
    points = T.(F.points); N = F.N; K = length(points)-1
    # Fs = _getFs(N, po
    Fs = _getFs(N, points)
    m_vals = []
    θs = []
    rs = []
    for i in 1:lastindex(us)
        (_, θ, r, vals) = element_plotvalues(Fs[i]*us[i])
        append!(m_vals, [vals])
        append!(θs, [θ])
        append!(rs, [r])
    end 
    return (θs, rs, m_vals)
end