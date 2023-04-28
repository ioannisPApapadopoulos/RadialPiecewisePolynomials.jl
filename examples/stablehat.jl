using SemiclassicalOrthogonalPolynomials, ClassicalOrthogonalPolynomials, Plots

zernikeannulusr(ρ, ℓ, m, a, b, r::T) where T = r^m * SemiclassicalJacobi{T}(inv(1-ρ^2),b,a,m)[(r^2 - 1)/(ρ^2 - 1), (ℓ-m) ÷ 2 + 1]

T = Float64
ρ = 0.5; t = inv(one(T)-ρ^2)
r = range(ρ,1;length=100)
m = 15; plot(r, zernikeannulusr.(ρ, m, m, 0, 0, r))
plot!(r, zernikeannulusr.(ρ, m+2, m, 0, 0, r))
plot!(r, zernikeannulusr.(ρ, m+4, m, 0, 0, r))
plot!(r, zernikeannulusr.(ρ, round(Int,m/ρ), m, 0, 0, r))


Q₀₀ = SemiclassicalJacobi{T}(t, 0, 0, m)
Q₀₁ = SemiclassicalJacobi{T}(t, 0, 1, m)
Q₁₀ = SemiclassicalJacobi{T}(t, 1, 0, m)
Q₁₁ = SemiclassicalJacobi{T}(t, 1, 1, m)


L₁₁ = (Weighted(Q₀₀) \ Weighted(Q₁₁)) / t^2

inv(L₁₁[1:100,1:100])

N = 100; inv(L₁₁[2:N+1,1:N])

M = 5; N = 100; norm(inv(L₁₁[M+2:N+2,M:N]),Inf)

M = 15; N = 100; L₁₁[M+2:N+2,M:N]




