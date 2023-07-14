using Test, RadialPiecewisePolynomials, Memoization
import RadialPiecewisePolynomials: _getγs

c1 = -10; c2 = 0; c3=0.6
function f0_(x, y)
    ρ = 0.2
    exp(c1*(x^2 + (y-c3)^2)) * (1-(x^2+y^2)) * ((x^2+y^2)-ρ^2)
end
function f0(xy)
    x,y = first(xy), last(xy)
    f0_(x,y)
end

@testset "finiteannulus" begin
    @testset "basics" begin
        N=10; F = FiniteContinuousZernike(N, [0.5; 0.7; 1])
        @test F isa FiniteContinuousZernike
        @test F.points == [0.5; 0.7; 1.0]
        @test F.N == N
    end

    # @testset "continuity" begin
    #     for (m, j) in zip([0, 1, 1, 4, 5], [1, 0, 1, 1, 0])
    #         C1 = ContinuousZernikeElementMode([0;0.3], m, j)
    #         C2 = ContinuousZernikeAnnulusElementMode([0.3;0.5], m, j)
    #         C3 = ContinuousZernikeAnnulusElementMode([0.5;0.8], m, j)  
    #         γs =_getγs([0;0.3;0.5;0.8], m)

    #         @test C1[[0.3, 0], 1] * γs[1] ≈ C2[[0.3+eps(), 0], 1]
    #         @test C2[[0.5-eps(), 0], 2] * γs[2] ≈ C3[[0.5+eps(), 0], 1]
    #     end
    # end

    @testset "expansion & mass & differentiation matrices" begin
        Memoization.empty_all_caches!()
        ρ = 0.2
        N = 50; points = [0.2; 0.5; 0.8; 1.0]
        K = length(points)-1

        # Just annuli elements
        F = FiniteContinuousZernike(N, points)
        fc = F \ f0.(axes(F,1))
        (θs, rs, vals) = finite_plotvalues(F, fc)
        vals_, err = inf_error(F, θs, rs, vals, f0)
        @test err < 1e-15
        M = F' * F
        @test length(M) == 2N-1
        @test size(M[1]) == (K*N/2-(K-1), K*N/2-(K-1))
        @test size(M[end]) == (7, 7)
        ∇ = Derivative(axes(F,1)); Δ = (∇*F)' * (∇*F)
        @test length(Δ) == 2N-1
        @test size(Δ[1]) == (K*N/2-(K-1), K*N/2-(K-1))
        @test size(Δ[end]) == (7, 7)
        @test sum(transpose.(fc) .* (M .* fc)) ≈  0.0055779595855720305
        @test sum(transpose.(fc) .* (Δ .* fc)) ≈  0.16873822986436868


        # disk + annuli elements
        N = 50; points = [0.0; 0.5; 0.8; 1.0]
        K = length(points)-1

        F = FiniteContinuousZernike(N, points)
        fc = F \ f0.(axes(F,1))
        (θs, rs, vals) = finite_plotvalues(F, fc)
        vals_, err = inf_error(F, θs, rs, vals, f0)
        @test err < 1e-15
        M = F' * F
        @test length(M) == 2N-1
        @test size(M[1]) == (K*(N/2-1), K*(N/2-1))
        @test size(M[end]) == (6, 6)
        ∇ = Derivative(axes(F,1)); Δ = (∇*F)' * (∇*F)
        @test length(Δ) == 2N-1
        @test size(Δ[1]) == (K*(N/2-1), K*(N/2-1))
        @test size(Δ[end]) == (6, 6)
        @test sum(transpose.(fc) .* (M .* fc)) ≈  0.005578088780274445
        @test sum(transpose.(fc) .* (Δ .* fc)) ≈ 0.16877589535690113
    end

end