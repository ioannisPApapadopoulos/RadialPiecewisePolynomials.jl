using Test, RadialPiecewisePolynomials, Memoization
import RadialPiecewisePolynomials: _getγs

f0(xy) = exp(-first(xy)^2-last(xy)^2)
f6(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)^6*cos(6*atan(last(xy), first(xy)))
f1c(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)*cos(atan(last(xy), first(xy)))
f1s(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)*sin(atan(last(xy), first(xy)))


@testset "annulusmode" begin
    @testset "basics" begin
        N=10; F = ContinuousZernikeMode(N, [0.5; 0.7; 1], 0, 1)
        @test F isa ContinuousZernikeMode
        @test F.points == [0.5; 0.7; 1.0]
        @test F.m == 0
        @test F.j == 1
        @test F.N == N
    end

    @testset "continuity" begin
        for (m, j) in zip([0, 1, 1, 4, 5], [1, 0, 1, 1, 0])
            C1 = ContinuousZernikeElementMode([0;0.3], m, j)
            C2 = ContinuousZernikeAnnulusElementMode([0.3;0.5], m, j)
            C3 = ContinuousZernikeAnnulusElementMode([0.5;0.8], m, j)  
            γs =_getγs([0;0.3;0.5;0.8], m)

            @test C1[[0.3, 0], 1] * γs[1] ≈ C2[[0.3+eps(), 0], 1]
            @test C2[[0.5-eps(), 0], 2] * γs[2] ≈ C3[[0.5+eps(), 0], 1]
        end
    end

    @testset "expansion & mass & differentiation matrices" begin
        Memoization.empty_all_caches!()

        ρ = 0.2
        N = 20; points = [0.2; 0.5; 0.8; 1.0]
        s = 0.2^(-1/3)
        equi_points = reverse([s^(-j) for j in 0:3])
        K = length(points)-1

        # Just annuli elements
        F = ContinuousZernikeMode(N, points, 0, 1)
        fc = F \ f0.(axes(F,1))
        (uc, θs, rs, vals) = element_plotvalues(F*fc)
        vals_, err = inf_error(F, θs, rs, vals, f0)
        @test err < 1e-9
        M = F' * F
        @test size(M) == (K*N-(K-1), K*N-(K-1))
        ∇ = Derivative(axes(F,1)); Δ = (∇*F)' * (∇*F)
        @test size(Δ) == (K*N-(K-1), K*N-(K-1))
        @test fc' * (M * fc) ≈  π/2 * (exp(-2*0.2^2) - exp(-2))
        @test fc' * (Δ * fc) ≈  1.856554980031349

        F = ContinuousZernikeMode(N, points, 1, 0)
        fc = F \ f1s.(axes(F,1))
        (uc, θs, rs, vals) = element_plotvalues(F*fc)
        vals_, err = inf_error(F, θs, rs, vals, f1s)
        @test err < 1e-9
        M = F' * F
        @test size(M) == (K*N-(K-1), K*N-(K-1))
        ∇ = Derivative(axes(F,1)); Δ = (∇*F)' * (∇*F)
        @test size(Δ) == (K*N-(K-1), K*N-(K-1))
        @test fc' * (M * fc) ≈ 0.2320693725039186
        @test fc' * (Δ * fc) ≈ 0.816915357578546

        Memoization.empty_all_caches!()
        F = ContinuousZernikeMode(N, points, 6, 1)
        fc = F \ f6.(axes(F,1))
        (uc, θs, rs, vals) = element_plotvalues(F*fc)
        vals_, err = inf_error(F, θs, rs, vals, f6)
        @test err < 1e-9
        M = F' * F
        @test size(M) == (K*N-(K-1), K*N-(K-1))
        ∇ = Derivative(axes(F,1)); Δ = (∇*F)' * (∇*F)
        @test size(Δ) == (K*N-(K-1), K*N-(K-1))
        @test fc' * (M * fc) ≈ 0.04005947846778158
        @test fc' * (Δ * fc) ≈ 2.686674285690333

        F = ContinuousZernikeMode(N, equi_points, 6, 1, same_ρs=true)
        @test F.same_ρs == true
        fc = F \ f6.(axes(F,1))
        (uc, θs, rs, vals) = element_plotvalues(F*fc)
        vals_, err = inf_error(F, θs, rs, vals, f6)
        @test err < 1e-9
        M = F' * F
        @test size(M) == (K*N-(K-1), K*N-(K-1))
        ∇ = Derivative(axes(F,1)); Δ = (∇*F)' * (∇*F)
        @test size(Δ) == (K*N-(K-1), K*N-(K-1))
        @test fc' * (M * fc) ≈ 0.04005947846778158
        @test fc' * (Δ * fc) ≈ 2.686674285690333

        # disk + annuli elements
        N = 20; points = [0.0; 0.5; 0.8; 1.0]
        s = 0.5^(-1/2)
        equi_points = [0.0; reverse([s^(-j) for j in 0:2])]
        K = length(points)-1

        F = ContinuousZernikeMode(N, points, 0, 1)
        fc = F \ f0.(axes(F,1))
        (uc, θs, rs, vals) = element_plotvalues(F*fc)
        vals_, err = inf_error(F, θs, rs, vals, f0)
        @test err < 1e-9
        M = F' * F
        @test size(M) == (K*(N-1), K*(N-1))
        ∇ = Derivative(axes(F,1)); Δ = (∇*F)' * (∇*F)
        @test size(Δ) == (K*(N-1), K*(N-1))
        @test fc' * (M * fc) ≈  π/2 * (1.0 - exp(-2))
        @test fc' * (Δ * fc) ≈ 1.866087658826886

        F = ContinuousZernikeMode(N, points, 1, 0)
        fc = F \ f1s.(axes(F,1))
        (uc, θs, rs, vals) = element_plotvalues(F*fc)
        vals_, err = inf_error(F, θs, rs, vals, f1s)
        @test err < 1e-9
        M = F' * F
        @test size(M) == (K*(N-1), K*(N-1))
        ∇ = Derivative(axes(F,1)); Δ = (∇*F)' * (∇*F)
        @test size(Δ) == (K*(N-1), K*(N-1))
        @test fc' * (M * fc) ≈ 0.233260957353361
        @test fc' * (Δ * fc) ≈ 0.933043829413443

        Memoization.empty_all_caches!()
        F = ContinuousZernikeMode(N, points, 6, 1)
        fc = F \ f6.(axes(F,1))
        (uc, θs, rs, vals) = element_plotvalues(F*fc)
        vals_, err = inf_error(F, θs, rs, vals, f6)
        @test err < 1e-9
        M = F' * F
        @test size(M) == (K*(N-1), K*(N-1))
        ∇ = Derivative(axes(F,1)); Δ = (∇*F)' * (∇*F)
        @test size(Δ) == (K*(N-1), K*(N-1))
        @test fc' * (M * fc) ≈ 0.04005947850206709
        @test fc' * (Δ * fc) ≈ 2.686674356967118

        F = ContinuousZernikeMode(N, equi_points, 6, 1, same_ρs=true)
        @test F.same_ρs == true
        fc = F \ f6.(axes(F,1))
        (uc, θs, rs, vals) = element_plotvalues(F*fc)
        vals_, err = inf_error(F, θs, rs, vals, f6)
        @test err < 1e-9
        M = F' * F
        @test size(M) == (K*(N-1), K*(N-1))
        ∇ = Derivative(axes(F,1)); Δ = (∇*F)' * (∇*F)
        @test size(Δ) == (K*(N-1), K*(N-1))
        @test fc' * (M * fc) ≈ 0.04005947850206709
        @test fc' * (Δ * fc) ≈ 2.686674356967118
    end

    @testset "assembly" begin
        Memoization.empty_all_caches!()

        ρ = 0.2
        N = 20; points = [0.2; 0.5; 0.8]
        K = length(points)-1

        # Just annuli elements
        F = ContinuousZernikeMode(N, points, 0, 1)
        fc = F \ f0.(axes(F,1))
        λ(r²) = r²
        M = F' * (λ.(axes(F,1)) .*F)
        @test size(M) == (K*N-(K-1), K*N-(K-1))
        # Integrate[Integrate[Exp[-r^2]^2*r^2*r,{r,0.2,0.8}],{y,0,2 Pi}]
        @test fc' * (M * fc) ≈ 0.2851314275977797 # mathematica

        λ(r²) = sin(r²)
        M = F' * (λ.(axes(F,1)) .*F)
        @test size(M) == (K*N-(K-1), K*N-(K-1))
        # Integrate[Integrate[Exp[-r^2]^2*Sin[r^2]*r, {r, 0.2, 0.8}], {y, 0, 2 Pi}]
        @test fc' * (M * fc) ≈ 0.277157468937116 # mathematica


        N = 20; points = [0.; 0.5; 0.7]
        K = length(points)-1

        # Just disk + annulus element
        F = ContinuousZernikeMode(N, points, 0, 1)
        fc = F \ f0.(axes(F,1))
        λ(r²) = r²
        M = F' * (λ.(axes(F,1)) .*F)
        @test size(M) == (K*(N-1), K*(N-1))
        # Integrate[Integrate[Exp[-r^2]^2*r^2*r,{r,0.0,0.7}],{y,0,2 Pi}]
        @test fc' * (M * fc) ≈ 0.2017562408711249 # mathematica

        λ(r²) = sin(r²)
        M = F' * (λ.(axes(F,1)) .*F)
        @test size(M) == (K*(N-1), K*(N-1))
        # Integrate[Integrate[Exp[-r^2]^2*Sin[r^2]*r, {r, 0.0, 0.7}], {y, 0, 2 Pi}]
        @test fc' * (M * fc) ≈ 0.1982900692096337 # mathematica
    end
end