using Test, RadialPiecewisePolynomials, LinearAlgebra
using ClassicalOrthogonalPolynomials, AlgebraicCurveOrthogonalPolynomials
import MultivariateOrthogonalPolynomials: Zernike, ModalTrav

@testset "diskelement" begin

    f0(xy) = exp(-first(xy)^2-last(xy)^2)
    f6(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)^6*cos(6*atan(last(xy), first(xy)))
    f1c(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)*cos(atan(last(xy), first(xy)))
    f1s(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)*sin(atan(last(xy), first(xy)))

    @testset "basics" begin
        C = ContinuousZernikeElementMode(0, 1)
        @test C isa ContinuousZernikeElementMode
        @test C.points == [0.0; 1.0]
        @test C.m == 0
        @test C.j == 1

        B = ContinuousZernikeElementMode([0.0; 0.3], 3, 0)
        @test B isa ContinuousZernikeElementMode
        @test B.points == [0.0; 0.3]
        @test B.m == 3
        @test B.j == 0
    end

    @testset "evaluation" begin
        N = 11 
        Z = Zernike(1)
        b = SVector(0.1,0.2)
        
        for (m,j) in [(0,1), (1,0), (1,1), (4,0), (5,1)] 
            C = ContinuousZernikeElementMode(m, j)     
            @test C[b, 2:N] ≈ ModalTrav(Weighted(Z)[b, Block.(1:2N+4)]).matrix[1:N-1, 2m+j]
            @test C[b, 1] ≈ ModalTrav(Z[b, Block.(1:2N+4)]).matrix[1, 2m+j]
        end
    end

    @testset "conversion" begin
        Z = Zernike(0,1)
        b = SVector(0.1,0.2)
        R = (Z \ Weighted(Z))
        N = 11

        for (m,j) in [(0,1), (1,0), (1,1), (4,0), (5,1)] 
            C = ContinuousZernikeElementMode(m, j)
            R̃ =  [[1.0; Zeros(∞)] R.ops[C.m+1]/2]
            @test C[b, 1:N]' ≈ ModalTrav(Z[b, Block.(1:3N)]).matrix[1:N,2*C.m+C.j]' * R̃[1:N, 1:N]
        end
    end

    @testset "transform scaling" begin
        ρ = 0.6
        C = ContinuousZernikeElementMode([0; ρ], 0, 1)
        C̃ = ContinuousZernikeElementMode([0; 1], 0, 1)

        @test C[SVector(0+eps(), 0), 1:10] ≈ C̃[SVector(0+eps(), 0), 1:10]
        @test C[SVector(ρ-eps(), 0), 1:10] ≈ C̃[SVector(1-eps(), 0), 1:10]

        f = C \ f0.(x)
        for b in ([0.55, 0], [0.51, 0.02])
            @test (C*f)[b] ≈ f0(b)
        end

        C = ContinuousZernikeElementMode([0; ρ], 1, 1)
        C̃ = ContinuousZernikeElementMode([0; 1], 1, 1)
        @test C[SVector(ρ-eps(), 0), 1:10] ≈ C̃[SVector(1-eps(), 0), 1:10]

        f = C \ f1c.(axes(C,1))
        for b in ([0.55, 0], [0.51, 0.02])
            @test (C*f)[b] ≈ f1c(b)
        end
    end

    @testset "mass matrix" begin
        # Test m = 0
        C = ContinuousZernikeElementMode([0; 1], 0, 1)
        fc = C \ f0.(axes(C,1))
        M = C' * C
        # ∫_0^2π ∫_0^1 exp(-r^2)^2 r dr dθ.
        N = 100
        @test fc[1:N]' * M[1:N,1:N] * fc[1:N] ≈  π/2 * (1.0 - exp(-2))

        # Test m = 1
        C = ContinuousZernikeElementMode([0; 1], 1, 0)
        fc = C \ f1s.(axes(C,1))
        M = C' * C
        # NumberForm[NIntegrate[Sin[2*x]^2, {x, 0, 2*Pi}]*NIntegrate[Exp[-r^2]^2*r^3, {r, 0, 1}], 16]
        N = 100
        @test fc[1:N]' * M[1:N,1:N] * fc[1:N] ≈ 0.233260957353361
        # Test m = 6
        C = ContinuousZernikeElementMode([0; 1], 6, 1)
        fc = C \ f6.(axes(C,1))
        M = C' * C 
        #NumberForm[NIntegrate[Sin[6*x]^2, {x, 0, 2*Pi}]*NIntegrate[Exp[-r^2]^2*r^13, {r, 0, 1}], 16]
        N = 100
        @test fc[1:N]' * M[1:N,1:N] * fc[1:N] ≈ 0.04005947850206709
    end

    @testset "mass matrix scaling" begin
        # Test mass matrix (m = 0)
        ρ = 0.6; C = ContinuousZernikeElementMode([0; ρ], 0, 1)
        f = C \ f0.(axes(C, 1))
        M = C' * C
        # <exp(-r^2), exp(-r^2)>_{L^2} = ∫_0^2π ∫_0^ρ  exp(-r^2)^2 r dr dθ
        # NumberForm[2*Pi*NIntegrate[Exp[-r^2]^2*r, {r, 0, 0.6}], 16]
        N = 20; @test f[1:N]' * M[1:N,1:N] * f[1:N] ≈ 0.806207671073844

        # Test mass matrix (m = 1)
        C = ContinuousZernikeElementMode([0; ρ], 1, 1)
        f = C \ f1c.(axes(C,1))
        M = C' * C
        # <exp(-r^2)*r*cos(θ), exp(-r^2)*r*cos(θ)>_{L^2} = ∫_0^2π ∫_α^β  (exp(-r^2)*r*cos(θ))^2 r dr dθ
        # NumberForm[NIntegrate[Cos[x]^2, {x, 0, 2*Pi}]*NIntegrate[Exp[-r^2]^2*r^3, {r, 0, 0.6}], 16]
        @test f[1:N]' * M[1:N,1:N] * f[1:N] ≈ 0.06392595973867163 # mathematica
    end

    @testset "differentiation matrix" begin
        C = ContinuousZernikeElementMode([0; 1], 0, 1)
        fc = C \ f0.(axes(C,1))
        ∇ = Derivative(axes(C,1)); Δ = (∇*C)' * (∇*C)
        # ∫_0^2π ∫_ρ^1 |∇ exp(-r^2)|^2 r dr dθ.
        # NumberForm[2*Pi*NIntegrate[4*r^2*Exp[-r^2]^2*r, {r, 0, 1}], 16]
        N = 100; @test fc[1:N]' * Δ[1:N,1:N] * fc[1:N] ≈ 1.866087658826886

        # Test m = 1
        C = ContinuousZernikeElementMode([0; 1], 1, 0)
        fc = C \ f1s.(axes(C,1))
        ∇ = Derivative(axes(C,1)); Δ = (∇*C)' * (∇*C)
        # ∫_0^2π ∫_ρ^1 |∇ exp(-r^2) r sin(θ)|^2 r dr dθ.
        # NumberForm[NIntegrate[Integrate[D[Exp[-r^2]*r*Sin[x], r]^2*r + D[Exp[-r^2]*r*Sin[x], x]^2/r, {r, 0, 1}], {x, 0, 2*Pi}], 16]
        N = 100; @test fc[1:N]' * Δ[1:N,1:N] * fc[1:N] ≈ 0.933043829413443

        # Test m = 6
        C = ContinuousZernikeElementMode([0; 1], 6, 1)
        fc = C \ f6.(axes(C,1))
        ∇ = Derivative(axes(C,1)); Δ = (∇*C)' * (∇*C)
        # ∫_0^2π ∫_0^1 |∇ exp(-r^2) r^6 cos(6θ)|^2 r dr dθ.
        # NumberForm[NIntegrate[Integrate[D[Exp[-r^2]*r^6 Cos[6 x], r]^2*r + D[Exp[-r^2]*r^6*Cos[6 x], x]^2/r, {r, 0, 1}], {x, 0, 2*Pi}], 16]
        N = 100; @test fc[1:N]' * Δ[1:N,1:N] * fc[1:N] ≈ 2.686674356967118
    end

    @testset "differentiation matrix scaling" begin
        ρ = 0.6; C = ContinuousZernikeElementMode([0; ρ], 0, 1)
        x = axes(C,1)
        f = C \ f0.(x)
        Δ = (Derivative(x)*C)' * (Derivative(x)*C)
        # <∇ exp(-r^2), ∇exp(-r^2)>_{L^2} = ∫_0^2π ∫_α^β |∇ exp(-r^2)|^2 r dr dθ
        # NumberForm[Integrate[Integrate[D[Exp[-r^2], r]^2*r, {r, 0, 0.6}], {y, 0, 2 Pi}], 16]
        N=30; @test f[1:N]' * Δ[1:N,1:N] * f[1:N] ≈  0.5114076779093716  # mathematica


        C = ContinuousZernikeElementMode([0; ρ], 1, 1)
        f = C \ f1c.(axes(C,1))
        Δ = (Derivative(x)*C)' * (Derivative(x)*C)
        # <∇[exp(-r^2)*r*cos(θ)], ∇[exp(-r^2)*r*cos(θ)]>_{L^2} = ∫_0^2π ∫_α^β  (exp(-r^2)*r*cos(θ))^2 r dr dθ
        # NumberForm[Integrate[Integrate[D[Exp[-r^2]*r*Cos[y], r]^2*r + D[Exp[-r^2]*r*Cos[y], y]^2/r, {r, 0, 0.6}], {y, 0, 2 Pi}], 16]
        @test f[1:N]' * Δ[1:N,1:N] * f[1:N] ≈ 0.608026291510947 # mathematica
    end
end