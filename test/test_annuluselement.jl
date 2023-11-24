using Test, RadialPiecewisePolynomials, LinearAlgebra
using ClassicalOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials, AnnuliOrthogonalPolynomials
using Memoization
import RadialPiecewisePolynomials: ModalTrav, Fill, ApplyArray

@testset "annuluselement" begin
    f0(xy) = exp(-first(xy)^2-last(xy)^2)
    f6(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)^6*cos(6*atan(last(xy), first(xy)))
    f1c(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)*cos(atan(last(xy), first(xy)))
    f1s(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)*sin(atan(last(xy), first(xy)))

    @testset "basics" begin
        C = ContinuousZernikeAnnulusElementMode([0.5; 1], 0, 1)
        @test C isa ContinuousZernikeAnnulusElementMode
        @test C.points == [0.5; 1.0]
        @test C.m == 0
        @test C.j == 1

        B = ContinuousZernikeAnnulusElementMode([0.1; 0.3], 3, 0)
        @test B isa ContinuousZernikeAnnulusElementMode
        @test B.points == [0.1; 0.3]
        @test B.m == 3
        @test B.j == 0
    end

    @testset "evaluation" begin
        N = 11 
        ρ = 0.2
        Z = ZernikeAnnulus(ρ, 1, 1)
        Z₀₁ = ZernikeAnnulus(ρ, 0, 1)
        Z₁₀ = ZernikeAnnulus(ρ, 1, 0)
        b = SVector(0.2,0.3)
        
        for (m,j) in [(0,1), (1,0), (1,1), (4,0), (5,1)] 
            C = ContinuousZernikeAnnulusElementMode([ρ; 1], m, j)     
            @test C[b, 3:N] ≈ ModalTrav(Weighted(Z)[b, Block.(1:2N+4)]).matrix[1:N-2, 2m+j]
            @test C[b, 1] ≈ ModalTrav(Weighted(Z₀₁)[b, Block.(1:2N+4)]).matrix[1, 2m+j]
            @test C[b, 2] ≈ ModalTrav(Weighted(Z₁₀)[b, Block.(1:2N+4)]).matrix[1, 2m+j]
        end
    end

    @testset "conversion" begin
        ρ = 0.2
        Z = ZernikeAnnulus(ρ,0,0)
        b = SVector(0.2,0.3)
        N = 11
        
        T = Float64
        t = inv(one(T)-ρ^2)

        for (m,j) in [(0,1), (1,0), (1,1), (4,0), (5,1)]
                C = ContinuousZernikeAnnulusElementMode([ρ; 1], m, j)
                Q₀₀ = SemiclassicalJacobi{T}(t, 0, 0, m)
                Q₀₁ = SemiclassicalJacobi{T}(t, 0, 1, m)
                Q₁₀ = SemiclassicalJacobi{T}(t, 1, 0, m)
                Q₁₁ = SemiclassicalJacobi{T}(t, 1, 1, m)
            
                L₁₁ = ApplyArray(*,Diagonal(Fill((one(T) - ρ^2)^2,∞)),(Weighted(Q₀₀) \ Weighted(Q₁₁)))
                L₀₁ = ApplyArray(*,Diagonal(Fill((one(T) - ρ^2),∞)),(Weighted(Q₀₀) \ Weighted(Q₀₁)))
                L₁₀ = ApplyArray(*,Diagonal(Fill((one(T) - ρ^2),∞)),(Weighted(Q₀₀) \ Weighted(Q₁₀)))

                R̃ =  Hcat(view(L₁₀,:,1), view(L₀₁,:,1), L₁₁)
                @test C[b, 1:N]' ≈ ModalTrav(Z[b, Block.(1:3N)]).matrix[1:N,2*C.m+C.j]' * R̃[1:N, 1:N]
        end
    end

    @testset "mass matrix" begin
        Memoization.empty_all_caches!()
        # Test m = 0
        ρ = 0.2

        C = ContinuousZernikeAnnulusElementMode([ρ; 1], 0, 1)
        fc = C \ f0.(axes(C,1))
        M = C' * C
        # ∫_0^2π ∫_ρ^1 exp(-r^2)^2 r dr dθ.
        N = 100
        @test fc[1:N]' * reshape(view(M, 1:N, 1:N)[:], N, N) * fc[1:N] ≈  π/2 * (exp(-2*0.2^2) - exp(-2))

        # Test m = 1
        C = ContinuousZernikeAnnulusElementMode([ρ; 1], 1, 0)
        fc = C \ f1s.(axes(C,1))
        M = C' * C
        # ∫_0^2π ∫_ρ^1 exp(-r^2)^2 r^2 sin(θ)^2 r dr dθ.
        N = 100
        @test fc[1:N]' * reshape(view(M, 1:N, 1:N)[:], N, N) * fc[1:N] ≈ 0.2320693725039186

        # Test m = 6
        Memoization.empty_all_caches!()
        C = ContinuousZernikeAnnulusElementMode([ρ; 1], 6, 1)
        fc = C \ f6.(axes(C,1))
        M = C' * C
        # ∫_0^2π ∫_ρ^1 exp(-r^2)^2 r^12 cos(6θ)^2 r dr dθ.
        N = 100
        @test fc[1:N]' * reshape(view(M, 1:N, 1:N)[:], N, N) * fc[1:N] ≈ 0.04005947846778158
    end

    @testset "differentiation matrix" begin
        Memoization.empty_all_caches!()
        # Test m = 0
        ρ = 0.2
        # function rhs0(xy)
        #     x,y = first(xy), last(xy)
        #     r2 = x^2 + y^2
        #     exp(-r2)
        # end
        C = ContinuousZernikeAnnulusElementMode([ρ; 1], 0, 1)
        fc = C \ f0.(axes(C,1))
        ∇ = Derivative(axes(C,1)); Δ = (∇*C)' * (∇*C)
        # ∫_0^2π ∫_ρ^1 |∇ exp(-r^2)|^2 r dr dθ.  
        N = 100; 
        @test fc[1:N]' * Δ[1:N,1:N] * fc[1:N] ≈ 1.856554980031349

        # Test m = 1
        # function rhs1(xy)
        #     x,y = first(xy), last(xy)
        #     r2 = x^2 + y^2
        #     θ = atan(y,x)
        #     exp(-r2) * sqrt(r2) * sin(θ)
        # end
        C = ContinuousZernikeAnnulusElementMode([ρ; 1], 1, 0)
        xy = axes(C,1)
        fc = C \ f1s.(axes(C,1))
        ∇ = Derivative(axes(C,1)); Δ = (∇*C)' * (∇*C)
        # ∫_0^2π ∫_ρ^1 |∇ exp(-r^2) r sin(θ)|^2 r dr dθ.
        N = 100; 
        @test fc[1:N]' * Δ[1:N,1:N] * fc[1:N] ≈ 0.816915357578546

        # Test m = 6
        # function rhs6(xy)
        #     x,y = first(xy), last(xy)
        #     r2 = x^2 + y^2
        #     θ = atan(y,x)
        #     exp(-r2) * sqrt(r2)^6 * cos(6θ)
        # end
        Memoization.empty_all_caches!()
        C = ContinuousZernikeAnnulusElementMode([ρ; 1], 6, 1)
        xy = axes(C,1)
        fc = C \ f6.(axes(C,1))
        ∇ = Derivative(axes(C,1)); Δ = (∇*C)' * (∇*C)
        # ∫_0^2π ∫_ρ^1 |∇ exp(-r^2) r sin(θ)|^2 r dr dθ.
        N = 100; 
        @test fc[1:N]' * Δ[1:N,1:N] * fc[1:N] ≈ 2.686674285690333
    end

    @testset "transform scaling" begin
        Memoization.empty_all_caches!()
        α = 0.5; β = 0.6; ρ = α / β
        C = ContinuousZernikeAnnulusElementMode([α; β], 0, 1)
        x = axes(C, 1)

        C̃ = ContinuousZernikeAnnulusElementMode([ρ; 1], 0, 1)

        @test C[SVector(α+eps(), 0), 1:10] ≈ C̃[SVector(ρ+eps(), 0), 1:10]
        @test C[SVector(β-eps(), 0), 1:10] ≈ C̃[SVector(1-eps(), 0), 1:10]

        f = C \ f0.(x)
        for b in ([0.55, 0], [0.51, 0.02])
            @test (C*f)[b] ≈ f0(b)
        end

        C = ContinuousZernikeAnnulusElementMode([α; β], 1, 1)
        C̃ = ContinuousZernikeAnnulusElementMode([ρ; 1], 1, 1)

        @test C[SVector(α+eps(), 0), 1:10] ≈ C̃[SVector(ρ+eps(), 0), 1:10]
        @test C[SVector(β-eps(), 0), 1:10] ≈ C̃[SVector(1-eps(), 0), 1:10]

        f = C \ f1c.(axes(C,1))
        for b in ([0.55, 0], [0.51, 0.02])
            @test (C*f)[b] ≈ f1c(b)
        end
    end

    @testset "mass matrix scaling" begin
        Memoization.empty_all_caches!()
        # Test mass matrix (m = 0)
        α = 0.5; β = 0.6;
        C = ContinuousZernikeAnnulusElementMode([α; β], 0, 1)
        x = axes(C, 1)
        f = C \ f0.(x)
        M = C' * C
        N=20
        # <exp(-r^2), exp(-r^2)>_{L^2} = ∫_0^2π ∫_α^β  exp(-r^2)^2 r dr dθ
        @test f[1:N]' * reshape(view(M, 1:N, 1:N)[:], N, N) * f[1:N] ≈ 0.188147476644037 # mathematica

        # Test mass matrix (m = 1)
        C = ContinuousZernikeAnnulusElementMode([α; β], 1, 1)
        f = C \ f1c.(axes(C,1))
        M = C' * C
        # <exp(-r^2)*r*cos(θ), exp(-r^2)*r*cos(θ)>_{L^2} = ∫_0^2π ∫_α^β  (exp(-r^2)*r*cos(θ))^2 r dr dθ
        @test f[1:N]' * reshape(view(M, 1:N, 1:N)[:], N, N) * f[1:N] ≈ 0.028502927676856027 # mathematica
    end

    @testset "differentiation matrix scaling" begin
        α = 0.2; β = 0.6;
        C = ContinuousZernikeAnnulusElementMode([α; β], 0, 1)
        x = axes(C, 1)
        f = C \ f0.(x)
        Δ = (Derivative(x)*C)' * (Derivative(x)*C)
        N=30
        # <∇ exp(-r^2), ∇exp(-r^2)>_{L^2} = ∫_0^2π ∫_α^β |∇ exp(-r^2)|^2 r dr dθ
        # Integrate[Integrate[D[Exp[-r^2], r]^2 * r , {r, 0.2, 0.6}], {y, 0, 2 Pi}]
        @test f[1:N]' * Δ[1:N,1:N] * f[1:N] ≈  0.5018749991138368  # mathematica

        # Test mass matrix (m = 1)

        C = ContinuousZernikeAnnulusElementMode([α; β], 1, 1)
        f = C \ f1c.(axes(C,1))
        Δ = (Derivative(x)*C)' * (Derivative(x)*C)
        # <∇[exp(-r^2)*r*cos(θ)], ∇[exp(-r^2)*r*cos(θ)]>_{L^2} = ∫_0^2π ∫_α^β  (exp(-r^2)*r*cos(θ))^2 r dr dθ
        # Integrate[Integrate[D[Exp[-r^2]*r*Cos[y], r]^2 * r + D[Exp[-r^2]*r*Cos[y], y]^2/r, {r, 0.2, 0.6}], {y, 0, 2 Pi}]
        @test f[1:N]' * Δ[1:N,1:N] * f[1:N] ≈ 0.4918978196760503 # mathematica
    end

    @testset "assembly" begin
        Memoization.empty_all_caches!()
        # Test weighted mass matrix (m = 0)
        α = 0.5; β = 0.7;
        C = ContinuousZernikeAnnulusElementMode([α; β], 0, 1);
        x = axes(C, 1);
        f = C \ f0.(x);
        λ(r2) = r2;
        # weighted mass matrix
        Z = C' * (λ.(axes(C,1)).*C);
        N=20;
        # bug to do with copy workaround
        ZN = reshape(view(Z,1:N,1:N)[:], N, N)
        # Integrate[Integrate[Exp[-r^2]^2*r^3, {r, 0.5, 0.7}], {y, 0, 2 Pi}]
        @test f[1:N]' * ZN * f[1:N] ≈ 0.1309101767474941 # mathematica

        λ(r2) = sin(r2);
        Z = C' * (λ.(axes(C,1)).*C);
        ZN = reshape(view(Z,1:N,1:N)[:], N, N)
        # Integrate[Integrate[Exp[-r^2]^2*Sin[r^2]*r, {r, 0.5, 0.7}], {y, 0, 2 Pi}]
        @test f[1:N]' * ZN * f[1:N] ≈ 0.1277872408136837 # mathematica

        C = ContinuousZernikeAnnulusElementMode([α; β], 1, 1)
        f = C \ f1c.(axes(C,1))
        λ(r2) = r2;
        Z = C' * (λ.(axes(C,1)).*C);
        ZN = reshape(view(Z,1:N,1:N)[:], N, N)
        # Integrate[Integrate[Exp[-r^2]^2*r^2*Cos[2y]^2*r^3,{r,0.5,0.7}],{y,0,2 Pi}]
        @test f[1:N]' * ZN * f[1:N] ≈ 0.02445414018764588 # mathematica

        λ(r2) = sin(r2);
        Z = C' * (λ.(axes(C,1)).*C);
        ZN = reshape(view(Z,1:N,1:N)[:], N, N)
        # Integrate[Integrate[Exp[-r^2]^2*r^2*Cos[2y]^2*r*Sin[r^2],{r,0.5,0.7}],{y,0,2 Pi}]
        @test f[1:N]' * ZN * f[1:N] ≈ 0.02383305045095835 # mathematica
    end
end