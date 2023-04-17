using Test, RadialPiecewisePolynomials

@testset "finiteannulusmode" begin
    @testset "basics" begin
        N=10; F = FiniteContinuousZernikeMode(N, [0.5; 0.7; 1], 0, 1)
        @test F isa FiniteContinuousZernikeMode
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

            @test C1[[0.3, 0], 1] ≈ C2[[0.3+eps(), 0], 1] / γs[1]
            @test C2[[0.5-eps(), 0], 2] ≈ C3[[0.5+eps(), 0], 1] / γs[2]
        end
    end

end