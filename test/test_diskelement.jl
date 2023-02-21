using Test, RadialPiecewisePolynomials, LinearAlgebra
using ClassicalOrthogonalPolynomials, AlgebraicCurveOrthogonalPolynomials
import MultivariateOrthogonalPolynomials: Zernike, ModalTrav

@testset "diskelement" begin

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

end