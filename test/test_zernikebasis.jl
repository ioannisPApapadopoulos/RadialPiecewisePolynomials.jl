
using Test, RadialPiecewisePolynomials, LinearAlgebra
using Memoization
import RadialPiecewisePolynomials: ModalTrav

f0(xy) = exp(-first(xy)^2-last(xy)^2)
f6(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)^6*cos(6*atan(last(xy), first(xy)))
f1c(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)*cos(atan(last(xy), first(xy)))
f1s(xy) = exp(-first(xy)^2-last(xy)^2) * sqrt(first(xy)^2+last(xy)^2)*sin(atan(last(xy), first(xy)))

c1 = -10; c2 = 0; c3=0.6
function u0_(x, y)
    ρ = 0.2
    exp(c1*(x^2 + (y-c3)^2)) * (1-(x^2+y^2)) * ((x^2+y^2)-ρ^2)
end
function u0(xy)
    x,y = first(xy), last(xy)
    u0_(x,y)
end

@testset "zernikeelement" begin
    @testset "basics" begin
        Z = ZernikeBasisMode([0.5; 1], 0, 0, 0, 1)
        @test Z isa ZernikeBasisMode
        @test Z.points == [0.5; 1.0]
        @test Z.m == 0
        @test Z.j == 1
        @test Z.a == 0
        @test Z.b == 0

        B = ZernikeBasisMode([0.1; 0.3], 0, 0, 3, 0)
        @test B isa ZernikeBasisMode
        @test B.points == [0.1; 0.3]
        @test B.m == 3
        @test B.j == 0
        @test B.a == 0
        @test B.b == 0
    end

    @testset "L²-inner product" begin

        # Annuli elements
        points = [0.5;0.6]
        C = ContinuousZernikeAnnulusElementMode(points, 0, 1)
        fc = C \ f0.(axes(C,1))

        Zs = ZernikeBasis(100, points, 0, 0)
        Z = ZernikeBasisMode(points,0,0,0,1)
        fz = Zs \ f0.(axes(Zs,1))
        fz = fz[1]
        @test fc[1:50]' * (C' * Z)[1:50,1:50] * fz ≈ 0.188147476644037 

    end
end

@testset "zernikebasismode" begin
    @testset "basics" begin
        N, points, a, b, m, j = 5, [0.1;0.3;1], 0, 0, 0, 1
        Z = ZernikeBasisMode(N, points, a, b, m, j)
        @test Z.N == N
        @test Z.points == points
        @test Z.a == a
        @test Z.b == b
        @test Z.m == m
        @test Z.j == j
    end

    @testset "L²-inner product" begin
        # annuli elements
        N, points, a, b, m, j = 50, [0.5;0.55;0.6], 0, 0, 0, 1
        Z = ZernikeBasisMode(N, points, a, b, m, j)
        F = ContinuousZernikeMode(N, points, m, j)

        Zs = ZernikeBasis(2N, points, 0, 0)
        fz = Zs \ f0.(axes(Zs,1))
        fz = fz[1]

        fc = F \ f0.(axes(F,1))
        @test fc' * (F' * Z) * fz ≈ 0.188147476644037

        N, points, a, b, m, j = 50, [0.2; 0.5; 0.8; 1.0], 0, 0, 1, 0
        Z = ZernikeBasisMode(N, points, a, b, m, j)
        F = ContinuousZernikeMode(N, points, m, j)

        Zs = ZernikeBasis(2N, points, 0, 0)
        fz = Zs \ f1s.(axes(Zs,1))
        fz = fz[2]

        fc = F \ f1s.(axes(F,1))
        @test fc' * (F' * Z) * fz ≈ 0.2320693725039186

        # disk & annuli elements
        N, points, a, b, m, j = 50, [0.0; 0.5; 0.8; 1.0], 0, 0, 0, 1
        Z = ZernikeBasisMode(N, points, a, b, m, j)
        F = ContinuousZernikeMode(N, points, m, j)

        Zs = ZernikeBasis(2N, points, 0, 0)
        fz = Zs \ f0.(axes(Zs,1))
        fz = fz[1]

        fc = F \ f0.(axes(F,1))
        @test fc' * (F' * Z) * fz ≈ π/2 * (1.0 - exp(-2))

        N, points, a, b, m, j = 50, [0.0; 0.5; 0.8; 1.0], 0, 0, 1, 0
        Z = ZernikeBasisMode(N, points, a, b, m, j)
        F = ContinuousZernikeMode(N, points, m, j)

        Zs = ZernikeBasis(2N, points, 0, 0)
        fz = Zs \ f1s.(axes(Zs,1))
        fz = fz[2]

        fc = F \ f1s.(axes(F,1))
        @test fc' * (F' * Z) * fz ≈ 0.233260957353361
    end
end

@testset "zernikebasis" begin
    @testset "basics" begin
        N, points, a, b = 5, [0.1;0.3;1], 0, 0
        Z = ZernikeBasis(N, points, a, b)
        @test Z.N == N
        @test Z.points == points
        @test Z.a == a
        @test Z.b == b
    end

    @testset "L²-inner product" begin
        N, points, a, b = 50, [0.2; 0.5; 0.8; 1.0], 0, 0
        Z = ZernikeBasis(N, points, a, b)
        F = ContinuousZernike(N, points)

        fz = Z \ u0.(axes(Z,1))
        fc = F \ u0.(axes(F,1))

        (θs, rs, vals) = _plotvalues(Z, fz)
        vals_, err = inf_error(Z, θs, rs, vals, u0)
        @test err < 1e-15

        @test sum(transpose.(fc) .* ((F' * Z) .* fz)) ≈ 0.0055779595855720305
    end
end