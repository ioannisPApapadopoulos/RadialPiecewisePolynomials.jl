module RadialPiecewisePolynomials

using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ContinuumArrays, DomainSets,
    FastTransforms, ForwardDiff, LinearAlgebra, MultivariateOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials,
    StaticArrays, QuasiArrays, FillArrays, LazyArrays, Memoization #, ArrayLayouts#, LazyBandedMatrices#, 


import ContinuumArrays: Weight, grid, ℵ₁, ℵ₀, @simplify, ProjectionFactorization, plan_grid_transform, unweighted, weight
import Base: in, axes, getindex, broadcasted, tail, +, -, *, /, \, convert, OneTo, show, summary, ==, oneto, diff
import SemiclassicalOrthogonalPolynomials: divmul, HalfWeighted, Interlace
import MultivariateOrthogonalPolynomials: BlockOneTo, ModalInterlace, BlockRange1, Plan, ModalTrav
import ClassicalOrthogonalPolynomials: checkpoints, ShuffledR2HC, TransformFactorization, ldiv, paddeddata, jacobimatrix, orthogonalityweight, SetindexInterlace, pad
import LinearAlgebra: eigvals, eigen, isapprox, SymTridiagonal, norm, factorize
import AlgebraicCurveOrthogonalPolynomials: factorize, ZernikeAnnulusITransform
import LazyArrays: Vcat
import SpecialFunctions: beta
import HypergeometricFunctions: _₂F₁general2

export SVector, Zeros, Ones, Vcat, Derivative, pad, paddeddata, Hcat,
        ContinuousZernikeElementMode, ContinuousZernikeAnnulusElementMode, grid, plotvalues, plotannulus,
        ContinuousZernikeAnnulusMode,
        FiniteContinuousZernikeAnnulusMode, zero_dirichlet_bcs!, element_plotvalues,
        FiniteContinuousZernikeAnnulus, finite_plotvalues

include("diskelement.jl")
include("annuluselement.jl")
include("finiteannulusmode.jl")
include("finiteannulus.jl")
include("annulus.jl")
end # module
