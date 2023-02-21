module RadialPiecewisePolynomials

using AlgebraicCurveOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ContinuumArrays, DomainSets,
    FastTransforms, ForwardDiff, LinearAlgebra, MultivariateOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials,
    StaticArrays, QuasiArrays, FillArrays, LazyArrays#, ArrayLayouts#, LazyBandedMatrices#, 


import ContinuumArrays: Weight, grid, ℵ₁, ℵ₀, @simplify, ProjectionFactorization, plan_grid_transform, unweighted, weight
import Base: in, axes, getindex, broadcasted, tail, +, -, *, /, \, convert, OneTo, show, summary, ==, oneto, diff
import SemiclassicalOrthogonalPolynomials: divmul, HalfWeighted, Interlace
import MultivariateOrthogonalPolynomials: BlockOneTo, ModalInterlace, BlockRange1, Plan, ModalTrav
import ClassicalOrthogonalPolynomials: checkpoints, ShuffledR2HC, TransformFactorization, ldiv, paddeddata, jacobimatrix, orthogonalityweight, SetindexInterlace, pad
import LinearAlgebra: eigvals, eigen, isapprox, SymTridiagonal, norm, factorize
import AlgebraicCurveOrthogonalPolynomials: factorize, ZernikeAnnulusITransform
import LazyArrays: Vcat

export SVector, Zeros, Ones, Vcat,
        ContinuousZernikeElementMode, ContinuousZernikeAnnulusElementMode, grid, plotvalues, plotannulus

include("diskelement.jl")
include("annuluselement.jl")

end # module
