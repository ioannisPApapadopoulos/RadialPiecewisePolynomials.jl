module RadialPiecewisePolynomials

using AnnuliOrthogonalPolynomials, BandedMatrices, BlockArrays, BlockBandedMatrices, ClassicalOrthogonalPolynomials, ContinuumArrays, DomainSets,
    FastTransforms, LinearAlgebra, MultivariateOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials,
    StaticArrays, QuasiArrays, FillArrays, LazyArrays, Memoization#, ArrayLayouts#, LazyBandedMatrices#, 


import BlockArrays: BlockSlice, block, blockindex, blockvec
import ContinuumArrays: Weight, grid, ℵ₁, ℵ₀, @simplify, ProjectionFactorization, plan_grid_transform, unweighted, weight
import Base: in, axes, getindex, broadcasted, tail, +, -, *, /, \, convert, OneTo, show, summary, ==, oneto, diff
import SemiclassicalOrthogonalPolynomials: HalfWeighted
import MultivariateOrthogonalPolynomials: BlockOneTo, ModalInterlace, BlockRange1, Plan, ModalTrav, ZernikeITransform
import ClassicalOrthogonalPolynomials: checkpoints, ShuffledR2HC, TransformFactorization, ldiv, paddeddata, jacobimatrix, orthogonalityweight, SetindexInterlace, pad, blockedrange, Clenshaw, recurrencecoefficients, _p0, colsupport
import LinearAlgebra: eigvals, eigen, isapprox, SymTridiagonal, norm, factorize
import AnnuliOrthogonalPolynomials: factorize, ZernikeAnnulusITransform
import LazyArrays: Vcat
import SpecialFunctions: beta
import HypergeometricFunctions: _₂F₁general2
import BlockBandedMatrices: _BandedBlockBandedMatrix, AbstractBandedBlockBandedMatrix, subblockbandwidths, blockbandwidths, AbstractBandedBlockBandedLayout, layout_replace_in_print_matrix

export SVector, Zeros, Ones, Vcat, Derivative, pad, paddeddata, Hcat, RadialCoordinate,
        ContinuousZernikeElementMode, ContinuousZernikeAnnulusElementMode, grid, plotvalues, plot_helper,
        ContinuousZernikeAnnulusMode,
        zero_dirichlet_bcs!, element_plotvalues,
        finite_plotvalues, inf_error, plot,
        FiniteContinuousZernikeMode, FiniteContinuousZernike, inf_error,
        FiniteZernikeBasis, ZernikeBasisMode, FiniteZernikeBasisMode,
        ArrowheadMatrix,
        list_2_modaltrav, modaltrav_2_list, adi_2_modaltrav, adi_2_list,
        get_rs, get_θs

get_rs(x) = x.r
get_θs(x) = x.θ

include("arrowhead.jl")
include("diskelement.jl")
include("annuluselement.jl")
include("finitecontinuousmode.jl")
include("finitecontinuous.jl")
include("finitezernike.jl")

end # module
