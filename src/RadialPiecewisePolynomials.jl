module RadialPiecewisePolynomials

using AnnuliOrthogonalPolynomials, BandedMatrices, BlockArrays, BlockBandedMatrices, BlockDiagonals, ClassicalOrthogonalPolynomials, ContinuumArrays, DomainSets,
    FastTransforms, LinearAlgebra, MultivariateOrthogonalPolynomials, SemiclassicalOrthogonalPolynomials,
    StaticArrays, QuasiArrays, FillArrays, LazyArrays, Memoization, SparseArrays, MatrixFactorizations#, ArrayLayouts#, LazyBandedMatrices#,


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
import PiecewiseOrthogonalPolynomials: BBBArrowheadMatrix

export SVector, Zeros, Ones, Vcat, Derivative, pad, paddeddata, Hcat, RadialCoordinate, Block,
        ContinuousZernikeElementMode, ContinuousZernikeAnnulusElementMode, grid, plotvalues, plot_helper,
        ContinuousZernikeAnnulusMode,
        zero_dirichlet_bcs!, element_plotvalues,
        finite_plotvalues, inf_error, plot,
        ContinuousZernikeMode, ContinuousZernike, inf_error,
        ZernikeBasis, ZernikeBasisMode, ZernikeBasisModeElement,
        BBBArrowheadMatrix,
        get_rs, get_θs, getNs,
        mass_matrix, assembly_matrix, stiffness_matrix, gram_matrix, piecewise_constant_assembly_matrix,
        sparse

get_rs(x) = x.r
get_θs(x) = x.θ

function getNs(N::Int)
    ms = ((0:2N) .÷ 2)[2:end-1]
    Ms = ((N + 1 .- ms) .÷ 2); Ms[Ms .<= 2] .= 3
    return Ms
end

include("diskelement.jl")
include("annuluselement.jl")
include("continuouszernikemodestruct.jl")
include("continuouszernikemode.jl")
include("continuouszernike.jl")
include("zernikebasis.jl")


end # module
