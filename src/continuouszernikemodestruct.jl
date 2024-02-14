# The continuous Zernike basis on a multiple elements but a single Fourier mode
# Kept in its own file since Revise.jl keeps failing to revise when modifying anything
# in the same file as this struct...?
struct ContinuousZernikeMode{T} <: Basis{T}
    N::Int
    points::AbstractVector{T}
    m::Int
    j::Int
    Cs::Tuple{Vararg{Basis}}
    normalize_constants::AbstractVector{<:AbstractVector{<:T}}
    same_ρs::Bool
    b::Int # Should remove once adaptive expansion has been figured out.
end