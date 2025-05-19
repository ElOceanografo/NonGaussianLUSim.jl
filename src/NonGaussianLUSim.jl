module NonGaussianLUSim

# Write your package code here.

using GeoStats
using GeoStats.GeoStatsProcesses: FieldProcess, _zeros, randinit, _pairwise, randsingle
using Distributions
using LinearAlgebra
using StatsBase
using DataFrames
using Random

export CovarianceProcess,
    NonGaussianProcess,
    ContinuousFieldProcess,
    LUNGS,
    LUNGSPrep,
    preprocess_lu,
    dist_params,
    preprocess_z,
    preprocess,
    randsingle,
    compare_z_distributions,
    choose_z_distribution

include("covariance_process.jl")

# wraps CovarianceProcess or LindgrenProcess
ContinuousFieldProcess = Union{CovarianceProcess, LindgrenProcess}
struct NonGaussianProcess{P<:ContinuousFieldProcess,D} <: FieldProcess
    spatial_process::P
    zfamily::D
end

struct LUNGS <: GeoStatsProcesses.FieldSimulationMethod end

# struct LUNGSPrep

# end
LUNGSPrep(lu_params, z_params) = (;lu_params, z_params)

dist_params(d::Type{Gamma}, μ, v) = (v / μ, μ^2 / v)
dist_params(d::Type{InverseGaussian}, μ, v) = (μ, μ^3 / v)
dist_params(d::Type{InverseGamma}, μ, v) = (μ^2 / v + 2, μ^3/v + μ)
dist_params(d::Type{LogNormal}, μ, v) = ( log(μ) - log(v/exp(2log(μ)) + 1) / 2, sqrt(log(v/exp(2log(μ)) + 1)) )

function preprocess_z(::CovarianceProcess, zfamily, lu_params, ϵ=cbrt(eps()))
    μx = copy(lu_params.d₂)
    data = lu_params.z₁
    L = lu_params.L₂₂

    μx[μx .< ϵ] .= ϵ 
    μx = μx .* mean(data) ./ mean(μx)
    μz = L \ μx
    μz[μz .<= ϵ] .= ϵ
    vz = ones(length(μz))
    return [zfamily(p...) for p in dist_params.(zfamily, μz, vz)]
end

function GeoStatsProcesses.preprocess(rng::AbstractRNG, process::NonGaussianProcess, method::LUNGS, init,
        domain, data, ϵ=cbrt(eps()))
    (; spatial_process, zfamily) = process
    lu_params = preprocess_lu(spatial_process, domain, data, init)
    z_params = preprocess_z(spatial_process, zfamily, lu_params, ϵ)
    return LUNGSPrep(lu_params, z_params)
end

  
function GeoStatsProcesses.randsingle(rng::AbstractRNG, process::NonGaussianProcess, ::LUNGS, domain, data, preproc)
    var₁, z₁ = nonneg_lusim(rng, process.spatial_process, preproc)
    cols = (var₁ => z₁,)
    return (; cols...)
end

function GeoStatsProcesses.randsingle(process::NonGaussianProcess, domain, data, preproc)
    rng = Random.default_rng()
    randsingle(rng, process::NonGaussianProcess, LUNGS(), domain, data, preproc)
end

GeoStatsProcesses.defaultsimulation(process::NonGaussianProcess, domain; data) = LUNGS()

function compare_z_distributions(candidate_dists, spatial_process, domain, data;
        n=500, verbose=false)
    icol = only(findall(name -> name != "geometry", names(data)))
    x = data[:, icol]
    h_data = normalize(fit(Histogram, x), mode=:density)
    bin_edges = h_data.edges[1]
    lu_params = preprocess_lu(spatial_process, domain, data)

    fit_list = []
    for Dist in candidate_dists
        if verbose
            println("Comparing with $(Dist)...")
        end
        process = NonGaussianProcess(spatial_process, Dist)
        z_params = preprocess_z(spatial_process, Dist, lu_params)
        preproc = LUNGSPrep(lu_params, z_params)
        fits = map(1:n) do i
            x_sim = randsingle(process, domain, data, preproc)[1]
            h_sim = normalize(fit(Histogram, x_sim, bin_edges), mode=:density)
            kld = evaluate(KLDivergence(), h_data.weights, h_sim.weights)
            (distribution=Dist, kld=kld)
        end
        push!(fit_list, DataFrame(fits))
    end
    dist_fits = vcat(fit_list...)
    dist_fits = dist_fits[isfinite.(dist_fits.kld), :]
    return combine(groupby(dist_fits, :distribution),
        :kld => mean,
        :kld => (x -> std(x) / sqrt(length(x))) => :kld_se
    )
end

function choose_z_distribution(candidate_dists, spatial_process, domain, data;
        n=500, verbose=false)
    dist_fits = compare_z_distributions(candidate_dists, spatial_process, domain, data;
        n, verbose)
    return dist_fits.distribution[argmin(dist_fits.kld_mean)] 
end

end # module