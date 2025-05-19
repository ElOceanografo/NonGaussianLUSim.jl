module NonGaussianLUSim

# Write your package code here.

using GeoStats
using GeoStats.GeoStatsProcesses: FieldProcess, _zeros, randinit, _pairwise
using Distributions
using LinearAlgebra
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
    randsingle

struct CovarianceProcess{F,M} <: FieldProcess
    func::F
    mean::M

    function CovarianceProcess{F,M}(func, mean) where {F,M}
        nm = length(mean)
        nf = nvariates(func)
        @assert nm == nf "mean must have $nf components, received $nm"
        new(func, mean)
  end
end
CovarianceProcess(func, mean) = CovarianceProcess{typeof(func),typeof(mean)}(func, mean)
CovarianceProcess(func) = CovarianceProcess(func, _zeros(nvariates(func)))


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


function preprocess_lu(::AbstractRNG, spatial_process::CovarianceProcess,
        domain, data, init=NearestInit())
    # spatial_process parameters
    f = spatial_process.func
    μ = spatial_process.mean

    # sanity checks
    isvalid(f) = isstationary(f) && issymmetric(f) #&& isbanded(f)
    if !isvalid(f)
        throw(ArgumentError("""
            LUNGS requires a geostatistical function that is stationary and symmetric.
            Covariances or composite functions of covariances satisfy these properties.
        """))
    end
    # initialize realization and mask
    real, mask = randinit(spatial_process, domain, data, init)
    # variable names
    vars = keys(real)
    # number of variables
    nvars = length(vars)
    if nvars > 1
        error("Non-Gaussian co-simulation is not implemented.")
    end
    var = only(vars)
    # sanity checks
    @assert length(vars) == nvariates(f) "incompatible number of variables for geostatistical function"
    # preprocess parameters for variable
    # retrieve data and simulation indices
    dinds = findall(mask[var])
    sinds = setdiff(1:nelements(domain), dinds)
    # data for variable
    z₁ = view(real[var], dinds)
    # mean for variable
    μ₁ = μ
    # centroids for data and simulation locations
    ddom = [centroid(domain, i) for i in dinds]
    sdom = [centroid(domain, i) for i in sinds]
    # marginalize function into covariance for variable
    # cov = nvars > 1 ? _marginalize(f, j) : f
    cov = f
    # covariance between simulation locations
    C₂₂ = _pairwise(cov, sdom)
    if isempty(dinds)
        d₂ = zero(eltype(z₁))
        L₂₂ = cholesky(Symmetric(C₂₂)).L
    else
        # covariance beween data locations
        C₁₁ = _pairwise(cov, ddom)
        C₁₂ = _pairwise(cov, ddom, sdom)

        L₁₁ = cholesky(Symmetric(C₁₁)).L
        B₁₂ = L₁₁ \ C₁₂
        A₂₁ = transpose(B₁₂)

        d₂ = A₂₁ * (L₁₁ \ z₁)
        L₂₂ = cholesky(Symmetric(C₂₂ - A₂₁ * B₁₂)).L
    end
    return (; var, z₁, μ₁, d₂, L₂₂, dinds, sinds)
    # (z₁, d₂, L₂₂, μ, dlocs, slocs)
    # (data=pars[1], μx=pars[2], L=pars[3], μ=pars[4], dlocs=pars[5], slocs=pars[6])
end

function preprocess_lu(spatial_process::CovarianceProcess, domain, data, init=NearestInit())
    return preprocess_lu(Random.default_rng(), spatial_process, domain, data, init)
end

dist_params(d::Type{Gamma}, μ, v) = (v / μ, μ^2 / v)
dist_params(d::Type{InverseGaussian}, μ, v) = (μ, μ^3 / v)
dist_params(d::Type{InverseGamma}, μ, v) = (μ^2 / v + 2, μ^3/v + μ)
dist_params(d::Type{LogNormal}, μ, v) = ( log(μ) - log(v/exp(2log(μ)) + 1) / 2, sqrt(log(v/exp(2log(μ)) + 1)) )

function preprocess_z(zfamily, lu_params, ϵ=cbrt(eps()))
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
    lu_params = preprocess_lu(process.spatial_process, domain, data, init)
    z_params = preprocess_z(process.zfamily, lu_params, ϵ)
    return LUNGSPrep(lu_params, z_params)
end

  
function GeoStatsProcesses.randsingle(rng::AbstractRNG, process::NonGaussianProcess, ::LUNGS, domain, data, preproc)

    # simulate first variable
    var₁, z₁ = nonneg_lusim(rng, process.spatial_process, preproc)

#   cols = if length(preproc) > 1
#     # simulate second variable
#     ρ = _rho(process.func)
#     var₂, z₂, _ = _lusim(rng, preproc[2], ρ, w₁)
#     (var₁ => z₁, var₂ => z₂)
#   else
#     (var₁ => z₁,)
#   end
    cols = (var₁ => z₁,)

    return (; cols...)
end

function nonneg_lumult(lu_params, z)
    data = lu_params.z₁
    L = lu_params.L₂₂
    dinds, sinds = lu_params.dinds, lu_params.sinds

    npts = length(dinds) + length(sinds)
    x = zeros(npts)
    x[sinds] = L * z
    x[dinds] = data
    return x
end

# function nonneg_lusim(rng, lu_params, z_params)
#     z = rand.(z_params)
#     return lu_params.var, nonneg_lumult(lu_params, z)
# end

function nonneg_lusim(rng, spatial_process::CovarianceProcess, preproc)
    # nonneg_lusim(rng, process, preproc.lu_params, preproc.z_params)
    z = rand.(preproc.z_params)
    var = preproc.lu_params.var
    return var, nonneg_lumult(preproc.lu_params, z)
end

end