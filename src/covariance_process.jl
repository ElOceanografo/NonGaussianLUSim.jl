
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

function nonneg_lusim(rng, spatial_process::CovarianceProcess, preproc)
    # nonneg_lusim(rng, process, preproc.lu_params, preproc.z_params)
    z = rand.(preproc.z_params)
    var = preproc.lu_params.var
    return var, nonneg_lumult(preproc.lu_params, z)
end