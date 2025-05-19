
function preprocess_lu(rng::AbstractRNG, spatial_process::LindgrenProcess,
        domain, data, init=NearestInit())
    lu_params = preprocess(rng::AbstractRNG, spatial_process::LindgrenProcess, nothing,
        init, domain, data)
    return lu_params
end

function preprocess_lu(spatial_process::LindgrenProcess, domain, data, init=NearestInit())
    return preprocess_lu(Random.default_rng(), spatial_process, domain, data, init)
end


function preprocess_z(::LindgrenProcess, zfamily, lu_params, ϵ=cbrt(eps()))
    μx = copy(lu_params.z̄)
    # data = lu_params.z₁
    Q = lu_params.Q

    # μx[μx .< ϵ] .= ϵ 
    # μx = μx .* mean(data) ./ mean(μx)
    μz = Q * μx
    μz[μz .<= ϵ] .= ϵ
    vz = ones(length(μz))
    return [zfamily(p...) for p in dist_params.(zfamily, μz, vz)]
end

function nonneg_lusim(rng, spatial_process::LindgrenProcess, preproc)
    F = preproc.lu_params.F
    z = rand.(preproc.z_params)
    x = F \ z
    var = preproc.lu_params.var
    return var, x
end