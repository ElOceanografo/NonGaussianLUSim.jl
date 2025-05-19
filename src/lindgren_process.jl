
function preprocess_lu(::AbstractRNG, spatial_process::LindgrenProcess,
        domain, data, init=NearestInit())

    lu_params = ()
    return lu_params
end

function preprocess_z(spatial_process::LindgrenProcess, zfamily, lu_params, ϵ=cbrt(eps()))

end

function nonneg_lusim(rng, spatial_process::LindgrenProcess, preproc)

end