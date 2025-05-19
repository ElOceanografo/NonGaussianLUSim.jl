using NonGaussianLUSim
using Test
using GeoStats
using DataFrames
using Distributions
using Random

Random.seed!(1234)

rng = Random.default_rng()
grid = CartesianGrid(500, 1)
proc = GaussianProcess(GaussianVariogram(range=30.0))
real = rand(rng, proc, grid)
real.field .= real.field .^ 2 .* (1:500)./2

ii = sample(eachindex(real.field), 50, replace=false)
data = georef(DataFrame(x = real.field[ii]), 
    centroid.(real.geometry[ii]))

vg_emp = EmpiricalVariogram(data, :x)
vg_fit = GeoStatsFunctions.fit(ExponentialVariogram, vg_emp)

sp1 = CovarianceProcess(vg_fit)
sp2 = LindgrenProcess(range(vg_fit), sill(vg_fit))

method = LUNGS()

spde_prep = GeoStatsProcesses.preprocess(rng, sp2, nothing, NearestInit(), grid, data)


lu_params = preprocess_lu(rng, sp1, grid, data, NearestInit())
lu_params = preprocess_lu(sp1, grid, data, NearestInit())
lu_params = preprocess_lu(sp1, grid, data)
z_params = preprocess_z(sp1, Gamma, lu_params)
preproc = LUNGSPrep(lu_params, z_params)


proc = NonGaussianProcess(sp1, Gamma)
preproc = GeoStatsProcesses.preprocess(rng, proc, LUNGS(), NearestInit(), grid, data)

sim1 = GeoStatsProcesses.randsingle(rng, proc, LUNGS(), grid, data, preproc)
sim1 = GeoStatsProcesses.randsingle(proc, grid, data, preproc)

sim2 = rand(proc, grid, method=LUNGS(), data=data)
sim2 = rand(proc, grid, data=data)


compare_z_distributions([Gamma, InverseGaussian], sp1, grid, data, n=10, verbose=true)
choose_z_distribution([Gamma, InverseGaussian], sp1, grid, data, n=10, verbose=true)

rand(proc, grid, 100, data=data)



@testset "NonGaussianLUSim.jl" begin

    sp1 = CovarianceProcess(vg_fit)
    sp2 = LindgrenProcess(range(vg_fit), sill(vg_fit))

    NonGaussianProcess(sp1, Gamma)
    @test_throws MethodError NonGaussianProcess(IndicatorProcess(LinearTransiogram()), Gamma)

    method = LUNGS()

    lu_prep = preprocess_lu(rng, sp1, grid, data, NearestInit())
    lu_prep = preprocess_lu(sp1, grid, data, NearestInit())
    lu_prep = preprocess_lu(sp1, grid, data)

end

# short range for more variability
sp2 = LindgrenProcess(range(vg_fit)/5, sill(vg_fit))

lu_params = preprocess_lu(rng, sp2, grid, data, NearestInit())
lu_params = preprocess_lu(sp2, grid, data, NearestInit())
lu_params = preprocess_lu(sp2, grid, data)
z_params = preprocess_z(sp2, Gamma, lu_params)
preproc = LUNGSPrep(lu_params, z_params)


data_gauss = georef(DataFrame(x = sqrt.(real.field[ii])), 
    centroid.(real.geometry[ii]))
lu_params = preprocess_lu(sp2, grid, data_gauss)
z_params = preprocess_z(sp2, Gamma, lu_params)
proc = NonGaussianProcess(sp2, InverseGamma)
preproc = GeoStatsProcesses.preprocess(rng, proc, LUNGS(), NearestInit(), grid, data)

sim1 = GeoStatsProcesses.randsingle(rng, proc, LUNGS(), grid, data, preproc)
sim1 = GeoStatsProcesses.randsingle(proc, grid, data, preproc)

sim2 = rand(proc, grid, method=LUNGS(), data=data)
sim2 = rand(proc, grid, data=data)


sims = rand(proc, grid, 100, data=data)

xx = [s.x for s in sims]

lines(xx[1])
for i in 2:100
    lines!(xx[i])
end
# (; var, Q, F, σ², i₁, i₂, z̄) = lu_params

# # unconditional realization at vertices
# w = randn(rng, eltype(F), size(F, 1))
# zᵤ = F \ w
# mean(zᵤ)
# std(zᵤ)
# # adjust variance
# s² = Statistics.var(zᵤ, mean=zero(eltype(zᵤ)))
# zᵤ .= √(ustrip(σ²) / s²) .* zᵤ
# mean(zᵤ)
# std(zᵤ)

# # view realization at data locations
# zᵤ₁ = view(zᵤ, i₁)

# # interpolate at simulation locations
# zᵤ₂ = -Q[i₂,i₂] \ (Q[i₂,i₁] * zᵤ₁)

# # merge the above results
# z̄ᵤ = similar(zᵤ)
# z̄ᵤ[i₁] .= zᵤ₁
# z̄ᵤ[i₂] .= zᵤ₂


# # add residual field
# z = z̄ .+ (zᵤ .- z̄ᵤ)
# Ez = (z̄ .- z̄ᵤ)
# z = zᵤ .+ Ez

# xx = [ustrip(coords(p).x) for p in data_gauss.geometry]

# lines(zᵤ)
# lines!(z̄ᵤ)
# scatter!(xx, data_gauss.x)

# lines!(z̄)lines!(Ez)
# lines!(z)

# lines(z̄)
# scatter!(ii, data_gauss.x)


