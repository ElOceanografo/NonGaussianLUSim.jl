using NonGaussianLUSim
using Test
using GeoStats
using DataFrames
using Distributions
using Random

Random.seed!(1234)

rng = Random.default_rng()
grid = CartesianGrid(50, 50)
proc = GaussianProcess(GaussianVariogram(range=30.0))
real = rand(rng, proc, grid)
real.field .= real.field .^ 2

ii = sample(eachindex(real.field), 100, replace=false)
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
z_params = preprocess_z(Gamma, lu_params)
preproc = LUNGSPrep(lu_params, z_params)


proc = NonGaussianProcess(sp1, Gamma)
preproc = GeoStatsProcesses.preprocess(rng, proc, LUNGS(), NearestInit(), grid, data)

sim1 = GeoStatsProcesses.randsingle(rng, proc, LUNGS(), grid, data, preproc)


sim2 = rand(proc, grid, method=LUNGS(), data=data)

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
