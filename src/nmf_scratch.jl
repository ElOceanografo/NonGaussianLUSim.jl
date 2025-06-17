using GeoStats
using LinearAlgebra
using SparseArrays
using DataFrames
using Plots


n = 100
s = [rand(2) for _ in 1:n]
s = [Point(si...) for si in s]
x = DataFrame(x = randn(n))

data = georef(x, s)

vge = EmpiricalVariogram(data, :x)
vgt = ExponentialVariogram(range=0.5, sill=1.0, nugget=0.1)

C = GeoStatsProcesses._pairwise(vgt, s)
minimum(C)
L = cholesky(C).L
minimum(L)
L1 = copy(L)
L1[L1 .< 0] .= 0#sqrt(eps())
isposdef(L1)
C1 = Symmetric(L1 * L1')
isposdef(C1)

maximum(C - C1)
plot(heatmap(C), heatmap(collect(C1)))
# res = NMF.solve!(NMF.ProjectedALS{eltype(C)}(maxiter=50), C, W, H)
# res = NMF.solve!(NMF.CoordinateDescent{Float64}(maxiter=50), C, W, H)
L2 = cholesky(C1).L

x = rand(n)
b = C * x
x0 = L \ b
x1 = L1 \ b
x2 = L2 \ b
plot([x1, x2, x3])

xc0 = C \ x
xc1 = (L1 * L1') \ x
plot([xc0, xc1])

plot(
    [
        sum(L, dims=1)',
        sum(L1, dims=1)',
        sum(L2, dims=1)'
    ]
)


stats = @timed nnmf(C, n, alg=:multmse, init=:nndsvd)

# not using :alspgrad, too slow w/ no real benefit
results = []
for alg in (:multmse, :multdiv, :projals, :cd, :greedycd)
    for init in (:random, :nndsvd, :nndsvda, :nndsvdar, :spa)
        println(alg, ", ", init)
        stats = @timed nnmf(C, n, alg=alg, init=init)
        res = stats.value
        asym = sqrt(mean(abs2, res.W - res.H'))
        rmse = sqrt(mean(abs2, C - (res.W * res.H)))
        tup = (alg = alg, init=init, time = stats.time,
            converged = res.converged, obj=res.objvalue, asym = asym, rmse=rmse) 
        push!(results, tup) 
    end
end
results = DataFrame(results)
sort(results, [:asym, :rmse])


# W, H = NMF.nndsvd(C, n, variant=:ar)
# res = (;W, H)
res = nnmf(C, n, alg=:multmse, init=:nndsvdar, maxiter=500)

C - (res.W * res.H)
res.W - res.H'
plot(heatmap(res.W), heatmap(res.H'))
plot(heatmap(C), heatmap(res.W * res.H))
plot(heatmap(C), heatmap(res.W * res.W'))
heatmap(C - (res.W * res.W'))
heatmap(C - (res.W * res.H))



W, H = NMF.nndsvd(C, n, variant=:ar)
# W, H = res.W, res.H
plot(L * x)
plot!(W * x)
plot!(W' * x)
plot!(H * x)
plot!(H' * x)

# # d1 = C - (W * H)
# # d2 = C - (W * W')

# W - H'
# (W - H')




function E(s, n)
    sparse([s+1], [s], [1], n, n)
end

function neville!(L, U, s, t)
    n = size(L, 1)
    N =  U[s+1, t] / U[s, t] * E(s, n)
    U .= (I - N) * U
    L .= L * (I + N)
    U[s+1, t] = 0
    return L, U
end

function nncholesky(C)
    @assert size(C,1) == size(C, 2)
    n = size(C, 1)
    L = collect(1.0*I(n))
    U = copy(C)
    # no checks, assumes all of C is nonzero
    for t in 1:n
        for s in (n-1):-1:t
            neville!(L, U, s, t)
            println("#"^50)
            display(L)
            display(U)
        end
    end
    return (; L, U)
end

n = 7
C = rand(n, n); C = C'C+I
isposdef(C)
L, U = nncholesky(C)



n = 5
C = rand(n, n); C = C'C+I
# C = rationalize.([0.0 1 2 1; 0 2 4 2; 0 1 2 3; 0 3 6 11])
L = collect(1.0*I(n))
U = copy(C)

neville!(L, U, n-1, 1)
L
U

neville!(L, U, n-2, 1)
L
U

neville!(L, U, n-3, 1)
L
U

neville!(L, U, n-4, 1)
L
U

neville!(L, U, n-1, 2)
L
U

neville!(L, U, n-2, 2)
L
U

neville!(L, U, n-3, 2)
L
U

neville!(L, U, n-1, 3)
L
U


neville!(L, U, n-2, 3)
L
U

neville!(L, U, n-1, 4)
L
U










t = 2
s = n-1
N = U[s+1, t] / U[s, t] * E(s, n)
U = (I - N) * U
L = L * (I + N)

# L = collect(1.0*I(n));
# U = copy(C);
# neville!(L, U, s, t);
# U
# L

s = n-2
N = U[s+1, t] / U[s, t] * E(s, n)
U = (I - N) * U
L = L * (I + N)

s = n-3
N = U[s+1, t] / U[s, t] * E(s, n)
U = (I -N) * U
L = L * (I + N)

s = n-4
N = U[s+1, t] / U[s, t] * E(s, n)
U = (I -N) * U
L = L * (I + N)


t = 2
s = n-1
N = U[s+1, t] / U[s, t] * E(s, n)
U = (I -N) * U
L = L * (I + N)

s = n-2
N = U[s+1, t] / U[s, t] * E(s, n)
U = (I -N) * U
L = L * (I + N)

t = 3
s = n-1
N = U[s+1, t] / U[s, t] * E(s, n)
U = (I -N) * U
L = L * (I + N)

C
L * U

F = cholesky(C)
F.L