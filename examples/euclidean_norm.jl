using OpenCL, CLBLAS, CLBlast, BenchmarkTools
@static if VERSION < v"0.7-"
    LA = Base.LinAlg
else
    using LinearAlgebra, Random, Printf
    LA = LinearAlgebra
end

CLBLAS.setup()

function run_nrm2(n::Integer, T=Float32::Union{Type{Float32},Type{Float64}})
    srand(12345)
    x = rand(T, n)

    @printf("\nn = %d, eltype = %s\n", n, T)

    println("BLAS:")
    res_true = LA.BLAS.nrm2(n, x, 1)
    display(@benchmark LA.BLAS.nrm2($n, $x, $1))
    println()

    for device in cl.devices()
        println("-"^70)
        @printf("Platform name   : %s\n", device[:platform][:name])
        @printf("Platform version: %s\n", device[:platform][:version])
        @printf("Device name     : %s\n", device[:name])
        @printf("Device type     : %s\n", device[:device_type])
        println()

        ctx = cl.Context(device)
        queue = cl.CmdQueue(ctx)

        println("CLBLAS:")
        x_cl = cl.CLArray(queue, x)
        res_cl = CLBLAS.nrm2(n, x_cl, 1)
        @assert res_cl ≈ res_true
        display(@benchmark CLBLAS.nrm2($n, $x_cl, $1))
        println()

        println("CLBlast:")
        x_cl = cl.CLArray(queue, x)
        res_cl = CLBlast.nrm2(n, x_cl, 1)
        @assert res_cl ≈ res_true
        display(@benchmark CLBlast.nrm2($n, $x_cl, $1))
        println()
    end
end

run_nrm2(2^16, Float32)
