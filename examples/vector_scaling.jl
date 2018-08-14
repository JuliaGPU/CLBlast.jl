using OpenCL, CLBLAS, CLBlast, BenchmarkTools
@static if VERSION < v"0.7-"
    LA = Base.LinAlg
else
    using LinearAlgebra, Random, Printf
    LA = LinearAlgebra
end

CLBLAS.setup()

function run_scal(n::Integer, T=Float32)
    srand(12345)
    x = rand(T, n)
    α = rand(T)

    @printf("\nn = %d, eltype = %s\n", n, T)

    println("BLAS:")
    x_true = copy(x)
    LinAlg.BLAS.scal!(n, α, x_true, 1)
    xx = copy(x)
    display(@benchmark LinAlg.BLAS.scal!($n, $α, $xx, $1))
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
        CLBLAS.scal!(n, α, x_cl, 1)
        @assert cl.to_host(x_cl) ≈ x_true
        display(@benchmark CLBLAS.scal!($n, $α, $x_cl, $1))
        println()

        println("CLBlast:")
        x_cl = cl.CLArray(queue, x)
        CLBlast.scal!(n, α, x_cl, 1)
        # or CLBlast.scal!(α, x_cl)
        @assert cl.to_host(x_cl) ≈ x_true
        display(@benchmark CLBlast.scal!($n, $α, $x_cl, $1))
        println()
    end
end

run_scal(2^10, Float32)
