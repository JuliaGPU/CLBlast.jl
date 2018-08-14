using OpenCL, CLBLAS, CLBlast, BenchmarkTools
@static if VERSION < v"0.7-"
    LA = Base.LinAlg
else
    using LinearAlgebra, Random, Printf
    LA = LinearAlgebra
end

CLBLAS.setup()

function run_axpy(n::Integer, T=Float32)
    srand(12345)
    y = rand(T, n)
    x = rand(T, n)
    α = rand(T)

    @printf("\nn = %d, eltype = %s\n", n, T)

    println("BLAS:")
    y_true = copy(y)
    LinAlg.BLAS.axpy!(α, x, y_true)
    yy = copy(y)
    display(@benchmark LinAlg.BLAS.axpy!($α, $x, $yy))
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
        y_cl = cl.CLArray(queue, y)
        x_cl = cl.CLArray(queue, x)
        CLBLAS.axpy!(α, x_cl, y_cl)
        @assert cl.to_host(y_cl) ≈ y_true
        display(@benchmark CLBLAS.axpy!($α, $x_cl, $y_cl))
        println()

        println("CLBlast:")
        y_cl = cl.CLArray(queue, y)
        x_cl = cl.CLArray(queue, x)
        CLBlast.axpy!(α, x_cl, y_cl)
        @assert cl.to_host(y_cl) ≈ y_true
        display(@benchmark CLBlast.axpy!($α, $x_cl, $y_cl))
        println()
    end
end

run_axpy(2^10, Float32)
