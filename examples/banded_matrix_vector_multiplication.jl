using OpenCL, CLBLAS, CLBlast, BenchmarkTools
@static if VERSION < v"0.7-"
    LA = Base.LinAlg
else
    using LinearAlgebra, Random, Printf
    LA = LinearAlgebra
end

CLBLAS.setup()

function run_gbmv(m::Integer, n::Integer, kl::Integer, ku::Integer, T=Float32)
    srand(12345)
    A = rand(T, kl+ku+1, n)
    y = rand(T, m)
    x = rand(T, n)
    α = rand(T)
    β = rand(T)

    @printf("\nm = %d, n = %d, kl = %d, ku = %d, eltype = %s\n", m, n, kl, ku, T)

    println("BLAS:")
    y_true = copy(y)
    LinAlg.BLAS.gbmv!('N', m, kl, ku, α, A, x, β, y_true)
    yy = copy(y)
    display(@benchmark LinAlg.BLAS.gbmv!($('N'), $m, $kl, $ku, $α, $A, $x, $β, $yy))
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
        A_cl = cl.CLArray(queue, A)
        y_cl = cl.CLArray(queue, y)
        x_cl = cl.CLArray(queue, x)
        CLBLAS.gbmv!('N', m, kl, ku, α, A_cl, x_cl, β, y_cl)
        @assert cl.to_host(y_cl) ≈ y_true
        display(@benchmark CLBLAS.gbmv!($('N'), $m, $kl, $ku, $α, $A_cl, $x_cl, $β, $y_cl))
        println()

        println("CLBlast:")
        A_cl = cl.CLArray(queue, A)
        y_cl = cl.CLArray(queue, y)
        x_cl = cl.CLArray(queue, x)
        CLBlast.gbmv!('N', m, kl, ku, α, A_cl, x_cl, β, y_cl)
        @assert cl.to_host(y_cl) ≈ y_true
        display(@benchmark CLBlast.gbmv!($('N'), $m, $kl, $ku, $α, $A_cl, $x_cl, $β, $y_cl))
        println()
    end
end

run_gbmv(2^10, 2^10, 5, 5, Float32)
