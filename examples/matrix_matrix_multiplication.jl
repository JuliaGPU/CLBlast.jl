using OpenCL, CLBLAS, CLBlast, BenchmarkTools
@static if VERSION < v"0.7-"
    LA = Base.LinAlg
else
    using LinearAlgebra, Random, Printf
    LA = LinearAlgebra
end

CLBLAS.setup()

function run_gemm(m::Integer, n::Integer, k::Integer, T=Float32)
    srand(12345)
    A = rand(T, m, k)
    B = rand(T, k, n)
    C = rand(T, m, n)
    α = rand(T)
    β = rand(T)

    @printf("\nm = %d, n = %d, k = %d, eltype = %s\n", m, n, k, T)

    println("BLAS:")
    C_true = copy(C)
    LinAlg.BLAS.gemm!('N', 'N', α, A, B, β, C_true)
    CC = copy(C)
    display(@benchmark LinAlg.BLAS.gemm!($('N'), $('N'), $α, $A, $B, $β, $CC))
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
        B_cl = cl.CLArray(queue, B)
        C_cl = cl.CLArray(queue, C)
        CLBLAS.gemm!('N', 'N', α, A_cl, B_cl, β, C_cl)
        @assert cl.to_host(C_cl) ≈ C_true
        display(@benchmark CLBLAS.gemm!($('N'), $('N'), $α, $A_cl, $B_cl, $β, $C_cl))
        println()


        println("CLBlast:")
        A_cl = cl.CLArray(queue, A)
        B_cl = cl.CLArray(queue, B)
        C_cl = cl.CLArray(queue, C)
        CLBlast.gemm!('N', 'N', α, A_cl, B_cl, β, C_cl)
        @assert cl.to_host(C_cl) ≈ C_true
        display(@benchmark CLBlast.gemm!($('N'), $('N'), $α, $A_cl, $B_cl, $β, $C_cl))
        println()
    end
end

run_gemm(2^10, 2^10, 2^10, Float32)
