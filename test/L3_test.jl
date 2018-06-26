srand(12345)

@testset "gemm!" begin 
    for elty in elty_L1
        is_linux() && elty == Complex64 && continue

        A = rand(elty, m_L3, k_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, k_L3, n_L3)
        B_cl = cl.CLArray(queue, B)
        C = rand(elty, m_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = rand(elty)

        CLBlast.gemm!('N', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        LinAlg.BLAS.gemm!('N', 'N', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        @test_throws ArgumentError CLBlast.gemm!('A', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws ArgumentError CLBlast.gemm!('N', 'A', α, A_cl, B_cl, β, C_cl, queue=queue)
    end 
end
