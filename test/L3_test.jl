srand(12345)

@testset "gemm!" begin 
    for elty in elty_L1
        is_linux() && elty == Complex64 && continue

        # A_mul_B!
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

        @test_throws DimensionMismatch CLBlast.gemm!('T', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('C', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('N', 'T', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('N', 'C', α, A_cl, B_cl, β, C_cl, queue=queue)

        @test_throws ArgumentError CLBlast.gemm!('A', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws ArgumentError CLBlast.gemm!('N', 'A', α, A_cl, B_cl, β, C_cl, queue=queue)

        # At_mul_B!, Ac_mul_B!
        A = rand(elty, k_L3, m_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, k_L3, n_L3)
        B_cl = cl.CLArray(queue, B)
        C = rand(elty, m_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = rand(elty)

        CLBlast.gemm!('T', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        LinAlg.BLAS.gemm!('T', 'N', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        CLBlast.gemm!('C', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        LinAlg.BLAS.gemm!('C', 'N', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        @test_throws DimensionMismatch CLBlast.gemm!('N', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('N', 'T', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('N', 'C', α, A_cl, B_cl, β, C_cl, queue=queue)

        # A_mul_Bt!, A_mul_Bc!
        A = rand(elty, m_L3, k_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, n_L3, k_L3)
        B_cl = cl.CLArray(queue, B)
        C = rand(elty, m_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = rand(elty)

        CLBlast.gemm!('N', 'T', α, A_cl, B_cl, β, C_cl, queue=queue)
        LinAlg.BLAS.gemm!('N', 'T', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        CLBlast.gemm!('N', 'C', α, A_cl, B_cl, β, C_cl, queue=queue)
        LinAlg.BLAS.gemm!('N', 'C', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        @test_throws DimensionMismatch CLBlast.gemm!('N', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('T', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('C', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)

        # At_mul_Bt!, Ac_mul_Bt!, At_mul_Bt!, Ac_mul_Bc!
        A = rand(elty, k_L3, m_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, n_L3, k_L3)
        B_cl = cl.CLArray(queue, B)
        C = rand(elty, m_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = rand(elty)

        CLBlast.gemm!('T', 'T', α, A_cl, B_cl, β, C_cl, queue=queue)
        LinAlg.BLAS.gemm!('T', 'T', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        CLBlast.gemm!('C', 'T', α, A_cl, B_cl, β, C_cl, queue=queue)
        LinAlg.BLAS.gemm!('C', 'T', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        CLBlast.gemm!('T', 'C', α, A_cl, B_cl, β, C_cl, queue=queue)
        LinAlg.BLAS.gemm!('T', 'C', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        CLBlast.gemm!('C', 'C', α, A_cl, B_cl, β, C_cl, queue=queue)
        LinAlg.BLAS.gemm!('C', 'C', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        @test_throws DimensionMismatch CLBlast.gemm!('N', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('T', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('C', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('N', 'T', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemm!('N', 'C', α, A_cl, B_cl, β, C_cl, queue=queue)
    end 
end

@testset "symm!" begin 
    for elty in elty_L1
        is_linux() && elty == Complex64 && continue

        # multiply from the left
        A = rand(elty, m_L3, m_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, m_L3, n_L3)
        B_cl = cl.CLArray(queue, B)
        C = rand(elty, m_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = rand(elty)

        for uplo in ['U','L']
            CLBlast.symm!('L', uplo, α, A_cl, B_cl, β, C_cl, queue=queue)
            LinAlg.BLAS.symm!('L', uplo, α, A, B, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B
            @test cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.symm!('L', uplo, α, B_cl, A_cl, β, C_cl, queue=queue)
            @test_throws DimensionMismatch CLBlast.symm!('R', uplo, α, A_cl, B_cl, β, C_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.symm!('A', 'U', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws ArgumentError CLBlast.symm!('U', 'A', α, A_cl, B_cl, β, C_cl, queue=queue)

        # multiply from the right
        A = rand(elty, n_L3, n_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, m_L3, n_L3)
        B_cl = cl.CLArray(queue, B)
        C = rand(elty, m_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = rand(elty)

        for uplo in ['U','L']
            CLBlast.symm!('R', uplo, α, A_cl, B_cl, β, C_cl, queue=queue)
            LinAlg.BLAS.symm!('R', uplo, α, A, B, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B
            @test cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.symm!('L', uplo, α, A_cl, B_cl, β, C_cl, queue=queue)
            @test_throws DimensionMismatch CLBlast.symm!('R', uplo, α, B_cl, A_cl, β, C_cl, queue=queue)
        end
    end
end
