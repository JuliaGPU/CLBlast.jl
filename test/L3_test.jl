@static if VERSION < v"0.7-"
    srand(12345)
else
    Random.seed!(12345)
end

@testset "gemm!" begin
    for elty in eltypes
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
        Compat.LinearAlgebra.BLAS.gemm!('N', 'N', α, A, B, β, C)
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
        Compat.LinearAlgebra.BLAS.gemm!('T', 'N', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        CLBlast.gemm!('C', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gemm!('C', 'N', α, A, B, β, C)
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
        Compat.LinearAlgebra.BLAS.gemm!('N', 'T', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        CLBlast.gemm!('N', 'C', α, A_cl, B_cl, β, C_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gemm!('N', 'C', α, A, B, β, C)
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
        Compat.LinearAlgebra.BLAS.gemm!('T', 'T', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        CLBlast.gemm!('C', 'T', α, A_cl, B_cl, β, C_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gemm!('C', 'T', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        CLBlast.gemm!('T', 'C', α, A_cl, B_cl, β, C_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gemm!('T', 'C', α, A, B, β, C)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(B_cl, queue=queue) ≈ B
        @test cl.to_host(C_cl, queue=queue) ≈ C

        CLBlast.gemm!('C', 'C', α, A_cl, B_cl, β, C_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gemm!('C', 'C', α, A, B, β, C)
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
    for elty in eltypes
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
            Compat.LinearAlgebra.BLAS.symm!('L', uplo, α, A, B, β, C)
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
            Compat.LinearAlgebra.BLAS.symm!('R', uplo, α, A, B, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B
            @test cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.symm!('L', uplo, α, A_cl, B_cl, β, C_cl, queue=queue)
            @test_throws DimensionMismatch CLBlast.symm!('R', uplo, α, B_cl, A_cl, β, C_cl, queue=queue)
        end
    end
end

@testset "hemm!" begin
    for elty in eltypes
        elty <: Complex || continue

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
            CLBlast.hemm!('L', uplo, α, A_cl, B_cl, β, C_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.hemm!('L', uplo, α, A, B, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B
            @test cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.hemm!('L', uplo, α, B_cl, A_cl, β, C_cl, queue=queue)
            @test_throws DimensionMismatch CLBlast.hemm!('R', uplo, α, A_cl, B_cl, β, C_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.hemm!('A', 'U', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws ArgumentError CLBlast.hemm!('U', 'A', α, A_cl, B_cl, β, C_cl, queue=queue)

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
            CLBlast.hemm!('R', uplo, α, A_cl, B_cl, β, C_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.hemm!('R', uplo, α, A, B, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B
            @test cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.hemm!('L', uplo, α, A_cl, B_cl, β, C_cl, queue=queue)
            @test_throws DimensionMismatch CLBlast.hemm!('R', uplo, α, B_cl, A_cl, β, C_cl, queue=queue)
        end
    end
end

@testset "syrk!" begin
    for elty in eltypes
        # A*A'
        A = rand(elty, n_L3, k_L3)
        A_cl = cl.CLArray(queue, A)
        C = rand(elty, n_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = rand(elty)

        for uplo in ['U','L']
            CLBlast.syrk!(uplo, 'N', α, A_cl, β, C_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.syrk!(uplo, 'N', α, A, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.syrk!(uplo, 'T', α, A_cl, β, C_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.syrk!('A', 'N', α, A_cl, β, C_cl, queue=queue)
        @test_throws ArgumentError CLBlast.syrk!('U', 'A', α, A_cl, β, C_cl, queue=queue)

        # A'*A
        A = rand(elty, k_L3, n_L3)
        A_cl = cl.CLArray(queue, A)
        C = rand(elty, n_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = rand(elty)

        for uplo in ['U','L']
            CLBlast.syrk!(uplo, 'T', α, A_cl, β, C_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.syrk!(uplo, 'T', α, A, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.syrk!(uplo, 'N', α, A_cl, β, C_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.syrk!('U', 'C', α, A_cl, β, C_cl, queue=queue)
    end
end

@testset "herk!" begin
    for elty in eltypes
        elty <: Complex || continue

        # A*A'
        A = rand(elty, n_L3, k_L3)
        A_cl = cl.CLArray(queue, A)
        C = rand(elty, n_L3, n_L3)
        for i in 1:n_L3
            C[i,i] = real(C[i,i])
        end
        C_cl = cl.CLArray(queue, C)
        α = real(rand(elty))
        β = real(rand(elty))

        for uplo in ['U','L']
            CLBlast.herk!(uplo, 'N', α, A_cl, β, C_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.herk!(uplo, 'N', α, A, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test_skip cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.herk!(uplo, 'C', α, A_cl, β, C_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.herk!('A', 'N', α, A_cl, β, C_cl, queue=queue)
        @test_throws ArgumentError CLBlast.herk!('U', 'A', α, A_cl, β, C_cl, queue=queue)

        # A'*A
        A = rand(elty, k_L3, n_L3)
        A_cl = cl.CLArray(queue, A)
        C = rand(elty, n_L3, n_L3)
        for i in 1:n_L3
            C[i,i] = real(C[i,i])
        end
        C_cl = cl.CLArray(queue, C)
        α = real(rand(elty))
        β = real(rand(elty))

        for uplo in ['U','L']
            CLBlast.herk!(uplo, 'C', α, A_cl, β, C_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.herk!(uplo, 'C', α, A, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test_skip cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.herk!(uplo, 'N', α, A_cl, β, C_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.herk!('U', 'T', α, A_cl, β, C_cl, queue=queue)
    end
end

@testset "syr2k!" begin
    for elty in eltypes
        # A*B'
        A = rand(elty, n_L3, k_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, n_L3, k_L3)
        B_cl = cl.CLArray(queue, B)
        C = rand(elty, n_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = rand(elty)

        for uplo in ['U','L']
            CLBlast.syr2k!(uplo, 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.syr2k!(uplo, 'N', α, A, B, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B
            @test cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.syr2k!(uplo, 'T', α, A_cl, B_cl, β, C_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.syr2k!('A', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws ArgumentError CLBlast.syr2k!('U', 'A', α, A_cl, B_cl, β, C_cl, queue=queue)

        # A'*B
        A = rand(elty, k_L3, n_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, k_L3, n_L3)
        B_cl = cl.CLArray(queue, B)
        C = rand(elty, n_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = rand(elty)

        for uplo in ['U','L']
            CLBlast.syr2k!(uplo, 'T', α, A_cl, B_cl, β, C_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.syr2k!(uplo, 'T', α, A, B, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B
            @test cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.syr2k!(uplo, 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        end
    end
end

@testset "her2k!" begin
    for elty in eltypes
        elty <: Complex || continue

        # A*B'
        A = rand(elty, n_L3, k_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, n_L3, k_L3)
        B_cl = cl.CLArray(queue, B)
        C = rand(elty, n_L3, n_L3)
        C_cl = cl.CLArray(queue, C)
        for i in 1:n_L3
            C[i,i] = real(C[i,i])
        end
        α = rand(elty)
        β = real(rand(elty))

        for uplo in ['U','L']
            CLBlast.her2k!(uplo, 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.her2k!(uplo, 'N', α, A, B, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B
            @test_skip cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.her2k!(uplo, 'C', α, A_cl, B_cl, β, C_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.her2k!('A', 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        @test_throws ArgumentError CLBlast.her2k!('U', 'A', α, A_cl, B_cl, β, C_cl, queue=queue)

        # A'*B
        A = rand(elty, k_L3, n_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, k_L3, n_L3)
        B_cl = cl.CLArray(queue, B)
        C = rand(elty, n_L3, n_L3)
        for i in 1:n_L3
            C[i,i] = real(C[i,i])
        end
        C_cl = cl.CLArray(queue, C)
        α = rand(elty)
        β = real(rand(elty))

        for uplo in ['U','L']
            CLBlast.her2k!(uplo, 'C', α, A_cl, B_cl, β, C_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.her2k!(uplo, 'C', α, A, B, β, C)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B
            @test_skip cl.to_host(C_cl, queue=queue) ≈ C

            @test_throws DimensionMismatch CLBlast.her2k!(uplo, 'N', α, A_cl, B_cl, β, C_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.her2k!('U', 'T', α, A_cl, B_cl, β, C_cl, queue=queue)
    end
end

@testset "trmm!" begin
    for elty in eltypes
        # multiply from the left
        A = rand(elty, m_L3, m_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, m_L3, n_L3)
        B_cl = cl.CLArray(queue, B)
        α = rand(elty)

        for uplo in ['U','L'], transA in ['N','T','C'], diag in ['N','U']
            CLBlast.trmm!('L', uplo, transA, diag, α, A_cl, B_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.trmm!('L', uplo, transA, diag, α, A, B)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B

            @test_throws DimensionMismatch CLBlast.trmm!('L', uplo, transA, diag, α, B_cl, A_cl, queue=queue)
            @test_throws DimensionMismatch CLBlast.trmm!('R', uplo, transA, diag, α, A_cl, B_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.trmm!('A', 'U', 'N', 'N', α, A_cl, B_cl, queue=queue)
        @test_throws ArgumentError CLBlast.trmm!('U', 'A', 'N', 'N', α, A_cl, B_cl, queue=queue)

        # multiply from the right
        A = rand(elty, n_L3, n_L3)
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, m_L3, n_L3)
        B_cl = cl.CLArray(queue, B)
        α = rand(elty)
        β = rand(elty)

        for uplo in ['U','L'], transA in ['N','T','C'], diag in ['N','U']
            CLBlast.trmm!('R', uplo, transA, diag, α, A_cl, B_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.trmm!('R', uplo, transA, diag, α, A, B)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B

            @test_throws DimensionMismatch CLBlast.trmm!('L', uplo, transA, diag, α, A_cl, B_cl, queue=queue)
            @test_throws DimensionMismatch CLBlast.trmm!('R', uplo, transA, diag, α, B_cl, A_cl, queue=queue)
        end
    end
end

@testset "trsm!" begin
    for elty in eltypes
        # On Travis, there is some strange error
        # https://travis-ci.org/JuliaGPU/CLBlast.jl/jobs/414395954#L347
        @static if VERSION < v"0.7-"
            is_apple() && break
        else
            Sys.isapple() && break
        end

        # multiply from the left
        A = rand(elty, m_L3, m_L3)
        for i in 1:m_L3
            A[i,i] = i
        end
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, m_L3, n_L3)
        B_cl = cl.CLArray(queue, B)
        α = rand(elty)

        for uplo in ['U','L'], transA in ['N','T','C'], diag in ['N','U']
            CLBlast.trsm!('L', uplo, transA, diag, α, A_cl, B_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.trsm!('L', uplo, transA, diag, α, A, B)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B

            @test_throws DimensionMismatch CLBlast.trsm!('L', uplo, transA, diag, α, B_cl, A_cl, queue=queue)
            @test_throws DimensionMismatch CLBlast.trsm!('R', uplo, transA, diag, α, A_cl, B_cl, queue=queue)
        end

        @test_throws ArgumentError CLBlast.trsm!('A', 'U', 'N', 'N', α, A_cl, B_cl, queue=queue)
        @test_throws ArgumentError CLBlast.trsm!('U', 'A', 'N', 'N', α, A_cl, B_cl, queue=queue)

        # multiply from the right
        A = rand(elty, n_L3, n_L3)
        for i in 1:n_L3
            A[i,i] = i
        end
        A_cl = cl.CLArray(queue, A)
        B = rand(elty, m_L3, n_L3)
        B_cl = cl.CLArray(queue, B)
        α = rand(elty)
        β = rand(elty)

        for uplo in ['U','L'], transA in ['N','T','C'], diag in ['N','U']
            CLBlast.trsm!('R', uplo, transA, diag, α, A_cl, B_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.trsm!('R', uplo, transA, diag, α, A, B)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(B_cl, queue=queue) ≈ B

            @test_throws DimensionMismatch CLBlast.trsm!('L', uplo, transA, diag, α, A_cl, B_cl, queue=queue)
            @test_throws DimensionMismatch CLBlast.trsm!('R', uplo, transA, diag, α, B_cl, A_cl, queue=queue)
        end
    end
end
