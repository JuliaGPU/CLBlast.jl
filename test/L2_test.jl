@static if VERSION < v"0.7-"
    srand(12345)
else
    Random.seed!(12345)
end

@testset "gemv!" begin
    for elty in eltypes
        A = rand(elty, m_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        β = rand(elty)

        @test_throws DimensionMismatch CLBlast.gemv!('T', α, A_cl, x_cl, β, y_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemv!('C', α, A_cl, x_cl, β, y_cl, queue=queue)
        CLBlast.gemv!('N', α, A_cl, x_cl, β, y_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gemv!('N', α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws DimensionMismatch CLBlast.gemv!('N', α, A_cl, y_cl, β, x_cl, queue=queue)
        CLBlast.gemv!('T', α, A_cl, y_cl, β, x_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gemv!('T', α, A, y, β, x)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws DimensionMismatch CLBlast.gemv!('N', α, A_cl, y_cl, β, x_cl, queue=queue)
        CLBlast.gemv!('C', α, A_cl, y_cl, β, x_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gemv!('C', α, A, y, β, x)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws ArgumentError CLBlast.gemv!('A', α, A_cl, y_cl, β, x_cl, queue=queue)
    end
end

@testset "gbmv!" begin
    for elty in eltypes
        A = rand(elty, kl+ku+1, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        β = rand(elty)

        CLBlast.gbmv!('N', m_L2, kl, ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gbmv!('N', m_L2, kl, ku, α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.gbmv!('T', m_L2, kl, ku, α, A_cl, y_cl, β, x_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gbmv!('T', m_L2, kl, ku, α, A, y, β, x)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.gbmv!('C', m_L2, kl, ku, α, A_cl, y_cl, β, x_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.gbmv!('C', m_L2, kl, ku, α, A, y, β, x)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws ArgumentError CLBlast.gbmv!('A', m_L2, kl, ku, α, A_cl, y_cl, β, x_cl, queue=queue)
    end
end

@testset "hemv!" begin
    for elty in eltypes
        elty <: Complex || continue

        A = rand(elty, n_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        β = rand(elty)

        CLBlast.hemv!('U', α, A_cl, x_cl, β, y_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.hemv!('U', α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.hemv!('L', α, A_cl, x_cl, β, y_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.hemv!('L', α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws ArgumentError CLBlast.hemv!('A', α, A_cl, y_cl, β, x_cl, queue=queue)

        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        @test_throws DimensionMismatch CLBlast.hemv!('U', α, A_cl, x_cl, β, y_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.hemv!('U', α, A_cl, y_cl, β, x_cl, queue=queue)
    end
end

@testset "hbmv!" begin
    for elty in eltypes
        elty <: Complex || continue

        A = rand(elty, ku+1, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        β = rand(elty)

        CLBlast.hbmv!('U', ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.hbmv!('U', ku, α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.hbmv!('L', ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.hbmv!('L', ku, α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws ArgumentError CLBlast.hbmv!('A', ku, α, A_cl, x_cl, β, y_cl, queue=queue)

        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        @test_throws DimensionMismatch CLBlast.hbmv!('U', ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.hbmv!('U', ku, α, A_cl, y_cl, β, x_cl, queue=queue)
    end
end

@testset "symv!" begin
    for elty in eltypes
        elty <: Real || continue

        A = rand(elty, n_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        β = rand(elty)

        CLBlast.symv!('U', α, A_cl, x_cl, β, y_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.symv!('U', α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.symv!('L', α, A_cl, x_cl, β, y_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.symv!('L', α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws ArgumentError CLBlast.symv!('A', α, A_cl, y_cl, β, x_cl, queue=queue)

        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        @test_throws DimensionMismatch CLBlast.symv!('U', α, A_cl, x_cl, β, y_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.symv!('U', α, A_cl, y_cl, β, x_cl, queue=queue)
    end
end

@testset "sbmv!" begin
    for elty in eltypes
        elty <: Real || continue

        A = rand(elty, ku+1, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        β = rand(elty)

        CLBlast.sbmv!('U', ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.sbmv!('U', ku, α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.sbmv!('L', ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.sbmv!('L', ku, α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws ArgumentError CLBlast.sbmv!('A', ku, α, A_cl, x_cl, β, y_cl, queue=queue)

        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        @test_throws DimensionMismatch CLBlast.sbmv!('U', ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.sbmv!('U', ku, α, A_cl, y_cl, β, x_cl, queue=queue)
    end
end

@testset "trmv!" begin
    for elty in eltypes
        A = rand(elty, n_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)

        for uplo in ['U','L'], trans in ['N','T','C'], diag in ['N','U']
            CLBlast.trmv!(uplo, trans, diag, A_cl, x_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.trmv!(uplo, trans, diag, A, x)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(x_cl, queue=queue) ≈ x
        end

        @test_throws ArgumentError CLBlast.trmv!('A', 'N', 'N', A_cl, x_cl, queue=queue)
        @test_throws ArgumentError CLBlast.trmv!('U', 'A', 'N', A_cl, x_cl, queue=queue)
        @test_throws ArgumentError CLBlast.trmv!('U', 'N', 'A', A_cl, x_cl, queue=queue)

        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        @test_throws DimensionMismatch CLBlast.trmv!('U', 'N', 'N', A_cl, y_cl, queue=queue)

        A = rand(elty, m_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        @test_throws DimensionMismatch CLBlast.trmv!('U', 'N', 'N', A_cl, x_cl, queue=queue)
    end
end

@testset "trsv!" begin
    for elty in eltypes
        A = rand(elty, n_L2, n_L2)
        for i in 1:n_L2
            A[i,i] = i
        end
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)

        for uplo in ['U','L'], trans in ['N','T','C'], diag in ['N','U']
            CLBlast.trsv!(uplo, trans, diag, A_cl, x_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.trsv!(uplo, trans, diag, A, x)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(x_cl, queue=queue) ≈ x
        end

        @test_throws ArgumentError CLBlast.trsv!('A', 'N', 'N', A_cl, x_cl, queue=queue)
        @test_throws ArgumentError CLBlast.trsv!('U', 'A', 'N', A_cl, x_cl, queue=queue)
        @test_throws ArgumentError CLBlast.trsv!('U', 'N', 'A', A_cl, x_cl, queue=queue)

        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        @test_throws DimensionMismatch CLBlast.trsv!('U', 'N', 'N', A_cl, y_cl, queue=queue)

        A = rand(elty, m_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        @test_throws DimensionMismatch CLBlast.trsv!('U', 'N', 'N', A_cl, x_cl, queue=queue)
    end
end

@testset "ger!" begin
    for elty in eltypes
        A = rand(elty, m_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, m_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)

        CLBlast.ger!(α, x_cl, y_cl, A_cl, queue=queue)
        Compat.LinearAlgebra.BLAS.ger!(α, x, y, A)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws DimensionMismatch CLBlast.ger!(α, x_cl, x_cl, A_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.ger!(α, y_cl, y_cl, A_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.ger!(α, y_cl, x_cl, A_cl, queue=queue)
    end
end

@testset "her!" begin
    for elty in eltypes
        elty <: Complex || continue

        A = rand(elty, n_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        α = real(rand(elty))

        for uplo in ['U', 'L']
            CLBlast.her!(uplo, α, x_cl, A_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.her!(uplo, α, x, A)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(x_cl, queue=queue) ≈ x
        end

        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        @test_throws DimensionMismatch CLBlast.her!('U', α, y_cl, A_cl, queue=queue)
    end
end

@testset "syr!" begin
    for elty in eltypes
        elty <: Real || continue

        A = rand(elty, n_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        α = real(rand(elty))

        for uplo in ['U', 'L']
            CLBlast.syr!(uplo, α, x_cl, A_cl, queue=queue)
            Compat.LinearAlgebra.BLAS.syr!(uplo, α, x, A)
            @test cl.to_host(A_cl, queue=queue) ≈ A
            @test cl.to_host(x_cl, queue=queue) ≈ x
        end

        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        @test_throws DimensionMismatch CLBlast.syr!('U', α, y_cl, A_cl, queue=queue)
    end
end
