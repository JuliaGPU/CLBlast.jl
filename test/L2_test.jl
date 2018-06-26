srand(12345)

@testset "gemv!" begin 
    for elty in elty_L1
        A = rand(elty, m_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        β = rand(elty)

        is_linux() && elty == Complex64 && continue

        @test_throws DimensionMismatch CLBlast.gemv!('T', α, A_cl, x_cl, β, y_cl, queue=queue)
        @test_throws DimensionMismatch CLBlast.gemv!('C', α, A_cl, x_cl, β, y_cl, queue=queue)
        CLBlast.gemv!('N', α, A_cl, x_cl, β, y_cl, queue=queue)
        LinAlg.BLAS.gemv!('N', α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws DimensionMismatch CLBlast.gemv!('N', α, A_cl, y_cl, β, x_cl, queue=queue)
        CLBlast.gemv!('T', α, A_cl, y_cl, β, x_cl, queue=queue)
        LinAlg.BLAS.gemv!('T', α, A, y, β, x)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws DimensionMismatch CLBlast.gemv!('N', α, A_cl, y_cl, β, x_cl, queue=queue)
        CLBlast.gemv!('C', α, A_cl, y_cl, β, x_cl, queue=queue)
        LinAlg.BLAS.gemv!('C', α, A, y, β, x)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws ArgumentError CLBlast.gemv!('A', α, A_cl, y_cl, β, x_cl, queue=queue)
    end 
end

@testset "gbmv!" begin 
    for elty in elty_L1
        A = rand(elty, kl+ku+1, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, m_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        β = rand(elty)

        is_linux() && elty == Complex64 && continue

        CLBlast.gbmv!('N', m_L2, kl, ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        LinAlg.BLAS.gbmv!('N', m_L2, kl, ku, α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.gbmv!('T', m_L2, kl, ku, α, A_cl, y_cl, β, x_cl, queue=queue)
        LinAlg.BLAS.gbmv!('T', m_L2, kl, ku, α, A, y, β, x)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.gbmv!('C', m_L2, kl, ku, α, A_cl, y_cl, β, x_cl, queue=queue)
        LinAlg.BLAS.gbmv!('C', m_L2, kl, ku, α, A, y, β, x)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        @test_throws ArgumentError CLBlast.gbmv!('A', m_L2, kl, ku, α, A_cl, y_cl, β, x_cl, queue=queue)
    end 
end

@testset "hemv!" begin 
    for elty in elty_L1
        elty <: Complex || continue

        A = rand(elty, n_L2, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        β = rand(elty)

        is_linux() && elty == Complex64 && continue

        CLBlast.hemv!('U', α, A_cl, x_cl, β, y_cl, queue=queue)
        LinAlg.BLAS.hemv!('U', α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.hemv!('L', α, A_cl, x_cl, β, y_cl, queue=queue)
        LinAlg.BLAS.hemv!('L', α, A, x, β, y)
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
    for elty in elty_L1
        elty <: Complex || continue

        A = rand(elty, ku+1, n_L2)
        A_cl = cl.CLArray(queue, A)
        x = rand(elty, n_L2)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L2)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        β = rand(elty)

        is_linux() && elty == Complex64 && continue

        CLBlast.hbmv!('U', ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        LinAlg.BLAS.hbmv!('U', ku, α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.hbmv!('L', ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        LinAlg.BLAS.hbmv!('L', ku, α, A, x, β, y)
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
    for elty in elty_L1
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
        LinAlg.BLAS.symv!('U', α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.symv!('L', α, A_cl, x_cl, β, y_cl, queue=queue)
        LinAlg.BLAS.symv!('L', α, A, x, β, y)
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
    for elty in elty_L1
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
        LinAlg.BLAS.sbmv!('U', ku, α, A, x, β, y)
        @test cl.to_host(A_cl, queue=queue) ≈ A
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y

        CLBlast.sbmv!('L', ku, α, A_cl, x_cl, β, y_cl, queue=queue)
        LinAlg.BLAS.sbmv!('L', ku, α, A, x, β, y)
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
