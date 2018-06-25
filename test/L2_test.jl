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
    end 
end
