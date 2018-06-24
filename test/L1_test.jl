
@testset "asum" begin 
    for elty in elty_L1
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        @test LinAlg.BLAS.asum(length(x), x, 1) ≈ CLBlast.asum(length(x_cl), x_cl, 1, queue=queue)
    end 
end

@testset "sum" begin 
    for elty in elty_L1
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        sumx = sum(x)
        @test real(sumx)+imag(sumx) ≈ CLBlast.sum(length(x_cl), x_cl, 1, queue=queue)
    end 
end
