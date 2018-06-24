
@testset "swap!" begin 
    for elty in elty_L1
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L1)
        y_cl = cl.CLArray(queue, y)
        CLBlast.swap!(length(x_cl), x_cl, 1, y_cl, 1, queue=queue)
        @test cl.to_host(y_cl, queue=queue) ≈ x
        @test cl.to_host(x_cl, queue=queue) ≈ y
    end 
end

@testset "scal!" begin 
    for elty in elty_L1
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        α = rand(elty)
        CLBlast.scal!(length(x_cl), α, x_cl, 1, queue=queue)
        LinAlg.BLAS.scal!(length(x), α, x, 1)
        if elty == Complex64
            @test_broken cl.to_host(x_cl, queue=queue) ≈ x
        else
            @test cl.to_host(x_cl, queue=queue) ≈ x
        end

        for α in (2, 2.f0, 2.0, 2+0im)
            CLBlast.scal!(length(x_cl), α, x_cl, 1, queue=queue)
            x .= α .* x
            if elty == Complex64
                @test_broken cl.to_host(x_cl, queue=queue) ≈ x
            else
                @test cl.to_host(x_cl, queue=queue) ≈ x
            end
        end
    end
end

@testset "copy!" begin 
    for elty in elty_L1
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L1)
        y_cl = cl.CLArray(queue, y)
        CLBlast.copy!(length(x_cl), x_cl, 1, y_cl, 1, queue=queue)
        @test cl.to_host(y_cl, queue=queue) ≈ x
        @test cl.to_host(x_cl, queue=queue) ≈ x
    end 
end

@testset "axpy!" begin 
    for elty in elty_L1
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L1)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)
        @test_skip CLBlast.axpy!(length(x_cl), α, x_cl, 1, y_cl, 1, queue=queue)
        y .= α .* x .+ y
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test_broken cl.to_host(y_cl, queue=queue) ≈ y
    end 
end

@testset "dot" begin 
    for elty in elty_L1
        elty <: Real || continue
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L1)
        y_cl = cl.CLArray(queue, y)
        @test LinAlg.BLAS.dot(length(x), x, 1, y, 1) ≈ CLBlast.dot(length(x_cl), x_cl, 1, y_cl, 1, queue=queue)
    end 
end

@testset "dotu" begin 
    for elty in elty_L1
        elty <: Complex || continue
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L1)
        y_cl = cl.CLArray(queue, y)
        @test LinAlg.BLAS.dotu(length(x), x, 1, y, 1) ≈ CLBlast.dotu(length(x_cl), x_cl, 1, y_cl, 1, queue=queue)
    end 
end



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
