@static if VERSION < v"0.7-"
    srand(12345)
else
    Random.seed!(12345)
end

@testset "swap!" begin
    for elty in eltypes
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
    for elty in eltypes
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y_cl = cl.CLArray(queue, x)
        α = rand(elty)
        CLBlast.scal!(length(x_cl), α, x_cl, 1, queue=queue)
        Compat.LinearAlgebra.BLAS.scal!(length(x), α, x, 1)
        @test cl.to_host(x_cl, queue=queue) ≈ x

        CLBlast.scal!(α, y_cl, queue=queue)
        @test cl.to_host(y_cl, queue=queue) ≈ x

        for α in (2, 2.f0, 2.0, 2+0im)
            CLBlast.scal!(length(x_cl), α, x_cl, 1, queue=queue)
            x .= α .* x
            @test cl.to_host(x_cl, queue=queue) ≈ x
        end
    end
end

@testset "copy!" begin
    for elty in eltypes
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
    for elty in eltypes
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L1)
        y_cl = cl.CLArray(queue, y)
        α = rand(elty)

        CLBlast.axpy!(length(x_cl), α, x_cl, 1, y_cl, 1, queue=queue)
        y_true = α .* x .+ y
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y_true

        x_cl = cl.CLArray(queue, x)
        y_cl = cl.CLArray(queue, y)
        CLBlast.axpy!(α, x_cl, y_cl, queue=queue)
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y_true

        x_cl = cl.CLArray(queue, x[1:n_L1-1])
        @test_throws DimensionMismatch CLBlast.axpy!(α, x_cl, y_cl, queue=queue)
    end
end

@testset "dot" begin
    for elty in eltypes
        elty <: Real || continue
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L1)
        y_cl = cl.CLArray(queue, y)
        @test Compat.LinearAlgebra.BLAS.dot(length(x), x, 1, y, 1) ≈ CLBlast.dot(length(x_cl), x_cl, 1, y_cl, 1, queue=queue)
    end
end

@testset "dotu" begin
    for elty in eltypes
        elty <: Complex || continue
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L1)
        y_cl = cl.CLArray(queue, y)
        @test Compat.LinearAlgebra.BLAS.dotu(length(x), x, 1, y, 1) ≈ CLBlast.dotu(length(x_cl), x_cl, 1, y_cl, 1, queue=queue)
    end
end

@testset "dotc" begin
    for elty in eltypes
        elty <: Complex || continue
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L1)
        y_cl = cl.CLArray(queue, y)
        @test Compat.LinearAlgebra.BLAS.dotc(length(x), x, 1, y, 1) ≈ CLBlast.dotc(length(x_cl), x_cl, 1, y_cl, 1, queue=queue)
    end
end

@testset "nrm2" begin
    for elty in eltypes
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        @test Compat.LinearAlgebra.BLAS.nrm2(length(x), x, 1) ≈ CLBlast.nrm2(length(x_cl), x_cl, 1, queue=queue)
    end
end

@testset "asum" begin
    for elty in eltypes
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        @test Compat.LinearAlgebra.BLAS.asum(length(x), x, 1) ≈ CLBlast.asum(length(x_cl), x_cl, 1, queue=queue)
    end
end

@testset "sum" begin
    for elty in eltypes
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        sumx = sum(x)
        @test real(sumx)+imag(sumx) ≈ CLBlast.sum(length(x_cl), x_cl, 1, queue=queue)
    end
end

_internalnorm(z) = abs(real(z)) + abs(imag(z))
@testset "iamax" begin
    for elty in eltypes
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        #NOTE: +1 due to zero based indexing in OpenCL vs. 1 based indexing in Julia
        idx_CLBlast = CLBlast.iamax(length(x_cl), x_cl, 1, queue=queue) + 1
        idx_BLAS = Compat.LinearAlgebra.BLAS.iamax(length(x), x, 1)
        if elty <: Real
            @test _internalnorm(x[idx_BLAS]) ≈ _internalnorm(x[idx_CLBlast])
        else
            @test_broken _internalnorm(x[idx_BLAS]) ≈ _internalnorm(x[idx_CLBlast])
        end
    end
end

@testset "iamin" begin
    for elty in eltypes
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        #NOTE: +1 due to zero based indexing in OpenCL vs. 1 based indexing in Julia
        idx_CLBlast = CLBlast.iamin(length(x_cl), x_cl, 1, queue=queue) + 1
        @test_broken minimum(_internalnorm, x) ≈ _internalnorm(x[idx_CLBlast])
    end
end

@testset "had!" begin
    for elty in eltypes
        x = rand(elty, n_L1)
        x_cl = cl.CLArray(queue, x)
        y = rand(elty, n_L1)
        y_cl = cl.CLArray(queue, y)
        z = rand(elty, n_L1)
        z_cl = cl.CLArray(queue, z)
        α = rand(elty)
        β = rand(elty)

        CLBlast.had!(length(x_cl), α, x_cl, 1, y_cl, 1, β, z_cl, 1, queue=queue)
        z .= α .* x .* y .+ β .* z
        @test cl.to_host(x_cl, queue=queue) ≈ x
        @test cl.to_host(y_cl, queue=queue) ≈ y
        @test cl.to_host(z_cl, queue=queue) ≈ z
    end
end
