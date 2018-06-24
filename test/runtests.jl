module TestCLBlast

@static if VERSION < v"0.7.0-DEV.2005"
    using Base.Test
else
    using Test
end

using CLBlast, OpenCL

const n_L1 = 64
const elty_L1 = (Float32, Float64, Complex64, Complex128)

device, ctx, queue = cl.create_compute_context()

@time @testset "BLAS Level 1" begin include("L1_test.jl") end

end # module
