module TestCLBlast

using Compat
using Compat.Test
using Compat.Random

using CLBlast, OpenCL

const n_L1 = 64
const m_L2 = 60
const n_L2 = 50
const kl = 2
const ku = 3
const m_L3 = 6
const n_L3 = 5
const k_L3 = 4
@compat const eltypes = (Float32, Float64, ComplexF32, ComplexF64)

device, ctx, queue = cl.create_compute_context()

@time @testset "BLAS Level 1" begin include("L1_test.jl") end
@time @testset "BLAS Level 2" begin include("L2_test.jl") end
@time @testset "BLAS Level 3" begin include("L3_test.jl") end
@time @testset "Auxiliary Tests" begin include("auxiliary_test.jl") end

end # module
