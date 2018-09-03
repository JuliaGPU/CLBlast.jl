# CLBlast

[![Build Status](https://gitlab.com/JuliaGPU/CLArrays.jl/badges/master/pipeline.svg)](https://gitlab.com/JuliaGPU/CLArrays.jl/pipelines)
[![Build Status](https://travis-ci.org/JuliaGPU/CLBlast.jl.svg?branch=master)](https://travis-ci.org/JuliaGPU/CLBlast.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/i1saoodeqrepiodl?svg=true)](https://ci.appveyor.com/project/ranocha/CLBlast-jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaGPU/CLBlast.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaGPU/CLBlast.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaGPU/CLBlast.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaGPU/CLBlast.jl?branch=master)

Wrapper of [CLBlast](https://github.com/CNugteren/CLBlast), a "tuned OpenCL BLAS library".
This package has been inspired by [CLBLAS.jl](https://github.com/JuliaGPU/CLBLAS.jl) and
the BLAS module of [Julia](https://github.com/JuliaLang/julia) and is designed similarly.

## Current Status

Most low-level bindings and high-level wrappers of BLAS level 1, 2, and 3 routines are implemented.


## Example

```julia
using CLBlast, OpenCL
@static if VERSION < v"0.7-"
    LA = LinAlg
else
    using Random, LinearAlgebra
    LA = LinearAlgebra
end

device, context, queue = cl.create_compute_context()

# setup data
α = 1.f0
β = 1.f0
A = rand(Float32, 10, 8)
B = rand(Float32, 8, 6)
C = zeros(Float32, 10, 6)

# transfer data
A_cl = cl.CLArray(queue, A)
B_cl = cl.CLArray(queue, B)
C_cl = cl.CLArray(queue, C)

# compute
LA.BLAS.gemm!('N', 'N', α, A, B, β, C)
CLBlast.gemm!('N', 'N', α, A_cl, B_cl, β, C_cl)

# compare results
@assert cl.to_host(C_cl) ≈ C
```


## Installation

Since this package is registered, you can add it using `]` (activate package mode) and
```julia
(v0.7) pkg> add CLBlast
```
on Julia `v0.7` or newer and using
```julia
julia> Pkg.add("CLBlast")
```
on Julia `v0.6`. During the build process, a suitable version of CLBlast will be
downloaded and build. On Linux, you have to install `clang`, since the available
binaries of CLBlast will fail to work with complex numbers from Julia.

