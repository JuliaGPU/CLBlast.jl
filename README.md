# CLBlast

[![Build Status](https://travis-ci.org/JuliaGPU/CLBlast.jl.svg?branch=master)](https://travis-ci.org/JuliaGPU/CLBlast.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/i1saoodeqrepiodl?svg=true)](https://ci.appveyor.com/project/ranocha/CLBlast-jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaGPU/CLBlast.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaGPU/CLBlast.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaGPU/CLBlast.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaGPU/CLBlast.jl?branch=master)

Wrapper of [CLBlast](https://github.com/CNugteren/CLBlast), a "tuned OpenCL BLAS library".
This package has been inspired by [CLBLAS.jl](https://github.com/JuliaGPU/CLBLAS.jl) and
the BLAS module of [Julia](https://github.com/JuliaLang/julia) and is designed similarly.

## Current Status

Most low-level bindings and high-level wrappers of BLAS level 1, 2, and 3 routines are implemented.
