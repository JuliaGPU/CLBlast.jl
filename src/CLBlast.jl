__precompile__(true)

module CLBlast

using Compat
using OpenCL.cl

depsfile = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
if isfile(depsfile)
    include(depsfile)
else
    error("CLBlast not properly installed. Please run Pkg.build(\"CLBlast\") then restart Julia.")
end

include("constants.jl")
include("error.jl")
include("auxiliary_functions.jl")

include("L1/L1.jl")
include("L2/L2.jl")
include("L3/L3.jl")

end # module
