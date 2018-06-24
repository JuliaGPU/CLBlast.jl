__precompile__(true)

module CLBlast

using OpenCL: cl

depsfile = joinpath(dirname(@__FILE__), "..", "deps", "deps.jl")
if isfile(depsfile)
    include(depsfile)
else
    error("CLBlast not properly installed. Please run Pkg.build(\"CLBlast\") then restart Julia.")
end

include("L1/L1.jl")


end # module
