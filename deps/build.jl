using BinDeps
using Compat

@BinDeps.setup
libnames = ["libCLBlast", "libclblast", "clblast"]
libCLBlast = library_dependency("libCLBlast", aliases = libnames)
version = "1.4.1"

if Compat.Sys.iswindows()
    if Sys.ARCH == :x86_64
        uri = URI("https://github.com/CNugteren/CLBlast/releases/download/" *
                  version * "/CLBlast-" * version * "-Windows-x64.zip")
        basedir = @__DIR__
        provides(
            Binaries, uri,
            libCLBlast, unpacked_dir = ".",
            installed_libpath = joinpath(basedir, "libCLBlast", "lib"), os = :Windows
        )
    else
        error("Only 64 bit windows supported with automatic build.")
    end
end

if Compat.Sys.islinux()
    #=if Sys.ARCH == :x86_64
        name, ext = splitext(splitext(basename(baseurl * "Linux-x64.tar.gz"))[1])
        uri = URI(baseurl * "Linux-x64.tar.gz")
        basedir = joinpath(@__DIR__, name)
        provides(
            Binaries, uri,
            libCLBlast, unpacked_dir = basedir,
            installed_libpath = joinpath(basedir, "lib"), os = :Linux
        )
    else
        error("Only 64 bit linux supported with automatic build.")
    end=#
    provides(Sources, URI("https://github.com/CNugteren/CLBlast/archive/" * version * ".tar.gz"),
             libCLBlast, unpacked_dir="CLBlast-" * version)

    builddir = joinpath(@__DIR__, "src", "CLBlast-" * version, "build")
    libpath = joinpath(builddir, "libclblast.so")
    provides(BuildProcess,
        (@build_steps begin
            GetSources(libCLBlast)
            CreateDirectory(builddir)
            FileRule(libpath, @build_steps begin
                ChangeDirectory(builddir)
                `cmake -DSAMPLES=OFF -DTESTS=OFF -DTUNERS=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..`
                `make -j 4`
            end)
        end),
        libCLBlast, installed_libpath=libpath, os=:Linux)
end

if Compat.Sys.isapple()
    using Homebrew
    provides(Homebrew.HB, "homebrew/core/clblast", libCLBlast, os = :Darwin)
end

if Compat.Sys.islinux()
    # BinDeps.jl seems to be broken, cf. https://github.com/JuliaLang/BinDeps.jl/issues/172
    wd = pwd()
    sourcedir = joinpath(@__DIR__, "CLBlast-" * version)
    builddir = joinpath(sourcedir, "build")
    libpath = joinpath(builddir, "libclblast.so")
    if !isdir(sourcedir)
        url = "https://github.com/CNugteren/CLBlast/archive/" * version * ".tar.gz"
        download_finished = false
        try
            run(pipeline(`wget -q -O - $url`, `tar xzf -`))
            global download_finished = true
        catch e
            println(e)
        end
        if download_finished == false
            try
                run(pipeline(`curl -L $url`, `tar xzf -`))
                global download_finished = true
            catch e
                println(e)
            end
        end
    end
    isdir(builddir) || mkdir(builddir)
    cd(builddir)
    run(`cmake -DSAMPLES=OFF -DTESTS=OFF -DTUNERS=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ..`)
    run(`make -j 4`)
    cd(wd)
    open(joinpath(@__DIR__, "deps.jl"), "w") do file
        write(file,
"""
if VERSION >= v"0.7.0-DEV.3382"
    using Libdl
end
# Macro to load a library
macro checked_lib(libname, path)
    if Libdl.dlopen_e(path) == C_NULL
        error("Unable to load \n\n\$libname (\$path)\n\nPlease ",
              "re-run Pkg.build(CLBlast), and restart Julia.")
    end
    quote
        const \$(esc(libname)) = \$path
    end
end

# Load dependencies
@checked_lib libCLBlast "$libpath"
""")
    end
else
    @BinDeps.install Dict(:libCLBlast => :libCLBlast)
end
