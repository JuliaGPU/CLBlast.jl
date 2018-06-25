using BinDeps
using Compat

@BinDeps.setup
libnames = ["libCLBlast", "libclblast", "clblast"]
libCLBlast = library_dependency("libCLBlast", aliases = libnames)
baseurl = "https://github.com/CNugteren/CLBlast/releases/download/1.4.0/CLBlast-1.4.0-"


if is_windows()
    if Sys.ARCH == :x86_64
        uri = URI(baseurl * "Windows-x64.zip")
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

if is_linux()
    if Sys.ARCH == :x86_64
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
    end
end

if is_apple()
    using Homebrew
    provides(Homebrew.HB, "homebrew/core/clblast", libCLBlast, os = :Darwin)
end

@BinDeps.install Dict("libCLBlast" => "libCLBlast")
