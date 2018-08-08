"""
    clear_cache()

CLBlast stores binaries of compiled kernels into a cache in case the same kernel
is used later on for thesame device. This cache can be cleared to free up system
memory or it can be useful in case of debugging.
"""
@compat function clear_cache()
    err = ccall((:CLBlastClearCache, libCLBlast), cl.CL_int, ())
    if err != cl.CL_SUCCESS
        println(stderr, "Calling function `clear_cache` failed!")
        throw(cl.CLError(err))
    end
    return err
end

@compat function fill_cache(device::cl.Device)
    err = ccall((:CLBlastFillCache, libCLBlast), cl.CL_int, (cl.CL_device_id,), pointer(device))
    if err != cl.CL_SUCCESS
        println(stderr, "Calling function `fill_cache($device)` failed!")
        throw(cl.CLError(err))
    end
    return err
end
