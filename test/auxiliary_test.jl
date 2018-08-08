@compat for error_code in keys(CLBlast._clblast_status_codes)
    println(devnull, CLBlast.CLBlastError(error_code))
end

CLBlast.clear_cache()
CLBlast.fill_cache(device)
