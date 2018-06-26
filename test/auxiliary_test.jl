for error_code in keys(CLBlast._clblast_status_codes)
    println(DevNull, CLBlast.CLBlastError(error_code))
end

CLBlast.clear_cache()
#CLBlast.fill_cache()
