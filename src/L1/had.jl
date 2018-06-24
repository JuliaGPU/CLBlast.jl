
for (func, elty) in [(:CLBlastShad, Float32), (:CLBlastDhad, Float64),
                     (:CLBlastChad, Complex64), (:CLBlastZhad, Complex128)]
    #TODO: (:CLBlastHhad, Float16)

    @eval function $func(n::Integer, alpha::$elty,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         y_buffer::cl.CL_mem, y_offset::Integer, y_inc::Integer,
                         beta::$elty,
                         z_buffer::cl.CL_mem, z_offset::Integer, z_inc::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast), 
            cl.CL_int,
            (UInt64, $elty, Ptr{Void}, UInt64, UInt64, Ptr{Void}, UInt64, UInt64, $elty, Ptr{Void}, UInt64, UInt64, Ptr{Void}, Ptr{Void}),
            n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, beta, z_buffer, z_offset, z_inc, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(STDERR, "Calling function $(string(func)) failed!")
            throw(cl.CLError(err))
        end
        return err
    end

    @eval function had!(n::Integer, α::Number,
                        x::cl.CLArray{$elty}, x_inc::Integer,
                        y::cl.CLArray{$elty}, y_inc::Integer,
                        β::Number,
                        z::cl.CLArray{$elty}, z_inc::Integer;
                        queue::cl.CmdQueue=cl.queue(z))
        # output event
        event = cl.Event(C_NULL)
        alpha = convert($elty, α)
        beta = convert($elty, β)

        $func(Csize_t(n), alpha,
              pointer(x), Csize_t(0), Csize_t(x_inc),
              pointer(y), Csize_t(0), Csize_t(y_inc),
              beta,
              pointer(z), Csize_t(0), Csize_t(z_inc),
              queue, event)

        # wait for kernel
        cl.wait(event)

        nothing
    end

end
