
for (func, elty) in [(:CLBlastSscal, Float32), (:CLBlastDscal, Float64),
                     (:CLBlastCscal, Complex64), (:CLBlastZscal, Complex128)]
    #TODO: (:CLBlastHscal, Float16)

    @eval function $func(n::Integer, alpha::$elty,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast), 
            cl.CL_int,
            (Csize_t, $elty, Ptr{Void}, Csize_t, Csize_t, Ptr{Void}, Ptr{Void}),
            n, alpha, x_buffer, x_offset, x_inc, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(STDERR, "Calling function $(string($func)) failed!")
            throw(cl.CLError(err))
        end
        return err
    end

    @eval function scal!(n::Integer, α::Number,
                         x::cl.CLArray{$elty}, x_inc::Integer;
                         queue::cl.CmdQueue=cl.queue(x))
        # output event
        event = cl.Event(C_NULL)
        alpha = convert($elty, α)

        $func(Csize_t(n), alpha,
              pointer(x), Csize_t(0), Csize_t(x_inc),
              queue, event)

        # wait for kernel
        cl.wait(event)

        nothing
    end

end
