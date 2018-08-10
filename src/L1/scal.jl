
@compat for (func, elty) in [(:CLBlastSscal, Float32), (:CLBlastDscal, Float64),
                     (:CLBlastCscal, ComplexF32), (:CLBlastZscal, ComplexF64)]
    #TODO: (:CLBlastHscal, Float16)

    @eval function $func(n::Integer, alpha::$elty,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            n, alpha, x_buffer, x_offset, x_inc, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(cl.CLError(err))
        end
        return err
    end

    @eval function scal!(n::Integer, α::Number,
                         x::cl.CLArray{$elty}, x_inc::Integer;
                         queue::cl.CmdQueue=cl.queue(x))
        # output event
        event::cl.Event = cl.Event(C_NULL)
        alpha = convert($elty, α)

        $func(Csize_t(n), alpha,
              pointer(x), Csize_t(0), Csize_t(x_inc),
              queue, event)

        # wait for kernel
        cl.wait(event)

        x
    end

    @eval function scal!(α::Number,
                         x::cl.CLArray{$elty};
                         queue::cl.CmdQueue=cl.queue(x))
        scal!(length(x), α, x, 1, queue=queue)
    end

end
