
@compat for (func, elty) in [(:CLBlastSaxpy, Float32), (:CLBlastDaxpy, Float64),
                     (:CLBlastCaxpy, ComplexF32), (:CLBlastZaxpy, ComplexF64)]
    #TODO: (:CLBlastHaxpy, Float16)

    @eval function $func(n::Integer, alpha::$elty,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         y_buffer::cl.CL_mem, y_offset::Integer, y_inc::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(cl.CLError(err))
        end
        return err
    end

    @eval function axpy!(n::Integer, α::Number,
                         x::cl.CLArray{$elty}, x_inc::Integer,
                         y::cl.CLArray{$elty}, y_inc::Integer;
                         queue::cl.CmdQueue=cl.queue(x))
        # output event
        event::cl.Event = cl.Event(C_NULL)
        alpha = convert($elty, α)

        $func(Csize_t(n), alpha,
              pointer(x), Csize_t(0), Csize_t(x_inc),
              pointer(y), Csize_t(0), Csize_t(y_inc),
              queue, event)

        # wait for kernel
        cl.wait(event)

        y
    end

    @eval function axpy!(α::Number,
                         x::cl.CLArray{$elty},
                         y::cl.CLArray{$elty};
                         queue::cl.CmdQueue=cl.queue(x))
        n = length(x)
        if n != length(y)
            throw(DimensionMismatch("x has length $n while y has length $(length(y))!"))
        end

        axpy!(n, α, x, 1, y, 1, queue=queue)
    end

end
