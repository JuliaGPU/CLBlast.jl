
@compat for (func, elty) in [(:CLBlastSger, Float32), (:CLBlastDger, Float64),
                     (:CLBlastCgerc, ComplexF32), (:CLBlastZgerc, ComplexF64)]
    #TODO: (:CLBlastHger, Float16)

    @eval function $func(layout::CLBlastLayout,
                         m::Integer, n::Integer,
                         alpha::$elty,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         y_buffer::cl.CL_mem, y_offset::Integer, y_inc::Integer,
                         a_buffer::cl.CL_mem, a_offset::Integer, a_ld::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Cint, Csize_t, Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t,
              Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            Cint(layout), m, n, alpha, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc,
               a_buffer, a_offset, a_ld, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(CLBlastError(err))
        end
        return err
    end

    @eval function ger!(α::Number, x::cl.CLArray{$elty}, y::cl.CLArray{$elty},
                        A::cl.CLArray{$elty,2};
                        queue::cl.CmdQueue=cl.queue(A))
        # check and convert arguments
        m, n = size(A)
        if length(x) != m || length(y) != n
            throw(DimensionMismatch("`A` has dimensions $(size(A)), `x` has length $(length(x)) and `y` has length $(length(y))."))
        end
        alpha = convert($elty, α)
        layout = CLBlastLayoutColMajor

        # output event
        event::cl.Event = cl.Event(C_NULL)

        $func(layout,
              m, n,
              alpha,
              pointer(x), 0, 1,
              pointer(y), 0, 1,
              pointer(A), 0, size(A,1),
              queue, event)

        # wait for kernel
        cl.wait(event)

        A
    end

end
