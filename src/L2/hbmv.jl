
@compat for (func, elty) in [(:CLBlastChbmv, ComplexF32), (:CLBlastZhbmv, ComplexF64)]
    #TODO: (:CLBlastHhbmv, Float16)

    @eval function $func(layout::CLBlastLayout, triangle::CLBlastTriangle,
                         n::Integer, k::Integer,
                         alpha::$elty,
                         a_buffer::cl.CL_mem, a_offset::Integer, a_ld::Integer,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         beta::$elty,
                         y_buffer::cl.CL_mem, y_offset::Integer, y_inc::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Cint, Cint, Csize_t, Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t,
              Ptr{Cvoid}, Csize_t, Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            Cint(layout), Cint(triangle), n, k, alpha, a_buffer, a_offset, a_ld,
              x_buffer, x_offset, x_inc, beta, y_buffer, y_offset, y_inc, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(CLBlastError(err))
        end
        return err
    end

    @eval function hbmv!(uplo::Char, k::Integer, α::Number,
                         A::cl.CLArray{$elty,2}, x::cl.CLArray{$elty},
                         β::Number, y::cl.CLArray{$elty};
                         queue::cl.CmdQueue=cl.queue(y))
        # check and convert arguments
        n = size(A,2)
        if length(x) != n || length(y) != n
            throw(DimensionMismatch("x has length $(length(x)) and y has length $(length(y)) while $n is required."))
        end
        if uplo == 'U'
            triangle = CLBlastTriangleUpper
        elseif uplo == 'L'
            triangle = CLBlastTriangleLower
        else
            throw(ArgumentError("Upper/lower marker `uplo` is $(uplo) but only 'U' and 'L' are allowed."))
        end
        alpha = convert($elty, α)
        beta  = convert($elty, β)
        layout = CLBlastLayoutColMajor

        # output event
        event::cl.Event = cl.Event(C_NULL)

        $func(layout, triangle,
              n, k,
              alpha,
              pointer(A), 0, size(A,1),
              pointer(x), 0, 1,
              beta,
              pointer(y), 0, 1,
              queue, event)

        # wait for kernel
        cl.wait(event)

        y
    end

end
