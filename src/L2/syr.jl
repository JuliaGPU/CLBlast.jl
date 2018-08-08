
@compat for (func, elty) in [(:CLBlastSsyr, Float32), (:CLBlastDsyr, Float64)]
    #TODO: (:CLBlastHsyr, Float16)

    @eval function $func(layout::CLBlastLayout, triangle::CLBlastTriangle,
                         n::Integer,
                         alpha::$elty,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         a_buffer::cl.CL_mem, a_offset::Integer, a_ld::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Cint, Cint, Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t,
              Ptr{Cvoid}, Ptr{Cvoid}),
            Cint(layout), Cint(triangle), n, alpha, x_buffer, x_offset, x_inc,
               a_buffer, a_offset, a_ld, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(CLBlastError(err))
        end
        return err
    end

    @eval function syr!(uplo::Char, α::Number, x::cl.CLArray{$elty},
                        A::cl.CLArray{$elty,2};
                        queue::cl.CmdQueue=cl.queue(A))
        # check and convert arguments
        m, n = size(A)
        if m != n
            throw(DimensionMismatch("`A` has dimensions $(size(A)) but must be square."))
        end
        if length(x) != n
            throw(DimensionMismatch("`A` has dimensions $(size(A)) and `x` has length $(length(x))."))
        end
        if uplo == 'U'
            triangle = CLBlastTriangleUpper
        elseif uplo == 'L'
            triangle = CLBlastTriangleLower
        else
            throw(ArgumentError("Upper/lower marker `uplo` is $(uplo) but only 'U' and 'L' are allowed."))
        end
        alpha = convert($elty, α)
        layout = CLBlastLayoutColMajor

        # output event
        event::cl.Event = cl.Event(C_NULL)

        $func(layout, triangle,
              n,
              alpha,
              pointer(x), 0, 1,
              pointer(A), 0, size(A,1),
              queue, event)

        # wait for kernel
        cl.wait(event)

        A
    end

end
