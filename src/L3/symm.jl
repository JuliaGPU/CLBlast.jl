
@compat for (func, elty) in [(:CLBlastSsymm, Float32), (:CLBlastDsymm, Float64),
                     (:CLBlastCsymm, ComplexF32), (:CLBlastZsymm, ComplexF64)]
    #TODO: (:CLBlastHsymm, Float16)

    @eval function $func(layout::CLBlastLayout,
                         side::CLBlastSide, triangle::CLBlastTriangle,
                         m::Integer, n::Integer,
                         alpha::$elty,
                         a_buffer::cl.CL_mem, a_offset::Integer, a_ld::Integer,
                         b_buffer::cl.CL_mem, b_offset::Integer, b_ld::Integer,
                         beta::$elty,
                         c_buffer::cl.CL_mem, c_offset::Integer, c_ld::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Cint, Cint, Cint, Csize_t, Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t,
              Ptr{Cvoid}, Csize_t, Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            Cint(layout), Cint(side), Cint(triangle), m, n, alpha, a_buffer, a_offset, a_ld,
              b_buffer, b_offset, b_ld, beta, c_buffer, c_offset, c_ld, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(CLBlastError(err))
        end
        return err
    end

    @eval function symm!(_side::Char, uplo::Char, α::Number, A::cl.CLArray{$elty,2},
                         B::cl.CLArray{$elty,2}, β::Number, C::cl.CLArray{$elty,2};
                         queue::cl.CmdQueue=cl.queue(C))
        # check and convert arguments
        if _side == 'L'
            side = CLBlastSideLeft
        elseif _side == 'R'
            side = CLBlastSideRight
        else
            throw(ArgumentError("Side marker `side` is $(_side) but only 'L' and 'R' are allowed."))
        end
        if uplo == 'U'
            triangle = CLBlastTriangleUpper
        elseif uplo == 'L'
            triangle = CLBlastTriangleLower
        else
            throw(ArgumentError("Upper/lower marker `uplo` is $(uplo) but only 'U' and 'L' are allowed."))
        end
        m, n = size(C)
        j = size(A,1)
        if j != size(A,2)
            throw(DimensionMismatch("`A` has dimensions $(size(A)) but must be square."))
        end
        if j != (_side == 'L' ? m : n)
            throw(DimensionMismatch("`A` has size $(size(A)), `C` has size ($m,$n)."))
        end
        if size(B,2) != n
            throw(DimensionMismatch("`B` has second dimension $(size(B,2)) but needs to match second dimension of `C`, $n."))
        end
        alpha = convert($elty, α)
        beta  = convert($elty, β)
        layout = CLBlastLayoutColMajor

        # output event
        event::cl.Event = cl.Event(C_NULL)

        $func(layout, side, triangle,
              m, n,
              alpha,
              pointer(A), 0, size(A,1),
              pointer(B), 0, size(B,1),
              beta,
              pointer(C), 0, size(C,1),
              queue, event)

        # wait for kernel
        cl.wait(event)

        C
    end

end
