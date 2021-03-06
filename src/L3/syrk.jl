
for (func, elty) in [(:CLBlastSsyrk, Float32), (:CLBlastDsyrk, Float64),
                     (:CLBlastCsyrk, ComplexF32), (:CLBlastZsyrk, ComplexF64)]
    #TODO: (:CLBlastHsyrk, Float16)

    @eval function $func(layout::CLBlastLayout,
                         triangle::CLBlastTriangle, a_transpose::CLBlastTranspose,
                         n::Integer, k::Integer,
                         alpha::$elty,
                         a_buffer::cl.CL_mem, a_offset::Integer, a_ld::Integer,
                         beta::$elty,
                         c_buffer::cl.CL_mem, c_offset::Integer, c_ld::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Cint, Cint, Cint, Csize_t, Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t,
              $elty, Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            Cint(layout), Cint(triangle), Cint(a_transpose), n, k, alpha,
              a_buffer, a_offset, a_ld, beta, c_buffer, c_offset, c_ld, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(CLBlastError(err))
        end
        return err
    end

    @eval function syrk!(uplo::Char, trans::Char, α::Number,
                         A::cl.CLArray{$elty,2}, β::Number, C::cl.CLArray{$elty,2};
                         queue::cl.CmdQueue=cl.queue(C))
        # check and convert arguments
        if trans == 'N'
            a_transpose = CLBlastTransposeNo
        elseif trans == 'T'
            a_transpose = CLBlastTransposeYes
        else
            throw(ArgumentError("Transpose marker `trans` is $(trans) but only 'N' and 'T' are allowed."))
        end
        if uplo == 'U'
            triangle = CLBlastTriangleUpper
        elseif uplo == 'L'
            triangle = CLBlastTriangleLower
        else
            throw(ArgumentError("Upper/lower marker `uplo` is $(uplo) but only 'U' and 'L' are allowed."))
        end
        n = size(C,1)
        if n != size(C,2)
            throw(DimensionMismatch("`C` has dimensions $(size(C)) but must be square."))
        end
        nn = size(A, trans == 'N' ? 1 : 2)
        if nn != n
            throw(DimensionMismatch("`C` has size ($n,$n), corresponding dimension of `A` is $nn."))
        end
        k  = size(A, trans == 'N' ? 2 : 1)
        alpha = convert($elty, α)
        beta  = convert($elty, β)
        layout = CLBlastLayoutColMajor

        # output event
        event::cl.Event = cl.Event(C_NULL)

        $func(layout, triangle, a_transpose,
              n, k,
              alpha,
              pointer(A), 0, size(A,1),
              beta,
              pointer(C), 0, size(C,1),
              queue, event)

        # wait for kernel
        cl.wait(event)

        C
    end

end
