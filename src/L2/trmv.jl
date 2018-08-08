
@compat for (func, elty) in [(:CLBlastStrmv, Float32), (:CLBlastDtrmv, Float64),
                     (:CLBlastCtrmv, ComplexF32), (:CLBlastZtrmv, ComplexF64)]
    #TODO: (:CLBlastHtrmv, Float16)

    @eval function $func(layout::CLBlastLayout, triangle::CLBlastTriangle,
                         a_transpose::CLBlastTranspose, diagonal::CLBlastDiagonal,
                         n::Integer,
                         a_buffer::cl.CL_mem, a_offset::Integer, a_ld::Integer,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Cint, Cint, Cint, Cint, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t,
              Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            Cint(layout), Cint(triangle), Cint(a_transpose), Cint(diagonal), n,
              a_buffer, a_offset, a_ld, x_buffer, x_offset, x_inc, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(CLBlastError(err))
        end
        return err
    end

    @eval function trmv!(uplo::Char, trans::Char, diag::Char,
                         A::cl.CLArray{$elty,2}, x::cl.CLArray{$elty};
                         queue::cl.CmdQueue=cl.queue(x))
        # check and convert arguments
        m, n = size(A)
        if m != n
            throw(DimensionMismatch("`A` has dimensions $(size(A)) but must be square."))
        end
        if length(x) != n
            throw(DimensionMismatch("`x` has length $(length(x)) while $n is required."))
        end
        if uplo == 'U'
            triangle = CLBlastTriangleUpper
        elseif uplo == 'L'
            triangle = CLBlastTriangleLower
        else
            throw(ArgumentError("Upper/lower marker `uplo` is $(uplo) but only 'U' and 'L' are allowed."))
        end
        if trans == 'N'
            a_transpose = CLBlastTransposeNo
        elseif trans == 'T' || (trans == 'C' && $elty <: Real)
            a_transpose = CLBlastTransposeYes
        elseif trans == 'C' && $elty <: Complex
            a_transpose = CLBlastTransposeConjugate
        else
            throw(ArgumentError("Transpose marker `trans` is $(uplo) but only 'N', 'T', and 'C' are allowed."))
        end
        if diag == 'N'
            diagonal = CLBlastDiagonalNonUnit
        elseif diag == 'U'
            diagonal = CLBlastDiagonalUnit
        else
            throw(ArgumentError("Diagonal marker `diag` is $(diag) but only 'N' and 'U' are allowed."))
        end
        layout = CLBlastLayoutColMajor

        # output event
        event::cl.Event = cl.Event(C_NULL)

        $func(layout, triangle, a_transpose, diagonal,
              n,
              pointer(A), 0, size(A,1),
              pointer(x), 0, 1,
              queue, event)

        # wait for kernel
        cl.wait(event)

        x
    end

end
