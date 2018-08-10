
@compat for (func, elty) in [(:CLBlastStrsm, Float32), (:CLBlastDtrsm, Float64),
                     (:CLBlastCtrsm, ComplexF32), (:CLBlastZtrsm, ComplexF64)]

    @eval function $func(layout::CLBlastLayout, side::CLBlastSide, triangle::CLBlastTriangle,
                         a_transpose::CLBlastTranspose, diagonal::CLBlastDiagonal,
                         m::Integer, n::Integer,
                         alpha::$elty,
                         a_buffer::cl.CL_mem, a_offset::Integer, a_ld::Integer,
                         b_buffer::cl.CL_mem, b_offset::Integer, b_ld::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Cint, Cint, Cint, Cint, Cint, Csize_t, Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t,
              Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            Cint(layout), Cint(side), Cint(triangle), Cint(a_transpose), Cint(diagonal),
              m, n, alpha, a_buffer, a_offset, a_ld, b_buffer, b_offset, b_ld, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(CLBlastError(err))
        end
        return err
    end

    @eval function trsm!(_side::Char, uplo::Char, transA::Char, diag::Char,
                         α::Number, A::cl.CLArray{$elty,2}, B::cl.CLArray{$elty,2};
                         queue::cl.CmdQueue=cl.queue(B))
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
        if transA == 'N'
            a_transpose = CLBlastTransposeNo
        elseif transA == 'T' || (transA == 'C' && $elty <: Real)
            a_transpose = CLBlastTransposeYes
        elseif transA == 'C' && $elty <: Complex
            a_transpose = CLBlastTransposeConjugate
        else
            throw(ArgumentError("Transpose marker `transA` is $(transA) but only 'N', 'T', and 'C' are allowed."))
        end
        if diag == 'N'
            diagonal = CLBlastDiagonalNonUnit
        elseif diag == 'U'
            diagonal = CLBlastDiagonalUnit
        else
            throw(ArgumentError("Diagonal marker `diag` is $(diag) but only 'N' and 'U' are allowed."))
        end
        m, n = size(B)
        nA = size(A,1)
        if nA != size(A,2)
            throw(DimensionMismatch("`A` has dimensions $(size(A)) but must be square."))
        end
        if nA != (_side == 'L' ? m : n)
            throw(DimensionMismatch("Size of A, $(size(A)), doesn't match $_side size of B with dims, $(size(B))."))
        end
        alpha = convert($elty, α)
        layout = CLBlastLayoutColMajor

        # output event
        event::cl.Event = cl.Event(C_NULL)

        $func(layout, side, triangle, a_transpose, diagonal,
              m, n,
              alpha,
              pointer(A), 0, size(A,1),
              pointer(B), 0, size(B,1),
              queue, event)

        # wait for kernel
        # the additional check is due to https://github.com/CNugteren/CLBlast/issues/311
        if event != cl.Event(C_NULL)
            cl.wait(event)
        end

        B
    end

end
