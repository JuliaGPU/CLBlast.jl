
@compat for (func, elty) in [(:CLBlastSgemm, Float32), (:CLBlastDgemm, Float64),
                     (:CLBlastCgemm, ComplexF32), (:CLBlastZgemm, ComplexF64)]
    #TODO: (:CLBlastHgemm, Float16)

    @eval function $func(layout::CLBlastLayout,
                         a_transpose::CLBlastTranspose, b_transpose::CLBlastTranspose,
                         m::Integer, n::Integer, k::Integer,
                         alpha::$elty,
                         a_buffer::cl.CL_mem, a_offset::Integer, a_ld::Integer,
                         b_buffer::cl.CL_mem, b_offset::Integer, b_ld::Integer,
                         beta::$elty,
                         c_buffer::cl.CL_mem, c_offset::Integer, c_ld::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Cint, Cint, Cint, Csize_t, Csize_t, Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t,
              Ptr{Cvoid}, Csize_t, Csize_t, $elty, Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            Cint(layout), Cint(a_transpose), Cint(b_transpose), m, n, k, alpha, a_buffer, a_offset, a_ld,
              b_buffer, b_offset, b_ld, beta, c_buffer, c_offset, c_ld, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(CLBlastError(err))
        end
        return err
    end

    @eval function gemm!(transA::Char, transB::Char, α::Number, A::cl.CLArray{$elty,2},
                         B::cl.CLArray{$elty,2}, β::Number, C::cl.CLArray{$elty,2};
                         queue::cl.CmdQueue=cl.queue(C))
        # check and convert arguments
        if transA == 'N'
            a_transpose = CLBlastTransposeNo
        elseif transA == 'T' || (transA == 'C' && $elty <: Real)
            a_transpose = CLBlastTransposeYes
        elseif transA == 'C' && $elty <: Complex
            a_transpose = CLBlastTransposeConjugate
        else
            throw(ArgumentError("Transpose marker `transA` is $(transA) but only 'N', 'T', and 'C' are allowed."))
        end
        if transB == 'N'
            b_transpose = CLBlastTransposeNo
        elseif transB == 'T' || (transB == 'C' && $elty <: Real)
            b_transpose = CLBlastTransposeYes
        elseif transB == 'C' && $elty <: Complex
            b_transpose = CLBlastTransposeConjugate
        else
            throw(ArgumentError("Transpose marker `transB` is $(transB) but only 'N', 'T', and 'C' are allowed."))
        end
        m = size(A, transA == 'N' ? 1 : 2)
        ka = size(A, transA == 'N' ? 2 : 1)
        kb = size(B, transB == 'N' ? 1 : 2)
        n = size(B, transB == 'N' ? 2 : 1)
        if ka != kb || m != size(C,1) || n != size(C,2)
            throw(DimensionMismatch("`A` has size ($m,$ka), `B` has size ($kb,$n), `C` has size $(size(C))."))
        end
        alpha = convert($elty, α)
        beta  = convert($elty, β)
        layout = CLBlastLayoutColMajor

        # output event
        event::cl.Event = cl.Event(C_NULL)

        $func(layout, a_transpose, b_transpose,
              m, n, ka,
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
