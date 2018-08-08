#import Base.LinAlg.BLAS: nrm2

@compat for (func, elty) in [(:CLBlastSnrm2, Float32), (:CLBlastDnrm2, Float64),
                     (:CLBlastScnrm2, ComplexF32), (:CLBlastDznrm2, ComplexF64)]
    #TODO: (:CLBlastHnrm2, Float16)

    @eval function $func(n::Integer, out_buffer::cl.CL_mem, out_offset::Integer,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast),
            cl.CL_int,
            (Csize_t, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Csize_t, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}),
            n, out_buffer, out_offset, x_buffer, x_offset, x_inc, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(stderr, "Calling function $(string($func)) failed!")
            throw(cl.CLError(err))
        end
        return err
    end

    @eval function nrm2(n::Integer, x::cl.CLArray{$elty}, x_inc::Integer;
                        queue::cl.CmdQueue=cl.queue(x))
        # output buffer and event
        ctx = cl.context(queue)
        out = zeros($elty, 1)
        out_buffer = cl.Buffer($elty, ctx, (:rw, :copy), hostbuf=out)
        event::cl.Event = cl.Event(C_NULL)

        $func(Csize_t(n), pointer(out_buffer), Csize_t(0), pointer(x), Csize_t(0), Csize_t(x_inc),
              queue, event)

        # wait for kernel and read return value
        cl.wait(event)
        cl.enqueue_read_buffer(queue, out_buffer, out, Csize_t(0), nothing, true)

        return real(first(out))
    end

end
