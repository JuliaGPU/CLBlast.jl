
for (func, elty) in [(:CLBlastCdotu, Complex64), (:CLBlastZdotu, Complex128)]

    @eval function $func(n::Integer, out_buffer::cl.CL_mem, out_offset::Integer,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         y_buffer::cl.CL_mem, y_offset::Integer, y_inc::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast), 
            cl.CL_int,
            (Csize_t, Ptr{Void}, Csize_t, Ptr{Void}, Csize_t, Csize_t, Ptr{Void}, Csize_t, Csize_t, Ptr{Void}, Ptr{Void}),
            n, out_buffer, out_offset, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(STDERR, "Calling function $(string($func)) failed!")
            throw(cl.CLError(err))
        end
        return err
    end

    @eval function dotu(n::Integer,
                        x::cl.CLArray{$elty}, x_inc::Integer,
                        y::cl.CLArray{$elty}, y_inc::Integer;
                        queue::cl.CmdQueue=cl.queue(x))
        # output buffer and event
        ctx = cl.context(queue)
        out = zeros($elty, 1)
        out_buffer = cl.Buffer($elty, ctx, (:rw, :copy), hostbuf=out)
        event = cl.Event(C_NULL)

        $func(Csize_t(n), pointer(out_buffer), Csize_t(0),
              pointer(x), Csize_t(0), Csize_t(x_inc),
              pointer(y), Csize_t(0), Csize_t(y_inc),
              queue, event)

        # wait for kernel and read return value
        cl.wait(event)
        cl.enqueue_read_buffer(queue, out_buffer, out, Csize_t(0), nothing, true)

        return first(out)
    end

end
