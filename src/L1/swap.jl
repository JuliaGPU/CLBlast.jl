
for (func, elty) in [(:CLBlastSswap, Float32), (:CLBlastDswap, Float64),
                     (:CLBlastCswap, Complex64), (:CLBlastZswap, Complex128)]
    #TODO: (:CLBlastHswap, Float16)

    @eval function $func(n::Integer,
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         y_buffer::cl.CL_mem, y_offset::Integer, y_inc::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast), 
            cl.CL_int,
            (UInt64, Ptr{Void}, UInt64, UInt64, Ptr{Void}, UInt64, UInt64, Ptr{Void}, Ptr{Void}),
            n, x_buffer, x_offset, x_inc, y_buffer, y_offset, y_inc, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(STDERR, "Calling function $(string(func)) failed!")
            throw(cl.CLError(err))
        end
        return err
    end

    @eval function swap!(n::Integer, 
                         x::cl.CLArray{$elty}, x_inc::Integer,
                         y::cl.CLArray{$elty}, y_inc::Integer;
                         queue::cl.CmdQueue=cl.queue(x))
        # output event
        event = cl.Event(C_NULL)

        $func(Csize_t(n),
              pointer(x), Csize_t(0), Csize_t(x_inc),
              pointer(y), Csize_t(0), Csize_t(y_inc),
              queue, event)

        # wait for kernel
        cl.wait(event)

        nothing
    end

end