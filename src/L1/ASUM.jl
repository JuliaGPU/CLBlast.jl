for func in [:CLBlastSasum, :CLBlastDasum, :CLBlastScasum, :CLBlastDzasum, :CLBlastHasum]
    #TODO: 

    @eval function $func(n::Integer, out_buffer::cl.CL_mem, out_offset::Integer, 
                         x_buffer::cl.CL_mem, x_offset::Integer, x_inc::Integer,
                         queue::cl.CmdQueue, event::cl.Event)
        err = ccall(
            ($(string(func)), libCLBlast), 
            cl.CL_int,
            (UInt64, Ptr{Void}, UInt64, Ptr{Void}, UInt64, UInt64, Ptr{Void}, Ptr{Void}),
            n, out_buffer, out_offset, x_buffer, x_offset, x_inc, Ref(queue), Ref(event)
        )
        if err != cl.CL_SUCCESS
            println(STDERR, "Calling function $(string(func)) failed!")
            throw(cl.CLError(err))
        end
        return err
    end

end
