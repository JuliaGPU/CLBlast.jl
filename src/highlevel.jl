#import Base.LinAlg.BLAS: asum

########################### L1 functions ###############################

# ASUM
for (func, elty) in [(:CLBlastSasum, Float32), (:CLBlastDasum, Float64),
                     (:CLBlastScasum, Complex64), (:CLBlastDzasum, Complex128)]

    @eval function asum(n::Integer, x::cl.CLArray{$elty}, x_inc::Integer;
                        queue::cl.CmdQueue=cl.queue(x))
        # output buffer and event
        ctx = cl.context(queue)
        out = zeros($elty, 1)
        out_buffer = cl.Buffer($elty, ctx, (:rw, :copy), hostbuf=out)
        event = cl.Event(C_NULL)

        $func(Csize_t(n), pointer(out_buffer), Csize_t(0), pointer(x), Csize_t(0), Csize_t(x_inc),
              queue, event)

        # read return value
        cl.wait(event)
        cl.enqueue_read_buffer(queue, out_buffer, out, Csize_t(0), nothing, true)

        return real(first(out))
    end

end
