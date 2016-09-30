using StatsBase

function pegasos(x::Array{Float64,2}, y::Array{Float64,1}; lambda=0.05, iterations=1000000, batchsize=1, projection=false)
    tic()
    
    s,d = size(x)
    w = zeros(Float64,d) # start with w as the zero vector

    for t in 1:iterations
        # compute step size
        stepsize = 1.0/(lambda*t)

        # draw batchsize indices at random
        indices = sample(1:s,batchsize,replace=false)

        w_new = 0
        for i in indices
            xi = vec(x[i,:])
            if y[i]*vecdot(w,xi) < 1
                w_new += y[i]*xi
            end
        end
        w_new *= stepsize/float(batchsize)
        w_new += (1-1.0/t)*w
        w = w_new

        # perform the optional projection step
        # to the ball of radius 1/sqrt(lambda):
        if projection
            w = min(1,1/(sqrt(lambda)*norm(w)))*w
        end
        
    end

    @printf "finished after %d iterations and %.3f seconds\n" iterations toq()
    return w
end
