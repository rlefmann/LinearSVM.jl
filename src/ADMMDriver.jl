using DistributedArrays

include("ADMMWorker.jl")


"""
`distributeData(x,y)`

Distributes x and y to the available worker processes and returns
them as distributed Arrays.
"""
function distributeData(x::Array{Float64,2}, y::Array{Float64,1})
    print("distributing x and y...\t\t")
    tic()
    N = nprocs()-1
    if N<2
        error("number of workers needs to be at least 2.")
    end
    x = distribute(x, dist=[N,1])
    y = distribute(y)
    @printf "finished after %.3f seconds\n" toq()
    return (x,y)
end


# TODO: if we dont normalize then z will grow very large. Where should the normalization happen??
"""
`admm(x,y;<keyword arguments>)`

Solves the SVM problem in parallel using the ADMM framework by Stephen Boyd.
See https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf.
The algorithm is developed by Caoxie Zhang, Honglak Lee and Kang G. Shin.
See http://www.jmlr.org/proceedings/papers/v22/zhang12a/zhang12a.pdf.
For solving the subproblem the dual coordinate descent method is used.


### Keyword arguments
* `C=1.0`:
the parameter C of the SVM problem
* `rho=1.0`
* `shrinking=false`: the method uses the shrinking heuristic by Thorsten Joachims if true. See https://www.cs.cornell.edu/people/tj/publications/joachims_99a.pdf
* `outer_iterations=10`: the number of admm iterations
* `inner_iterations=10`: the number of iterations of the dual coordinate descent performed during each admm iteration
* `parallel=true`: The subproblem is solved in parallel if true

### Returns
* weight vector `w`
"""
function admm(
x::DistributedArrays.DArray{Float64,2,Array{Float64,2}},
y::DistributedArrays.DArray{Float64,1,Array{Float64,1}};
C=1.0,
rho=1.0,
outer_iterations=10,
inner_iterations=10,
parallel=true
)
    tic()
    N = nprocs()-1
    if N<2
        error("number of workers needs to be at least 2.")
    end

    m,d = size(x)

    # initialize vectors:
    w = zeros(d,N) # every column w[:,i] is the w vector of one of the workers
    z = zeros(d)
    u = zeros(d,N)
    
    for t in 1:outer_iterations

        ## w-update:
        if parallel # parallel update
            @sync for i in 1:N
                @async begin
                    v = z - u[:,i]
                    worker = workers()[i]
                    w[:,i] = remotecall_fetch(worker,dualcdForADMM2,x,y,v,rho,C,true,inner_iterations,0.001)
                end
            end
        else # sequential update (but still distributed)
            for i in 1:N
                v = z - u[:,i]
                worker = workers()[i]
                w[:,i] = remotecall_fetch(worker,dualcdForADMM2,x,y,v,rho,C,true,inner_iterations,0.001)
                #w[:,i] /= vecnorm(w[:,i]) # normalizing w here leads to very slow convergence
            end
        end

        ## z-update:
        z = sum(broadcast(+,w,z), 2)
        #z = sum(w+z,2) # summiere w+z zeilenweise auf (eine Summe je Zeile)
        z /= (N+1/rho)
        z = vec(z)
        #z /= vecnorm(z)
        
        ## u-update:
        # subtract z from each column of u+w and store the result in u:
        u = broadcast(-,u+w,z)
    end

    @printf "finished after %.3f seconds (%d iterations)\n" toq() outer_iterations
    return z
end

