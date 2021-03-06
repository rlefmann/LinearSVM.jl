using DistributedArrays

function dualcdForADMM2(
x::DistributedArrays.DArray{Float64,2,Array{Float64,2}},
y::DistributedArrays.DArray{Float64,1,Array{Float64,1}},
v::Array{Float64,1},
rho,
C,
shrinking,
iterations,
eps)
    dualcdForADMM(x,y,v,rho=rho,C=C,shrinking=shrinking,iterations=iterations,eps=eps)
end



function dualcdForADMM(
x::DistributedArrays.DArray{Float64,2,Array{Float64,2}},
y::DistributedArrays.DArray{Float64,1,Array{Float64,1}},
v::Array{Float64,1};
rho=1.0,
C=2.0,
shrinking=false,
iterations=1000,
eps=0.001)
    x = localpart(x)
    y = localpart(y)

    s,d = size(x) # s: number of data points
    alpha = zeros(s)

    # compute initial w:
    # TODO: 1/rho missing
    w = v + (1.0/rho)*At_mul_B(x, y.*alpha)

    
    # Precompute (Q+D)_ii:
    QD = Array{Float64,1}(s)
    for i in 1:s
        xi = vec(x[i,:])
        QD[i] = (0.5/C)
        QD[i] += vecdot(xi,xi) # doesnt have to be multiplied by y[i]*y[i] since this product is always 1
    end
    
    
    # the active set as a set of booleans.
    # if an entry is true, then the corresponding point
    # is active
    active = trues(s)
    num_active = s
    
    PGmax_old = Inf
    PGmin_old = -Inf

    t=0 # number of outer iterations
    
    while true
        t+=1
        
        PGmax_new = -Inf
        PGmin_new = Inf
        
        
        #TODO: permutate data in active set
        
        for i in 1:s
            if shrinking == false || active[i] == true
                xi = vec(x[i,:])
                G = y[i]*vecdot(w,xi) - 1.0 + (0.5/C)*alpha[i]
                
                # project gradient into feasible set:
                PG = 0
                if alpha[i] == 0 # exact zero check is fine here
                    if G > PGmax_old
                        active[i] = false
                        num_active-=1
                        continue
                    elseif G < 0
                        PG = G
                        #PG = min(0,G)
                    end
                else
                    PG = G
                end
                
                PGmax_new = max(PGmax_new, PG)
                PGmin_new = min(PGmin_new, PG)
                
                # perform update of alpha and w:
                if abs(PG) > 1e-6
                    alpha_old = alpha[i]
                    alpha[i] = max(0, alpha[i] - G/QD[i])
                    # w update:
                    w = w + (alpha[i] - alpha_old)*y[i]*xi
                end
                
            end
        end # end inner for loop

        if t >= iterations
            return w
        end
        
        if PGmax_new - PGmin_new < eps
            if num_active == s
                return w
            else # reset active set to see if we have a solution for the full set of data points
                num_active = s
                PGmax_old = Inf
                PGmin_old = -Inf
                continue # this missing might have caused problems before
            end
        end
        
        PGmax_old = PGmax_new
        PGmin_old = PGmin_new
        
        if PGmax_old <= 0
            PGmax_old = Inf
        end
        
        if PGmin_old >= 0
            PGmin_old = -Inf
        end
        
    end # end outer while loop
end
