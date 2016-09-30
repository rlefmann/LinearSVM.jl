module LinearSVM

include("DualCD.jl")
include("Pegasos.jl")
include("ADMMDriver.jl")
#include("ADMMWorker.jl")
include("Utilities.jl")

# DualCD.jl:
export dualcd
export pegasos

# ADMMDriver.jl:
export distributeData
export admm

# Utilities.jl:
export loadDataset
export testResult

end # module
