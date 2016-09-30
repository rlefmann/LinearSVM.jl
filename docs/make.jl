using Documenter, LinearSVM

makedocs(modules=[LinearSVM], doctest=true)
deploydocs(deps = Deps.pip("mkdocs", "python-markdown-math"),
repo   = "https://github.com/rlefmann/LinearSVM.jl.git",
julia  = "0.4.5",
osname = "linux")
