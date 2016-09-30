using Documenter, LinearSVM

makedocs(modules=[LinearSVM], doctest=true)
deploydocs(deps = Deps.pip("mkdocs", "python-markdown-math"),
repo   = "github.com/GITHUBNAME/GITHUBREPO.git",
julia  = "0.4.5",
osname = "linux")
