using Observers
using Plots

export AQObserver

struct AQObserver <: Observer
    snode::Int
    Xs::Vector{Any}
    ys::Vector{Any}
end
AQObserver(snode::Int) = AQObserver(snode, [], [])

function Observers.notify_observer!(o::AQObserver, b::ModularBandit; kwargs...)
    snode = kwargs_get(kwargs, :snode)
    o.snode == snode || return  #only monitor snode

    p = kwargs_get(kwargs, :planner)
    tree = get(p.tree)

    children = tree.children[snode]
    actions = [tree.a_labels[c] for c in children]
    qs = [tree.q[c] for c in children]

    push!(o.Xs, actions) 
    push!(o.ys, qs) 
end

@recipe function plot(o::AQObserver, i)
    @series begin
        seriestype := :scatter
        xlabel := "action"
        ylabel := "q"
        title := "n=$i"
        x = o.Xs[i]
        y = o.ys[i]
        x, y 
    end
end

Base.length(o::AQObserver) = length(o.Xs)

function Plots.animate(o::AQObserver, fname="./aqobserver.gif"; kwargs...)
    anim = @animate for i=1:length(o)
        Plots.plot(o, i)
    end
    gif(anim, fname; kwargs...)
end

function kwargs_get(kwargs::Vector, key::Symbol)
    i = findfirst(x->x[1]==key, kwargs)
    i == 0 && error("key not found")
    kwargs[i][2]
end
