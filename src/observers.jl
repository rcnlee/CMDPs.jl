using Observers
using Plots

export AQObserver

struct AQObserver <: Observer
    snode::Int
    actions::Vector{Any}
    ns::Vector{Any}
    Xs::Vector{Any}
    ys::Vector{Any}
end
AQObserver(snode::Int) = AQObserver(snode, [], [], [], [])

function Observers.notify_observer!(o::AQObserver, b::ModularBandit; kwargs...)
    snode = kwargs_get(kwargs, :snode)
    o.snode == snode || return  #only monitor snode

    p = kwargs_get(kwargs, :planner)
    tree = get(p.tree)

    sanode = kwargs_get(kwargs, :sanode)
    push!(o.actions, tree.a_labels[sanode])

    children = tree.children[snode]
    actions = [tree.a_labels[c] for c in children]
    qs = [tree.q[c] for c in children]
    ns = [tree.n[c] for c in children]

    push!(o.Xs, actions) 
    push!(o.ys, qs) 
    push!(o.ns, ns)
end

@recipe function plot(o::AQObserver, i, ylim=nothing)
    @series begin
        seriestype := :scatter
        xlabel := "action"
        ylabel := "q"
        title := "n=$i"
        if ylim != nothing
            ylim := ylim
        end
        x = o.Xs[i]
        y = o.ys[i]
        x, y 
    end
end

Base.length(o::AQObserver) = length(o.Xs)

function Plots.animate(o::AQObserver, fname="./aqobserver.gif"; fps=2, ylim=nothing)
    anim = @animate for i=1:length(o)
        Plots.plot(o, i, ylim=ylim)
    end
    gif(anim, fname; fps=fps)
end

