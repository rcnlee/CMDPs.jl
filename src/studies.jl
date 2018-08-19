
struct RewardvsNSim
    mdp
    bandit_t
    n_iters::Int
    seed::Int
    max_steps::Int
    metadata::Dict
end
function RewardvsNSim(mdp, bandit_t, n_iters, seed, max_steps)
    RewardvsNSim(mdp, bandit_t, n_iters, seed, max_steps, Dict(:sim_algorithm=>string(bandit_t), :sim_mdp=>string(mdp), :sim_n_ters=>n_iters, :sim_seed=>seed, :sim_max_steps=>max_steps))
end
function POMDPs.simulate(sim::RewardvsNSim) 
    rng = MersenneTwister(sim.seed)
    mdp = sim.mdp
    b = sim.bandit_t(mdp)
    solver = ModularSolver(mdp, b, sim.seed; n_iterations=sim.n_iters)
    policy = solve(solver, mdp)
    s0 = initial_state(mdp, rng)
    hr = HistoryRecorder(; max_steps=sim.max_steps, rng=rng)
    h = simulate(hr, mdp, policy, s0)
    r_total = sum(h.reward_hist)
    r_total
end

@with_kw struct RewardvsN 
    mdp = Walk1D()
    bandits=[DPWBandit,RandomBandit,CBTSBandit]
    n_seeds=5
    n_iters=[10,25,50]
    max_steps=20
end
struct RewardvsNResult
    title::String
    data::DataFrame
end
function RewardvsN(::Type{Walk1D}) 
    mdp = Walk1D()
    RewardvsN(mdp, [DPWBandit,RandomBandit,CBTSDPWBandit], 200, [10,20,50,100,200,400,700,1000,1250], mdp.p.t_max)
end
RewardvsN(::Type{VDPTagMDP}) = RewardvsN(VDPTagMDP(), [DPWBandit,RandomBandit,CBTSDPWBandit], 500, [10,50,100,200,500,1000,1500,2000,2500,3000], 10)
RewardvsN(::Type{InvertedPendulum}) = RewardvsN(InvertedPendulum(), [DPWBandit,RandomBandit,CBTSDPWBandit], 20, [10,50,100,200,500,1000], 100)
RewardvsN(::Type{MountainCar}) = RewardvsN(MountainCar(), [DPWBandit,RandomBandit,CBTSDPWBandit], 20, [100,500,1000,2000,5000,10000], 20)
RewardvsN(::Type{LightDark2D}) = RewardvsN(LightDark2D(), [DPWBandit,RandomBandit,CBTSDPWBandit], 200, [50,100,200,500,1000,1500,2000], 10)

function generate_sim_q(study::RewardvsN)
    q = []
    for b in study.bandits, n in study.n_iters, i in 1:study.n_seeds
        push!(q, RewardvsNSim(study.mdp, b, n, i, study.max_steps))
    end
    q
end

function run_study(study::RewardvsN)
    df = DataFrame([String, Int, Int, Float64], [:alg, :seed, :n_iters, :r_total], 0)
    q = generate_sim_q(study)
    result = pmap(POMDPs.simulate, q)
    for (r, sim) in zip(result, q)
        push!(df, [string(sim.bandit_t), sim.seed, sim.n_iters, r])
    end
    RewardvsNResult(string(study.mdp), df)
end
@recipe function plot(result::RewardvsNResult, more_results::RewardvsNResult...)
    title := result.title
    xlabel := "number of iterations"
    ylabel := "reward"
    data = vcat(result.data, more_results...)
    df = aggregate(data, [:alg, :n_iters], [mean, std])[[:alg, :n_iters, :r_total_mean, :r_total_std]]
    for dd in groupby(df, :alg) 
        @series begin
            dd = sort(dd[:], :n_iters)
            label := split(dd[1,:alg], ".")[end]
            legend := :bottomright
            err := dd[:r_total_std] ./ sqrt(maximum(result.data[:seed]))
            dd[:n_iters], dd[:r_total_mean]
        end
    end
end
