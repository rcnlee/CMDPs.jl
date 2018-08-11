module CMDPs

export generate_sim_q, run_study
export RewardvsN, RewardvsNResult

using POMDPs, POMDPToolbox, POMDPModels, MCTS
using Parameters, DataFrames
using Walk1DMDP, VDPTag2 #, MountainCar, InvertedPendulum, VDPTag2
using RecipesBase


function MCTS.DPWBandit(mdp::Walk1DMDP.Walk1D; kwargs...)
    DPWBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=5.0,
        k_action=0.5,
        alpha_action=0.85,
        kwargs...)
end
function MCTS.ModularSolver(mdp::Walk1DMDP.Walk1D, b::ModularBandit, seed=0;
                            kwargs...) 
    rng = MersenneTwister(seed)
    ModularSolver(;
        n_iterations=100,
        bandit=b,
        depth=mdp.p.t_max,
        k_state=0.5,
        alpha_state=0.0,
        keep_tree=true,
        check_repeat_state=true,
        rng=rng,
        init_Q=0.0,
        init_N=0,
        next_action=RandomGaussian(rng),
        estimate_value=RolloutEstimator(RandomGaussian(rng)),
        #estimate_value=TrackedRolloutEstimator(rng, RandomSolver(rng)),
        track_best_path=true,
        kwargs...
       )
end
function MCTS.DPWBandit(mdp::VDPTagMDP; kwargs...)
    DPWBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=5.0,
        k_action=0.5,
        alpha_action=0.85,
        kwargs...)
end
function MCTS.ModularSolver(mdp::VDPTagMDP, b::ModularBandit, seed=0; 
                            kwargs...) 
    rng = MersenneTwister(seed)
    ModularSolver(;
        n_iterations=100,
        bandit=b,
        depth=30,
        k_state=0.5,
        alpha_state=0.85,
        keep_tree=true,
        check_repeat_state=true,
        rng=rng,
        init_Q=0.0,
        init_N=0,
        next_action=RandomActionGenerator(rng),
        estimate_value=RolloutEstimator(RandomGaussian(rng)),
        #estimate_value=TrackedRolloutEstimator(rng, RandomSolver(rng)),
        track_best_path=true,
        kwargs...
       )
end

struct CMCTSSim
    mdp
    bandit_t
    n_iters::Int
    seed::Int
    metadata::Dict
end
function CMCTSSim(mdp, bandit_t, n_iters, seed)
    CMCTSSim(mdp, bandit_t, n_iters, seed, Dict(:sim_algorithm=>string(bandit_t), :sim_mdp=>string(mdp), :sim_n_ters=>n_iters, :sim_seed=>seed))
end
function POMDPs.simulate(sim::CMCTSSim) 
    rng = MersenneTwister(sim.seed)
    mdp = sim.mdp
    b = sim.bandit_t(mdp)
    solver = ModularSolver(mdp, b, sim.seed; n_iterations=sim.n_iters)
    policy = solve(solver, mdp)
    s0 = initial_state(mdp, rng)
    hr = HistoryRecorder(; rng=rng)
    h = simulate(hr, mdp, policy, s0)
    r_total = sum(h.reward_hist)
    r_total
end

@with_kw struct RewardvsN 
    mdp = Walk1D()
    bandits=[DPWBandit]
    n_seeds=200
    n_iters=[20,50,100,250,500,750]
end
struct RewardvsNResult
    title::String
    data::DataFrame
end

function generate_sim_q(study::RewardvsN)
    q = []
    for b in study.bandits, n in study.n_iters, i in 1:study.n_seeds
        push!(q, CMCTSSim(study.mdp, b, n, i))
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
@recipe function plot(result::RewardvsNResult)
    title := result.title
    xlabel := "number of iterations"
    ylabel := "reward"
    df = aggregate(result.data, [:alg, :n_iters], [mean, std])[[:alg, :n_iters, :r_total_mean, :r_total_std]]
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

end # module
