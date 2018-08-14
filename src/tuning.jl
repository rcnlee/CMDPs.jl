struct TuneDPWSim
    mdp
    seed::Int
    n_iters::Int
    max_steps::Int
    k_action::Float64
    alpha_action::Float64
    ec::Float64
    k_state::Float64
    alpha_state::Float64
    metadata::Dict
end
function TuneDPWSim()
end
function POMDPs.simulate(sim::TuneDPWSim)
    rng = MersenneTwister(sim.seed)
    mdp = sim.mdp
    b = DPWBandit(mdp, 
                     k_action=sim.k_action,
                     alpha_action=sim.alpha_action,
                     exploration_constant=sim.ec,
                    )
    solver = ModularSolver(mdp, b, sim.seed; n_iterations=sim.n_iters,
                     k_state=sim.k_state,
                     alpha_state=sim.alpha_state)
    policy = solve(solver, mdp)
    s0 = initial_state(mdp, rng)
    hr = HistoryRecorder(; max_steps=sim.max_steps, rng=rng)
    h = simulate(hr, mdp, policy, s0)
    r_total = sum(h.reward_hist)
    r_total
end

@with_kw struct TuneDPW
    mdp=Walk1D()
    n_seeds=5
    n_iters=[10,25,50]
    max_steps=20
    k_actions=[0.0,1.0]
    alpha_actions=[0.0,1.0]
    ecs=[0.0,1.0]
    k_states=[0.0,1.0]
    alpha_states=[0.0,1.0]
end
struct TuneDPWResult
    title::String
    data::DataFrame
end
TuneDPW(::Type{VDPTagMDP}) = TuneDPW(VDPTagMDP(), 10, 
                                       [500],  #n_iters
                                       10, #max_steps 
                                       [0.1,0.5,1.0,2.0,5.0,10.0], #k_actions
                                       [0.1,0.25,0.5,0.75,0.9], #alpha_actions
                                       [0.25,0.5,0.75,1.0], #ec
                                       [0.1,0.5,1.0,2.0,5.0,10.0], #k_states
                                       [0.1,0.25,0.5,0.75,0.9]) #alpha_states
TuneDPW(::Type{MountainCar}) = TuneDPW(MountainCar(), 10, 
                                       [1000],  #n_iters
                                       20, #max_steps 
                                       [0.1,0.5,1.0,2.0,5.0,10.0], #k_actions
                                       [0.1,0.25,0.5,0.75,0.9], #alpha_actions
                                       [0.25,0.5,0.75,1.0], #ec
                                       [0.1,0.5,1.0,2.0,5.0,10.0], #k_states
                                       [0.1,0.25,0.5,0.75,0.9]) #alpha_states
TuneDPW(::Type{LightDark2D}) = TuneDPW(LightDark2D(), 10, 
                                       [500],  #n_iters
                                       10, #max_steps 
                                       [0.1,0.5,1.0,2.0,5.0,10.0], #k_actions
                                       [0.1,0.25,0.5,0.75,0.9], #alpha_actions
                                       [0.25,0.5,0.75,1.0], #ec
                                       [0.1,0.5,1.0,2.0,5.0,10.0], #k_states
                                       [0.1,0.25,0.5,0.75,0.9]) #alpha_states

function generate_sim_q(study::TuneDPW)
    q = []
    for seed in 1:study.n_seeds, n_iters in study.n_iters, k_action in study.k_actions, alpha_action in study.alpha_actions, ec in study.ecs, k_state in study.k_states, alpha_state in study.alpha_states
        push!(q, TuneDPWSim(study.mdp, seed, n_iters, study.max_steps, k_action, alpha_action, ec, k_state, alpha_state, Dict()))
    end
    q
end
function run_study(study::TuneDPW)
    df = DataFrame([String, Int, Int, Int, Float64, Float64, Float64, Float64, Float64, Float64], [:alg, :seed, :n_iters, :max_steps, :k_action, :alpha_action, :ec, :k_state, :alpha_state, :r_total], 0)
    q = generate_sim_q(study)
    result = pmap(POMDPs.simulate, q)
    for (r, sim) in zip(result, q)
        push!(df, ["DPWBandit", sim.seed, sim.n_iters, sim.max_steps, sim.k_action, sim.alpha_action, sim.ec, sim.k_state, sim.alpha_state, r])
    end
    TuneDPWResult(string(study.mdp), df)
end
function report(result::TuneDPWResult)
    df = aggregate(result.data, [:alg, :max_steps, :k_action, :alpha_action, :ec, :k_state, :alpha_state], [mean, std])[[:alg, :k_action, :alpha_action, :ec, :k_state, :alpha_state, :r_total_mean, :r_total_std]]
    dd = sort(df[:], :r_total_mean, rev=true)
    dd
end
