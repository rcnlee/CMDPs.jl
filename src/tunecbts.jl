struct TuneCBTSSim
    mdp
    seed::Int
    n_iters::Int
    max_steps::Int
    A_max::Float64
    n_proposes::Float64
    log_length_scale::Float64
    log_signal_sigma::Float64
    log_obs_noise::Float64
    ec::Float64
    k_state::Float64
    alpha_state::Float64
    metadata::Dict
end
function TuneCBTSSim()
end
function POMDPs.simulate(sim::TuneCBTSSim)
    rng = MersenneTwister(sim.seed)
    mdp = sim.mdp
    b = CBTSBandit(mdp, 
                     A_max=sim.A_max,
                     n_proposes=sim.n_proposes,
                     exploration_constant=sim.ec,
                     log_length_scale=sim.log_length_scale,
                     log_signal_sigma=sim.log_signal_sigma,
                     log_obs_noise=sim.log_obs_noise,
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

@with_kw struct TuneCBTS
    mdp=Walk1D()
    n_seeds=5
    n_iters=[10,25,50]
    max_steps=20
    A_max=[10.20]
    n_proposes=[10,20]
    log_length_scale=[-1.0,0.0]
    log_signal_sigma=[-1.0,0.0]
    log_obs_noise=[-2.0,-1.0]
    ecs=[0.0,1.0]
    k_states=[0.0,1.0]
    alpha_states=[0.0,0.5]
end
struct TuneCBTSResult
    title::String
    data::DataFrame
end
TuneCBTS(::Type{Walk1D}) = TuneCBTS(Walk1D(), 200, 
                                       [300],  #n_iters
                                       20, #max_steps 
                                       [10,20,30,50], #A_max
                                       [20,50,100], #n_proposes
                                       [-2.0,-1.0,0.0,0.5], #log_length_scale
                                       [0.0], #log_signal_sigma
                                       [-2.0,-1.0,0.0,0.5], #log_obs_noise
                                       [0.25,0.5,0.75,1.0], #ec
                                       [0.1,0.5,1.0,2.0,5.0,10.0,20.0], #k_states
                                       [0.0,0.1,0.25,0.5,0.75,0.9]) #alpha_states

function generate_sim_q(study::TuneCBTS)
    q = []
    for seed in 1:study.n_seeds, n_iters in study.n_iters, A_max in study.A_max, n_proposes in study.n_proposes, log_length_scale in study.log_length_scale, log_signal_sigma in study.log_signal_sigma, log_obs_noise in study.log_obs_noise, ec in study.ecs, k_state in study.k_states, alpha_state in study.alpha_states
        push!(q, TuneCBTSSim(study.mdp, seed, n_iters, study.max_steps, A_max, n_proposes, log_length_scale, log_signal_sigma, log_obs_noise, ec, k_state, alpha_state, Dict()))
    end
    q
end
function run_study(study::TuneCBTS)
    df = DataFrame([String, Int, Int, Int, Int, Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64], [:alg, :seed, :n_iters, :max_steps, :A_max, :n_proposes, :log_length_scale, :log_signal_sigma, :log_obs_noise, :ec, :k_state, :alpha_state, :r_total], 0)
    q = generate_sim_q(study)
    result = pmap(POMDPs.simulate, q)
    for (r, sim) in zip(result, q)
        push!(df, ["CBTSBandit", sim.seed, sim.n_iters, sim.max_steps, sim.A_max, sim.n_proposes, sim.log_length_scale, sim.log_signal_sigma, sim.log_obs_noise, sim.ec, sim.k_state, sim.alpha_state, r])
    end
    TuneCBTSResult(string(study.mdp), df)
end
function report(result::TuneCBTSResult)
    df = aggregate(result.data, [:alg, :max_steps, :A_max, :n_proposes, :log_length_scale, :log_signal_sigma, :log_obs_noise, :ec, :k_state, :alpha_state], [mean, std])[[:alg, :A_max, :n_proposes, :log_length_scale, :log_signal_sigma, :log_obs_noise, :ec, :k_state, :alpha_state, :r_total_mean, :r_total_std]]
    dd = sort(df[:], :r_total_mean, rev=true)
    dd
end
