
function MCTS.RandomBandit(mdp::Walk1DMDP.Walk1D; kwargs...)
    RandomBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        kwargs...)
end
function MCTS.DPWBandit(mdp::Walk1DMDP.Walk1D; kwargs...)
    DPWBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=0.25,
        k_action=2.0,
        alpha_action=0.9,
        kwargs...)
end
function MCTS.ModularSolver(mdp::Walk1DMDP.Walk1D, b::ModularBandit, seed=0;
                            kwargs...) 
    rng = MersenneTwister(seed)
    ModularSolver(;
        n_iterations=100,
        bandit=b,
        depth=mdp.p.t_max,
        k_state=0.1,
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
function MCTS.CBTSBandit(mdp::Walk1DMDP.Walk1D; kwargs...)
    CBTSBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=1.0,
        A_max=10,
        n_proposes=100,
        log_length_scale=0.0,
        log_signal_sigma=0.0,
        log_obs_noise=-1.0,
        action_dims=1,
        n_sig=2.0,  #number of standard deviations for GP-UCB
        kwargs...)
end
function MCTS.ModularSolver(mdp::Walk1DMDP.Walk1D, b::CBTSBandit, seed=0;
                            kwargs...) 
    rng = MersenneTwister(seed)
    ModularSolver(;
        n_iterations=100,
        bandit=b,
        depth=mdp.p.t_max,
        k_state=0.1,
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
Base.string(::Walk1D) = "Walk1D"
