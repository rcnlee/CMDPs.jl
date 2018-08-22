
function MCTS.RandomBandit(mdp::LightDark2D; kwargs...)
    RandomBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        kwargs...)
end
function MCTS.DPWBandit(mdp::LightDark2D; kwargs...)
    DPWBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=0.75,
        k_action=10.0,
        alpha_action=0.25,
        kwargs...)
end
function MCTS.CBTSBandit(mdp::LightDark2D; kwargs...)
    CBTSBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=1.0,
        A_max=20,
        n_proposes=100,
        log_length_scale=0.0,
        log_signal_sigma=0.0,
        log_obs_noise=-1.0,
        action_dims=2,
        n_sig=2.0,  #number of standard deviations for GP-UCB
        kwargs...)
end
function MCTS.CBTSDPWBandit(mdp::LightDark2D; kwargs...)
    CBTSDPWBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=1.0,
        k_action=5.0,
        alpha_action=0.5,
        n_proposes=100,
        log_length_scale=0.0,
        log_signal_sigma=0.0,
        log_obs_noise=-1.0,
        fit_qs=false,
        action_dims=2,
        n_sig=2.0,  #number of standard deviations for GP-UCB
        kwargs...)
end
function MCTS.ModularSolver(mdp::LightDark2D, b::ModularBandit, seed=0; 
                            kwargs...) 
    rng = MersenneTwister(seed)
    ModularSolver(;
        n_iterations=100,
        bandit=b,
        depth=10,
        k_state=0.1,
        alpha_state=0.9,
        keep_tree=true,
        check_repeat_state=true,
        rng=rng,
        init_Q=0.0,
        init_N=0,
        next_action=RandomGaussianLD2(rng),
        estimate_value=RolloutEstimator(RandomGaussianLD2(rng)),
        track_best_path=true,
        kwargs...
       )
end
Base.string(::LightDark2D) = "LightDark2D"
