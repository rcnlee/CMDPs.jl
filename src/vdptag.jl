
function MCTS.RandomBandit(mdp::VDPTagMDP; kwargs...)
    RandomBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        kwargs...)
end
function MCTS.DPWBandit(mdp::VDPTagMDP; kwargs...)
    DPWBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=0.75,
        k_action=10.0,
        alpha_action=0.5,
        kwargs...)
end
function MCTS.CBTSBandit(mdp::VDPTagMDP; kwargs...)
    CBTSBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=1.0,
        A_max=20,
        n_proposes=100,
        log_length_scale=0.0,
        log_signal_sigma=0.0,
        log_obs_noise=-1.0,
        action_dims=1,
        n_sig=2.0,  #number of standard deviations for GP-UCB
        kwargs...)
end
function MCTS.CBTSDPWBandit(mdp::VDPTagMDP; kwargs...)
    CBTSDPWBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=2.0,
        k_action=10.0,
        alpha_action=0.5,
        n_proposes=500,
        log_length_scale=0.0,
        log_signal_sigma=0.0,
        log_obs_noise=-1.0,
        fit_qs=false,
        action_dims=1,
        n_sig=2.0,  #number of standard deviations for GP-UCB
        kwargs...)
end
function MCTS.ModularSolver(mdp::VDPTagMDP, b::ModularBandit, seed=0; 
                            kwargs...) 
    rng = MersenneTwister(seed)
    ModularSolver(;
        n_iterations=100,
        bandit=b,
        depth=10,
        k_state=2.0,
        alpha_state=0.9,
        keep_tree=true,
        check_repeat_state=true,
        rng=rng,
        init_Q=0.0,
        init_N=0,
        next_action=RandomActionGenerator(rng),
        estimate_value=RolloutEstimator(RandomSolver(rng)),
        track_best_path=true,
        kwargs...
       )
end
Base.string(::VDPTagMDP) = "VDPTag"
