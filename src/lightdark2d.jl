
function MCTS.DPWBandit(mdp::LightDark2D; kwargs...)
    DPWBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=0.5,
        k_action=5.0,
        alpha_action=0.5,
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
        alpha_state=0.5,
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
