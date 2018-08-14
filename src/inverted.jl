
function MCTS.DPWBandit(mdp::InvertedPendulum; kwargs...)
    DPWBandit(; 
        enable_action_pw=true,
        check_repeat_action=true,
        exploration_constant=1.0,
        k_action=10.0,
        alpha_action=0.5,
        kwargs...)
end
function MCTS.ModularSolver(mdp::InvertedPendulum, b::ModularBandit, seed=0; 
                            kwargs...) 
    rng = MersenneTwister(seed)
    ModularSolver(;
        n_iterations=100,
        bandit=b,
        depth=10,
        k_state=10.0,
        alpha_state=0.5,
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
Base.string(::InvertedPendulum) = "Inverted Pendulum"
