
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
        k_action=10.0,
        alpha_action=0.25,
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
        alpha_state=0.5,
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
