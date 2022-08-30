# Evaluating-Soft-Actor-Critic-Agent-Performance-of-Equity-Trading-using-FinRL-
The goal of this project was to create a deep-learning trading agent that can be paired with a trading platform to effectively trade equities based on indicators and financial data.
The SAC algorithm determines the best course of action by maximising the entropy of the policy as well as the long-term expected payoff. A measurement of policy uncertainty given the situation is the policy entropy. Further entropy encourages greater exploration of the data. Exploration and exploitation of the environment are balanced by maximising both the reward and the entropy.
# How Soft Actor Critic Algorithm works
To estimate the policy and value function, a SAC agent maintains the following parameters:

Stochastic actor π(A|S;θ) — The actor, with parameters θ, outputs the mean ans standard deviation of conditional Gaussian probability of taking each continuous action A when in state S.

One or two Q-value critics Qk(S,A;ϕk) — The critics, each with parameters ϕk, take observation S and action A as inputs and return the corresponding expectation of the value function, which includes both the long-term reward and entropy.

One or two target critics Qtk(S,A;ϕtk) — To improve the stability of the optimization, the agent periodically sets the target critic parameters ϕtk to the latest corresponding critic parameter values. The number of target critics matches the number of critics

The actor in a SAC agent generates mean and standard deviation outputs. To select an action, the actor first randomly selects an unbounded action from a Gaussian distribution with these parameters. During training, the SAC agent uses the unbounded probability distribution to compute the entropy of the policy for the given observation.

If the action space of the SAC agent is bounded, the actor generates bounded actions by applying tanh and scaling operations to the unbounded action.

Source: https://www.mathworks.com/help/reinforcement-learning/ug/sac-agents.html

