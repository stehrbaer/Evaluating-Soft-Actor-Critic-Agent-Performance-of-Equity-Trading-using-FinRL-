# Evaluating-Soft-Actor-Critic-Agent-Performance-of-Equity-Trading-using-FinRL-
The goal of this project was to create a deep-learning trading agent that can be paired with a trading platform to effectively trade equities based on indicators and financial data.
The SAC algorithm determines the best course of action by maximising the entropy of the policy as well as the long-term expected payoff. A measurement of policy uncertainty given the situation is the policy entropy. Further entropy encourages greater exploration of the data. Exploration and exploitation of the environment are balanced by maximising both the reward and the entropy.

FinRL is open-source platform for financial reinforcement learning, helping users establish virtual environment to train, test and implement deep learning agents. 

# How Soft Actor Critic Algorithm works
To estimate the policy and value function, a SAC agent maintains the following parameters:

Stochastic actor π(A|S;θ) — The actor, with parameters θ, outputs the mean, standard deviation of conditional Gaussian probability of taking each continuous action A in state S.

One or two Q-value critics Qk(S,A;ϕk) — The critics, each with parameters ϕk, take observation S and action A as inputs and return the corresponding expectation of the value function, which output the long-term reward and entropy.

One or two target critics Qtk(S,A;ϕtk) — To further stabilize optimization, the agent periodically sets the target critic parameters ϕtk to the latest corresponding critic parameter values. The number of target critics matches the number of critics

The actor in a SAC agent generates mean and standard deviation outputs. To select an action, the actor first randomly selects an unbounded action from a Gaussian distribution with these parameters. During training, the SAC agent uses the unbounded probability distribution to compute the entropy of the policy for the given observation.

If the action space of the SAC agent is bounded, the actor generates bounded actions by applying tanh and scaling operations to the unbounded action.


<img width="565" alt="Screenshot 2022-08-30 at 19 07 45" src="https://user-images.githubusercontent.com/67784016/187511223-d15fa4cf-b36c-42dd-8136-c1824b5596b8.png">
Source: https://www.mathworks.com/help/reinforcement-learning/ug/sac-agents.html

# Framework for Assessment 

The deep learning network was to be assesed by trading equities over a given time-frame, measuring performance in terms of returns and risk management. Using AI4Finance's library FinRL and ElegantRL, several reinforcement-learning networks were tested, from which the best performing one was chosen to be further evaluated and deployed into a Alpacas paper-trading platform. 

Using FinRL, Yahoo Finance and ElegantRL as the base libraries for the model, the agent was trained using the Dow Jones Inudstrial Index, consisting of 30-selected equities. The Agent's dataset featured financial data such as OLHCV (Opening-Low-High_Closing and Volume) for the selected equities, to which several technical indicators were added to aid the algorithm in risk assessment and implementing policy towards the agent's actions (buy,hold,sell). 

The agent was also given risk measuring parameters such as the Volatility Index (VIX) and Turbulence, which included a risk threshold that programmed the agent to sell assets when volatility within the trading environment exceeded a certain threshold. The agent's environment was aimed to mimic the Alpaca Trading environment, with an initial amount of $100,000 made available for the agent to trade. 




# Assessment of Agent 

Throughout this project, the performance of several agents (TD3, PPO and SAC) was evaluated via performance of trading the DOW JONES Index (30 equities). As can be seen by the table below, the SAC agent performed substantially better than the other two agents, making it a prime candidate for further testing and implementation for live-trading.

<img width="450" alt="Screenshot 2022-08-30 at 19 11 56" src="https://user-images.githubusercontent.com/67784016/187511963-9ba2504c-7c2b-49bf-abff-4e157d6eb589.png">

The Soft Actor Critic agent was then adjusted with a higher learning rate (0.0001 to 0.0003), which resulted in the agent performance improving substantially. 

SAC Agent: Cumulative Returns vs. Dow Jones Benchmark (LR = 0.0001)

<img width="464" alt="Screenshot 2022-08-30 at 19 18 47" src="https://user-images.githubusercontent.com/67784016/187513370-d53e4a80-b575-4e18-ac03-62d2b56f28ac.png">



SAC Agent: Cumulative Returns vs. Dow Jones Benchmark (LR = 0.0003)

<img width="441" alt="Screenshot 2022-08-30 at 19 18 13" src="https://user-images.githubusercontent.com/67784016/187513183-55aff24d-a305-4d0d-8782-53bd2ac9d6ab.png">

# Risk assessment 

Using annual returns versus annual volatility, we can measure the agents performance of risk assessment. By using the Dow Jones Index equities we calculated ratios of expected returns to expected volatilty within the equities, which showed the following results. 

<img width="935" alt="Screenshot 2022-08-30 at 19 59 27" src="https://user-images.githubusercontent.com/67784016/187520383-078e4ae9-c070-4402-b813-cc1607c0053f.png">

Equities within the top left corner and adjacent show higher returns to volatility, making them more risk averse and favorable for investments. Equities that can be found toward the bottom and bottom right corner showcases low returns along with high volatility. 


We would expect the SAC Agent to be implement these risk assessment measures, which can be evaluated by looking at the agent's action history.

<img width="423" alt="Screenshot 2022-08-30 at 20 02 12" src="https://user-images.githubusercontent.com/67784016/187520830-5cfd4233-0863-4323-a583-a11f043b97f9.png">

The following plot compares the agents actions for the top eight equities with the highest return-to-volatility ratio versus the bottom-eight. As can be seen by the graph the agent does favor actions using equities with higher return to risk ratios, highlighting the algorithms ability to categorize risk factors when selecting to buy, sell or hold an asset. 

# Alpaca Performance 

Following the implementation of the agent via Alpaca's paper trading platform using ElegantRL, the algorithm was run for 3-days, using 60-minute time intervals for the data feed. 

<img width="1166" alt="Screenshot 2022-08-30 at 19 32 25" src="https://user-images.githubusercontent.com/67784016/187521598-a772f7ea-d787-498c-a6b6-f9dc226c76b0.png">

The portfolio showed an overall value of $102,800.44, an marginal increase from the starting value of $100,0000. 



Code Sourcing: @article{finrl2020,
    author  = {Liu, Xiao-Yang and Yang, Hongyang and Chen, Qian and Zhang, Runjia and Yang, Liuqing and Xiao, Bowen and Wang, Christina Dan},
    title   = {{FinRL}: A deep reinforcement learning library for automated stock trading in quantitative finance},
    journal = {Deep RL Workshop, NeurIPS 2020},
    year    = {2020}
}
