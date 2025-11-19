---
title:  "RL Course by David Silver Notes"
description: After being excited about RL for more than a year, I should have a concise and satisfying answer to the question, 'What is reinforcement learning?' Here it is gathered briefly.
date: 2018-12-14
author: ["Mohamad H. Danesh"]
showToc: true
disableAnchoredHeadings: false
---

## Lecture 1: Introduction to Reinforcement Learning

-   Planning: rules of the game are given, perfect model inside agent's head, plan ahead to find optimal policy(look ahead search or tree search).

-   In RL environment is unknown, in planning environment is known.

-   Types of RL agents:


	-   Policy based
	    
	-   Value function based
	    
	-   Actor critic(combines policy and value function)


-   Agent's model is a representation of the environment in the agent's head.

-   Agent is our brain, is the algorithm we come up with.

-   To get the maximum expected reward, risk is already included.

-   Value function: goodness/badness of states, so can be used to select between actions.

-   Fully observability: agent sees the environment state.

-   Data are not iid.

-   Reward is delayed.

-   There is only a reward signal, no supervision.


Reinforcement learning is the science of decision making.

## Lecture 2: Markov Decision Process

-   Partial ordering over policies:

    $$\pi' \geq \pi \Longleftarrow v_\pi' (s) \geq v_\pi (s), \forall s $$

-   Action-Value function Q is the same as value function, but also takes action as input:

    $$Q_\pi (s, a) = \mathbb{E} \pi [G_t | S_t = s, A_t = a] = \mathbb{E} \pi [R_{t+1} + \gamma * Q_\pi (S_{t+1}, A_{t+1}) | S_t = s, A_t = a]$$

-   Policy is a distribution over actions given states:
    $$\pi (a | s) = P [ A_t = a | S_t = s]$$

    Policies are stationary:
    $$A_t \approx \pi (\cdot | S_t), \forall t > 0$$


-   The returns are random samples, the value function is an expectation over there random samples.

-   Value function:
    $$v_\pi (s) = \mathbb{E} \pi [G_t | S_t = s] = \mathbb{E} \pi [R_{t+1} + \gamma * v_\pi (S_{t+1}) | S_t = s]$$

    It also can be expressed using matrices:
    $$v_\pi = R_\pi + \gamma * \rho_\pi * v$$

-   Total discounted reward from state $t$:
    $$G_t = R_{t+1} + \gamma * R_{t+2} + \gamma^2 * R_{t+3} ... + \gamma^{T-1} * R_T$$

-   Why having discount factor?


	-   Uncertainty in the future.
	    
	-   Mathematically convenient.
	    
	-   Avoids cycles.
	    
	-   Immediate rewards worth more that delayed rewards.


-   Reward function:
    $$R = \mathbb{E} [R_{t+1} | S_t = s, A_t = a]$$

-   Sample episodes from a MDP: different iterations over different states in each episode.

-   State transition probability:
    $$\rho_{ss'} = P [S_{t+1} = s' | S_t = s, A_t = a]$$

-   Showing an ad on a website is actually a bandit problem.

-   Almost all RL problems can be modeled using MDPs.

## Lecture 3: Planning by Dynamic Programming

-   Value Iteration:


	-   Goal: find optimal policy $\pi$
	    
	-   Start with an arbitrary value function: $v_1$  and apply Bellman backup to get to  $v_2$  and finally $v^*$
	    
	-   Unlike policy iteration, there is no explicit policy.
	    
	-   Intermediate value functions may not correspond to any policy.
	    
	-   Will converge.
	    
	-   Intuition: start with final reward and work backwards.


-   Policy Iteration:


	-   Given a policy $\pi$ (any policy works):
	    

    -   Evaluate the policy:  
                                
        $$V_\pi (s) = \mathbb{E} [ R_{t+1} + \gamma * R_{t+2} + ... | S_t = s ]$$
	    
	-   Improve the policy: $\pi' = greedy ( V_\pi )$
	    
	-   Will converge.


-   Policy Evaluation:


	-   Start with an arbitrary value function: $v_1$  and apply Bellman backup to get to $v_2$  and finally $v_\pi$
	    
	-   Use both synchronous and asynchronous backups.
	    
	-   Will converge.
	    
	-   $V^{k+1} = R^\pi + \gamma * P^\pi * V^k$


-   Input: $MDP(S, A, P, R, \gamma)$ - Output: optimal value function $V^*$ and optimal policy $\pi^*$

-   In planning full knowledge of MDP is given. We want to solve the MPD, i.e. finding a policy.

-   MDPs satisfy both DP properties.

-   Dynamic programming applies to problems that have:


	-   Optimal substructure.
	    
	-   Overlapping subproblems.


-   Dynamic Programming: method for solving complex problems. Breaking them down into subproblems. Solve subproblems and combine solutions.


	-   Dynamic: sequential component to the problem.
	    
	-   Programming: a mapping, e.g. a policy.

## Lecture 4: Model-Free Prediction

-   Bootstrapping: update involves estimated rewards.

-   Monte-Carlo Learning:


	-   In a nutshell, goes all the way through the trajectory and estimates the value by looking at the sample returns.
	    
	-   Useful only for episodic(terminating) settings, and applies once an episode is complete. No bootstrapping.
	    
	-   High variance, zero bias.
	    
	-   Not exploit Markov property. In partially observed environments, MC is a better choice.


-   Temporal-Difference Learning:


	-   Looks one step ahead, and then estimates the return.
	    
	-   Uses bootstrapping to learn, so learns from incomplete episodes. Does online learning.
	    
	-   Low variance, some bias.
	    
	-   Exploits Markov property. Implicitly building MDP structure and solving for that MDP structure(refer to the AB example). More efficient in Markov environments.
	    
	-   $TD(\lambda)$: looks into $\lambda$ steps of the future to update. Also $\lambda$ can be defined as averaging over all n-step returns.

## Lecture 5: Model Free Control

-   On-policy Learning:


	-   Learn about policy $\pi$ from experience sampled from $\pi$
	    
	-   $\epsilon$-greedy Exploration:
	    

		-   With probability = $1 - \epsilon$ choose the greedy action. With probability = $\epsilon$ choose a random action.
		    
		-   Guarantees to have improvements.
	    

	-   Greedy in the Limit of Infinite Exploration(GLIE):
	    

		-   All state-action pairs are explored infinitely many times.
	    
	-   Policy converges to the greedy policy.
	    

	-   SARSA:
	    

		-   $Q(S, A) \leftarrow Q(S, A) + \alpha * (R + \gamma * Q(S', A') - Q(S, A))$
		    
		-   Converges to the optimal action-value function, under some conditions.


-   Off-policy Learning:


	-   Learn about policy $\pi$ from experience sampled from $\mu$
	    
	-   Evaluate target policy $\pi (a | s)$
        to compute $v_\pi (s)$ or $q_\pi (s, a)$ while following the behavior policy 
        $\mu (a | s)$
	    
	-   Can learn from observing human behaviors or other agents.
	    
	-   Learn optimal policy while following exploratory policy.
	    
	-   Monte-carlo learning doesn't work in off-policy learning, because of really high variance. Over many steps, the target policy and the behavior policy never match enough to be useful.
	    
	-   Using TD leaning, it works best. Because policies only need to be similar over one step.
	    
	-   Q-Learning:
	    

		-  $ S'$ is gathered using the behavior policy.
		    
		-   $$Q(S, A) \leftarrow Q(S, A) + \alpha * (R + \gamma * Q(S, A') - Q(S, A))$$
		    
		-   Converges to the optimal action-value function.

## Lecture 6: Value Function Approximation

-   Large scale RL problems because of the huge state spaces become non-tabular.

-   So far, we had $V(s)$ or $Q(s, a)$ But with large scale problems, calculating them takes too much memory and time. Solution: estimate the value function with function approximation:

    $v' (s, w) ≈ v_\pi (s)$ or $q' (s, a, w) ≈ q_\pi (s, a)$

    Generalize from seen states to unseen ones. Update parameter $w$ using MC or TD learning.

-   Incremental Methods:


	-   Table lookup is a special case of linear value function approximation.
	    
	-   Target value function for MC is the return $G_t$ and for TD is the \lambda-return $G^\lambda _t$
	    
	-   Gradient descent is simple and appealing.


-   Batch Methods:


	-   GD is not sample efficient.
	    
	-   Least squares algorithms find parameter vector $w$ minimizing sum squared error.
	    
	-   Example: experience replay in DQN.

## Lecture 7: Policy Gradient Methods

- Policy gradient methods optimize the policy directly, instead of the value function.

- Parametrize the policy:
  $$\pi_\theta (s, a) = P [a | s, \theta]$$

- Point of using policy gradient methods is to being able to scale.

- Nash's equilibrium is the game theoretic notion of optimality.

- Finite Difference Policy Gradient:


	-   For each dimension in $\theta$ parameters, perturbing $\theta$ by small amount $\epsilon$ (look at the formula)
	    
	-   Uses $n$ evaluations to compute policy gradient in $n$ dimension.
	    
	-   Simple, noisy, inefficient.
	    
	-   Works for arbitrary policies.


-   Monte-Carlo Policy Gradient:


	-   Score function is $∇_\theta log \pi_\theta (s, a)$
	    
	-   In continuous action spaces, Gaussian policy should be used.
	    
	-   REINFORCE: update parameters by stochastic gradient ascent.
	    
	-   Has high variance.


-   Actor-Critic Policy Gradient:


	-   Use a critic to estimate the action-value function.
	    
	-   Critic: updates action-value function parameters $w$
	    
	-   Actor: updates policy parameters $\theta$ in direction suggested by critic.
	    
	-   Approximating the policy gradient introduces bias.
	    
	-   Using baseline function $B$ to reduce variance. Value function $V$ could be a good baseline.
	    
	-   Using advantage function to reduce the variance: $Q_\pi (s, a) - V_\pi (s)$
	    
	-   Critic estimates the advantage function.

## Lecture 8: Integrating Learning and Planning

-   So far, the course covered model-free RL.

-   Last lecture: learn policy directly from experience; Previous lectures: learn value function directly from experience; This lecture: learn model directly from experience.

-   Use <b>planning</b> to construct a value function or policy.

-   Model-Based RL:
    -   Plan value function (and/or policy) from model.
    -   Advantages:
        -   Can learn model by supervised learning methods.
        -   Car reason about model uncertainty.
    -   Disadvantages:
        -   First learn a model, then construct a value function: two sources of approximation error.
    -   Model is a parametrized representation of the MDP, i.e. representing state transitions and rewards.
    -   Learning $s, a \rightarrow r$ is a regression problem.
    -   Learning $s, a \rightarrow s'$ is a density estimation problem.
    -   Planning the MDP = Solving the MDP, i.e. figure out what's the best thing to do.
    -   Planning algorithms: value iteration, policy iteration, tree search, ...
-   Integrated Architectures:
    -   Put together the best parts of model-free and model-based RL.
    -   Two sources of experience: real: sampled from environment (true MDP); simulated: sampled from model (approximate MDP).
    -   Dyna:
        -   Learn a model from real experience.
        -   Learn and plan value function from real and simulated experience.
-   Simulation-Based Search:
    -   Forward search: select best action by lookahead. Doesn't explore the entire state space. Builds a search tree with current s as the root. Uses a model of MDP to look ahead. Doesn't solve the whole MDP, just sub-MDP starting from now.
    -   Simulation-based search is a forward search paradigm using sample-based planning.
    -   Simulates episodes of experience from now with the model.
    -   Applies model-free RL to simulated episodes.

## Lecture 9: Exploration and Exploitation

-   Three methods of exploration and exploitation:


	-   Random exploration: e.g. $\epsilon$-greedy
    
	-   Optimism in the face of uncertainty: estimate uncertainty on value, prefer to explore states/actions with highest uncertainty.
    
	-   Information state space: consider agent's information as part of its state, lookahead to see how information helps reward.


-   Types of exploration:


	-   State-action exploration: e.g. pick different action $A$ each time in state $S$
    
	-   Parameter exploration: parameterize policy $\pi (A | S , u)$,
        e.g.  pick different parameters and try for a while.


-   Multi-Armed Bandit:


	-   One-step decision making problem.
	    
	-   No state space, no transition function.
    
	-   $R (r) = P [ R = r | A = a ]$ 
        is an unknown probability distribution over rewards.
	    
	-   Regret is a function of gaps and the counts.
	    
	-   \epsilon-greedy has linear total regret. To resolve this, pick a decay schedule for $\epsilon_1, \epsilon_2$, ... . However, it's not possible to use because $V^*$ is needed to calculate the gaps.
	    
	-   One can transform multi-armed bandit problem into a sequential decision making problem.
	    
	-   Define an MDP over information states.
	    
	-   At each step, information state $S'$ summarizes all information accumulated so far.
	    
	-   MDP can then be solved by RL.


-   Contextual Bandits:


	-   $S = P [ S ]$ is an unknown distribution over states(or "contexts").
	    
	-   $R (r) = P [ R = r | S = s, A = a ]$ 
        is an unknown probability distribution over rewards.


-   MDPs:


	-   For unknown or poorly estimated states, replace reward function with $r_max$ Means to be very optimistic about uncertain states.
	    
	-   Augmented MDP: includes information state so that $S' = (S , I)$

## Lecture 10: Classic Games

-   Nash equilibrium is a joint policy for all players, so a way for others to pick actions such that every single player is playing the best response to all other players, i.e. no player would choose from Nash.

-   Single-Agent and Self-Play Reinforcement Learning: Nash equilibrium is fixed-point of self-play RL. Experience is generated by playing games between agents. Each agent learns best response to other players. One player’s policy determines another player’s environment.

-   Two-Player Zero-Sum Games: A two-player game has two (alternating) players: $R_1 + R_2 = 0$
