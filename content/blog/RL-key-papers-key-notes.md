---
title:  "Reinforcement Learning Key Papers Keynotes"
description: Keynotes from teh RL Key Papers of Spinning Up by OpenAI.
date: 2019-12-01
author: ["Mohamad H. Danesh"]
showToc: true
disableAnchoredHeadings: false
---

## [A Simple Neural Attentive Meta-Learner](https://openreview.net/forum?id=B1DmUzWAW&noteId=B1DmUzWAW), algorithm: SNAIL


	-   Uses a novel combination of temporal convolutions and soft attention; the former to aggregate information from past experience and the latter to pinpoint specific pieces of information.
	    
	-   Rather than training the learner on a single task (with the goal of generalizing to unseen samples from a similar data distribution) a meta-learner is trained on a distribution of similar tasks, with the goal of learning a strategy that generalizes to related but unseen tasks from a similar task distribution.
	    
	-   Combines temporal convolutions, which enable the meta-learner to aggregate contextual information from past experience, with causal attention, which allow it to pinpoint specific pieces of information within that context.
	    
	-   Soft attention treats the context as an unordered key-value store which it can query based on the content of each element. However, the lack of positional dependence can also be undesirable, especially in reinforcement learning, where the observations, actions, and rewards are intrinsically sequential.
	    
	-   Despite their individual shortcomings, temporal convolutions and attention complement each other: while the former provide high-bandwidth access at the expense of finite context size, the latter provide pinpoint access over an infinitely large context.
	    
	-   By interleaving TC layers with causal attention layers, SNAIL can have high-bandwidth access over its past experience without constraints on the amount of experience it can effectively use.


## [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400), algorithm: MAML


	-   Proposes an algorithm for meta-learning that is model-agnostic, in the sense that it is compatible with any model trained with gradient descent and applicable to a variety of different learning problems, including classification, regression, and reinforcement learning.
	    
	-   Aims to train models that can achieve rapid adaptation, a problem setting that is often formalized as few-shot learning.
	    
	-   Makes no assumption on the form of the model, other than to assume that it is parametrized by some parameter vector $\theta$, and that the loss function is smooth enough in $\theta$ that we can use gradient-based learning techniques.


## [Learning to Reinforcement Learn](https://arxiv.org/abs/1611.05763)


	-   Emerges a system that is trained using one RL algorithm, but whose recurrent dynamics implement a second, quite separate RL procedure.
	    
	-   Here, the tasks that make up the training series are interrelated RL problems, for example, a series of bandit problems varying only in their parameterization. Rather than presenting target outputs as auxiliary inputs, the agent receives inputs indicating the action output on the previous step and, critically, the quantity of reward resulting from that action.
	    
	-   At the start of a new episode, a new MDP task $m \approx D$ and an initial state for this task are sampled, and the internal state of the agent (i.e., the pattern of activation over its recurrent units) is reset. The agent then executes its action-selection strategy in this environment for a certain number of discrete time-steps. At each step $t$ an action $a_t \in A$ is executed as a function of the whole history $H_t = {x_0, a_0, r_0, . . . , x_{t-1}, a_{t-1}, r_{t-1}, x_t}$ of the agent interacting in the MDP $m$ during the current episode. The network weights are trained to maximize the sum of observed rewards over all steps and episodes.
	    
	-   After training, the agent’s policy is fixed (i.e. the weights are frozen, but the activations are changing due to input from the environment and the hidden state of the recurrent layer), and it is evaluated on a set of MDPs that are drawn either from the same distribution $D$ or slight modifications of that distribution (to test the generalization capacity of the agent). The internal state is reset at the beginning of the evaluation of any new episode.
	    
	-   Since the policy learned by the agent is history-dependent (as it makes uses of a recurrent network), when exposed to any new MDP environment, it is able to adapt and deploy a strategy that optimizes rewards for that task.
	    
	-   All reinforcement learning was conducted using the Advantage Actor-Critic algorithm.
	    
	-   reward and last action are additional inputs to the LSTM. For non-bandit environments, observation is also fed into the LSTM either as a one-hot or passed through an encoder model [3-layer encoder: two convolutional layers (first layer: 16 8x8 filters applied with stride 4, second layer: 32 4x4 filters with stride 2) followed by a fully connected layer with 256 units and then a ReLU non-linearity]. For bandit experiments, current time step is also fed in as input.
	    
	-   Deep meta-RL involves a combination of three ingredients: (1) Use of a deep RL algorithm to train a recurrent neural network, (2) a training set that includes a series of interrelated tasks, (3) network input that includes the action selected and reward received in the previous time interval.


## [RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779), algorithm: RL^2


	-   Benefits from their prior knowledge about the world.
	    
	-   The algorithm is encoded in the weights of the RNN, which are learned slowly through a general-purpose (“slow”) RL algorithm. The RNN receives all information a typical RL algorithm would receive, including observations, actions, rewards, and termination flags; and it retains its state across episodes in a given Markov Decision Process (MDP). The activations of the RNN store the state of the “fast” RL algorithm on the current (previously unseen) MDP.
	    
	-   Bayesian reinforcement learning provides a solid framework for incorporating prior knowledge into the learning process.
	    
	-   Views the learning process of the agent itself as an objective, which can be optimized using standard reinforcement learning algorithms.
	    
	-   Since the underlying MDP changes across trials, as long as different strategies are required for different MDPs, the agent must act differently according to its belief over which MDP it is currently in. Hence, the agent is forced to integrate all the information it has received, including past actions, rewards, and termination flags, and adapt its strategy continually. Hence, we have set up an end-to-end optimization process, where the agent is encouraged to learn a “fast” reinforcement learning algorithm.
	    
	-   it receives the tuple $(s, a, r, d)$ as input, which is embedded using a function $\phi(s, a, r, d)$ and provided as input to an RNN. To alleviate the difficulty of training RNNs due to vanishing and exploding gradients, they use Gated Recurrent Units (GRUs) which have been demonstrated to have good empirical performance. The output of the GRU is fed to a fully connected layer followed by a softmax function, which forms the distribution over actions.
	    
	-   Supervised learning vs reinforcement learning: agent must not only learn to exploit existing information, but also learn to explore, a problem that is usually not a factor in supervised learning.
	    
	-   The "fast" RL algorithm is a computation whose state is stored in the RNN activations, and the RNN’s weights are learned by a general-purpose "slow" reinforcement learning algorithm.


## [Thinking Fast and Slow with Deep Learning and Tree Search](https://arxiv.org/abs/1705.08439), algorithm: ExIt


	-   Decomposes the problem into separate planning and generalization tasks.
	    
	-   Planning new policies is performed by tree search, while a deep neural network generalizes those plans. Subsequently, tree search is improved by using the neural network policy to guide search, increasing the strength of new plans.
	    
	-   According to dual-process theory, human reasoning consists of two different kinds of thinking. System 1 is a fast, unconscious and automatic mode of thought, also known as intuition or heuristic process. System 2, an evolutionarily recent process unique to humans, is a slow, conscious, explicit and rule-based mode of reasoning.
	    
	-   In deep RL algorithms such as REINFORCE and DQN, neural networks make action selections with no lookahead; this is analogous to System 1. Unlike human intuition, their training does not benefit from a ‘System 2’ to suggest strong policies. In this paper, they present Expert Iteration (EXIT), which uses a Tree Search as an analogue of System 2; this assists the training of the neural network.
	    
	-   In Imitation Learning (IL), we attempt to solve the MDP by mimicking an expert policy $\pi^*$  that has been provided. Experts can arise from observing humans completing a task, or, in the context of structured prediction, calculated from labelled training data. The policy we learn through this mimicry is referred to as the apprentice policy.
	    
	-   Compared to IL techniques, Expert Iteration (EXIT) is enriched by an expert improvement step. Improving the expert player and then resolving the Imitation Learning problem allows us to exploit the fast convergence properties of Imitation Learning even in contexts where no strong player was originally known, including when learning tabula rasa.


## [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815), algorithm: AlphaZero


	-   AlphaGo Zero algorithm achieved superhuman performance in the game of Go, by representing Go knowledge using deep convolutional neural networks, trained solely by reinforcement learning from games of self-play.
	    
	-   AlphaGo Zero estimates and optimizes the probability of winning, assuming binary win/loss outcomes. AlphaZero instead estimates and optimizes the expected outcome, taking account of draws or potentially other outcomes.
	    
	-   The rules of Go are invariant to rotation and reflection. This fact was exploited in AlphaGo and AlphaGo Zero in two ways. First, training data was augmented by generating 8 symmetries for each position. Second, during MCTS, board positions were transformed using a randomly selected rotation or reflection before being evaluated by the neural network, so that the Monte- Carlo evaluation is averaged over different biases. The rules of chess and shogi are asymmetric, and in general symmetries cannot be assumed. AlphaZero does not augment the training data and does not transform the board position during MCTS.
	    
	-   In AlphaGo Zero, self-play games were generated by the best player from all previous iterations. After each iteration of training, the performance of the new player was measured against the best player; if it won by a margin of 55% then it replaced the best player and self-play games were subsequently generated by this new player. In contrast, AlphaZero simply maintains a single neural network that is updated continually, rather than waiting for an iteration to complete. Self-play games are generated by using the latest parameters for this neural network, omitting the evaluation step and the selection of best player.
	    
	-   AlphaGo Zero tuned the hyper-parameter of its search by Bayesian optimization. In Alp- haZero we reuse the same hyper-parameters for all games without game-specific tuning.
	    
	-   Training proceeded for 700,000 steps (mini-batches of size 4,096) starting from randomly initialized parameters, using 5,000 first-generation TPUs to generate self-play games and 64 second-generation TPUs to train the neural networks.


## [Recurrent World Models Facilitate Policy Evolution](https://arxiv.org/abs/1809.01999), algorithm: World Models


	-   A generative recurrent neural network is quickly trained in an unsupervised manner to model popular reinforcement learning environments through compressed spatio-temporal representations. The world model’s extracted features are fed into compact and simple policies trained by evolution, achieving state of the art results in various environments.
	    
	-   Trains the agent entirely inside of an environment generated by its own internal world model, and transfer this policy back into the actual environment.
	    
	-   Model M will be a large RNN that learns to predict the future given the past in an unsupervised manner. M’s internal representations of memories of past observations and actions are perceived and exploited by another NN called the controller (C) which learns through RL to perform some task without a teacher. A small and simple C limits C’s credit assignment problem to a comparatively small search space, without sacrificing the capacity and expressiveness of the large and complex M.
	    
	-   To overcome the problem of an agent exploiting imperfections of the generated environments, they adjust a temperature parameter of M to control the amount of uncertainty of the generated environments. They train C inside of a noisier and more uncertain version of its generated environment, and demonstrate that this approach helps prevent C from taking advantage of the imperfections of M.
	    
	-   M is only an approximate probabilistic model of the environment, it will occasionally generate trajectories that do not follow the laws governing the actual environment.


## [Model-Based Reinforcement Learning via Meta-Policy Optimization](https://arxiv.org/abs/1809.05214), algorithm: MB-MPO


	-   Using an ensemble of learned dynamic models, MB-MPO meta-learns a policy that can quickly adapt to any model in the ensemble with one policy gradient step.
	    
	-   Learning dynamics models can be done in a sample efficient way since they are trained with standard supervised learning techniques, allowing the use of off-policy data.
	    
	-   Accurate dynamics models can often be far more complex than good policies.
	    
	-   Model-bias: Model-based approaches tend to rely on accurate (learned) dynamics models to solve a task. If the dynamics model is not sufficiently precise, the policy optimization is prone to overfit on the deficiencies of the model, leading to suboptimal behavior or even to catastrophic failures.
	    
	-   Learning an ensemble of dynamics models and framing the policy optimization step as a meta-learning problem. Meta-learning, in the context of RL, aims to learn a policy that adapts fast to new tasks or environments.
	    
	-   Using the models as learned simulators, MB-MPO learns a policy that can be quickly adapted to any of the fitted dynamics models with one gradient step.
	    
	-   MB-MPO learns a robust policy in the regions where the models agree, and an adaptive one where the models yield substantially different predictions.
	    
	-   Current meta-learning algorithms can be classified in three categories. One approach involves training a recurrent or memory-augmented network that ingests a training dataset and outputs the parameters of a learner model. Another set of methods feeds the dataset followed by the test data into a recurrent model that outputs the predictions for the test inputs. The last category embeds the structure of optimization problems into the meta-learning algorithm.
	    
	-   While model-free RL does not explicitly model state transitions, model-based RL methods learn the transition distribution, also known as dynamics model, from the observed transitions.
	    
	-   MB-MPO frames model-based RL as meta-learning a policy on a distribution of dynamic models, advocates to maximize the policy adaptation, instead of robustness, when models disagree.
	    
	-   First, they initialize the models and the policy with different random weights. Then, they proceed to the data collection step. In the first iteration, a uniform random controller is used to collect data from the real-world, which is stored in a buffer D. At subsequent iterations, trajectories from the real-world are collected with the adapted policies $\{\pi_{\theta_1'} , ..., \pi_{\theta_K '} \}$, and then aggregated with the trajectories from previous iterations. The models are trained with the aggregated real-environment samples.
	    
	-   The algorithm proceeds by imagining trajectories from each the ensemble of models $\{f_{\phi_1} , ..., f_{\phi_K} \}$ using the policy $\pi_\theta$  . These trajectories are are used to perform the inner adaptation policy gradient step, yielding the adapted policies $\{\pi_{\theta_1'} , ..., \pi_{\theta_K'} \}$. Finally, they generate imaginary trajectories using the adapted policies $\pi_{\theta_k'}$ and models $f_{\phi_k}$  , and optimize the policy towards the meta-objective.


## [Model-Ensemble Trust-Region Policy Optimization](https://openreview.net/forum?id=SJJinbWRZ&noteId=SJJinbWRZ), algorithm: ME-TRPO


	-   Uses an ensemble of models to maintain the model uncertainty and regularize the learning process to overcome instability in training which is caused by the learned policy that tends to exploit regions where insufficient data is available for the model to be learned.
	    
	-   Shows that the use of likelihood ratio derivatives yields much more stable learning than backpropagation through time.
	    
	-   The standard approach for model-based reinforcement learning alternates between model learning and policy optimization. In the model learning stage, samples are collected from interaction with the environment, and supervised learning is used to fit a dynamics model to the observations. In the policy optimization stage, the learned model is used to search for an improved policy.
	    
	-   During model learning, they differentiate the neural networks by varying their weight initialization and training input sequences. Then, during policy learning, they regularize the policy updates by combining the gradients from the imagined stochastic roll-outs.
	    
	-   Standard model-based techniques require differentiating through the model over many time steps, a procedure known as backpropagation through time (BPTT). It is well-known in the literature that BPTT can lead to exploding and vanishing gradients.
	    
	-   Proposes to use likelihood ratio methods instead of BPTT to estimate the gradient, which only make use of the model as a simulator rather than for direct gradient computation. In particular, they use Trust Region Policy Optimization (TRPO), which imposes a trust region constraint on the policy to further stabilize learning.
	    
	-   The reward function is known but the transition function is unknown.
	    
	-   ME-TRPO combines three modifications to the vanilla approach. First, they fit a set of dynamics models $\{f_{\phi_1} , . . . , f_{\phi_K}\}$ (termed a model ensemble) using the same real world data. These models are trained via standard supervised learning, and they only differ by the initial weights and the order in which mini-batches are sampled. Second, they use Trust Region Policy Optimization (TRPO) to optimize the policy over the model ensemble. Third, they use the model ensemble to monitor the policy’s performance on validation data, and stops the current iteration when the policy stops improving.


## [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675), algorithm: STEVE


	-   By dynamically interpolating between model rollouts of various horizon lengths for each individual example, STEVE ensures that the model is only utilized when doing so does not introduce significant errors.
	    
	-   Interpolates between many different horizon lengths, favoring those whose estimates have lower uncertainty, and thus lower error. To compute the interpolated target, they replace both the model and Q-function with ensembles, approximating the uncertainty of an estimate by computing its variance under samples from the ensemble.
	    
	-   Uses DDPG as the base learning algorithm, but their technique is generally applicable to other methods that use TD objectives.
	    
	-   Complex environments require much smaller rollout horizon H, which limits the effectiveness of the approach.
	    
    -   From a single rollout of $H$ timesteps, they can compute $H+1$ distinct candidate targets by considering rollouts of various horizon lengths: $T^{MVE_0},T^{MVE_1},T^{MVE_2},...,T^{MVE_H}$. Standard TD learning uses $T^{MVE_0}$ as the target, while MVE uses $T^{MVE_H}$  as the target. They propose interpolating all of the candidate targets to produce a target which is better than any individual.


## [Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning](https://arxiv.org/abs/1803.00101), algorithm: MVE


	-   ‌Model-based value expansion controls for uncertainty in the model by only allowing imagination to fixed depth. By enabling wider use of learned dynamics models within a model-free reinforcement learning algorithm, they improve value estimation, which, in turn, reduces the sample complexity of learning.
	    
	-   Model-based (MB) methods can quickly arrive at near-optimal control with learned models under fairly restricted dynamics classes. In settings with nonlinear dynamics, fundamental issues arise with the MB approach: complex dynamics demand high-capacity models, which in turn are prone to overfitting in precisely those low-data regimes where they are most needed.
	    
	-   Reduces sample complexity while supporting complex non-linear dynamics by combining MB and MF learning techniques through disciplined model use for value estimation.
	    
	-   Model-based value expansion (MVE): a hybrid algorithm that uses a dynamics model to simulate the short-term horizon H and Q-learning to estimate the long-term value beyond the simulation horizon. This improves Q-learning by providing higher-quality target values for training.
	    
	-   MVE offers a single, simple, and adjustable notion of model trust (H), and fully utilizes the model to that extent.
	    
	-   MVE also demonstrates that state dynamics prediction enables on-policy imagination via the TD-k trick starting from off-policy data.
	    
	-   To deal with sparse reward signals, it is important to consider exploration with the model, not just refinement of value estimates.
	    
	-   MVE forms TD targets by combining a short term value estimate formed by unrolling the model dynamics and a long term value estimate using the learned $Q^\pi_\theta−$ function.
	    
	-   Replaces the standard Q-learning target with an improved target, computed by rolling the learned model out for $H$ steps.
	    
	-   Relies on task-specific tuning of the rollout horizon $H$.


## [Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning](https://arxiv.org/abs/1708.02596), algorithm: MBMF


	-   Medium-sized neural network models can in fact be combined with model predictive control (MPC) to achieve excellent sample complexity in a model-based reinforcement learning algorithm, producing stable and plausible gaits to accomplish various complex locomotion tasks.
	    
	-   Uses deep neural network dynamics models to initialize a model-free learner, in order to combine the sample efficiency of model-based approaches with the high task-specific performance of model-free methods.
	    
	-   Although such model-based methods are drastically more sample efficient and more flexible than task-specific policies learned with model-free reinforcement learning, their asymptotic performance is usually worse than model-free learners due to model bias. To address this issue, they use their model-based algorithm to initialize a model-free learner.
	    
	-   The learned model-based controller provides good rollouts, which enable supervised initialization of a policy that can then be fine-tuned with model-free algorithms, such as policy gradients.
	    
	-   Section IV - A: how to learn the dynamics function which is $f(s_t, a_t)$ and outputs the next state $s_{t+1}$. This function can be difficult to learn when the states $s_t$  and $s_{t+1}$  are too similar and the action has seemingly little effect on the output; this difficulty becomes more pronounced as the time between states $\Delta t$ becomes smaller and the state differences do not indicate the underlying dynamics well. They overcome this issue by instead learning a dynamics function that predicts the change in state st  over the time step duration of $\Delta t$. Thus, the predicted next state is as follows: $s_{t+1} = s_t + f(s_t , a_t)$. Note that increasing this $\Delta t$ increases the information available from each data point, and can help with not only dynamics learning but also with planning using the learned dynamics model.


## [Imagination-Augmented Agents for Deep Reinforcement Learning](https://arxiv.org/abs/1707.06203), algorithm: I2A


	-   Combines model-free and model-based aspects.
	    
	-   I2As learn to interpret predictions from a learned environment model to construct implicit plans in arbitrary ways, by using the predictions as additional context in deep policy networks.
	    
	-   Model-free approaches usually require large amounts of training data and the resulting policies do not readily generalize to novel tasks in the same environment, as they lack the behavioral flexibility constitutive of general intelligence.
	    
	-   Uses approximate environment models by "learning to interpret" their imperfect predictions.
	    
	-   Allows the agent to benefit from model-based imagination without the pitfalls of conventional model-based planning.
	    
	-   A key issue addressed by I2As is that a learned model cannot be assumed to be perfect; it might sometimes make erroneous or nonsensical predictions. They therefore do not want to rely solely on predicted rewards (or values predicted from predicted states), as is often done in classical planning.
	    
	-   It takes information about present and imagines possible futures, and chooses the one with the highest reward.
	    
	-   Offloads all uncertainty estimation and model use into an implicit neural network training process, inheriting the inefficiency of model-free methods.


## [Relational Recurrent Neural Networks](https://arxiv.org/abs/1806.01822), algorithm: RMC


	-   Proposes that it is fruitful to consider memory interactions along with storage and retrieval.
	    
	-   Hypothesizes that such a bias may allow a model to better understand how memories are related, and hence may give it a better capacity for relational reasoning over time.
	    
	-   RMC uses multi-head dot product attention to allow memories to interact with each other.
	    
	-   Relational reasoning is the process of understanding the ways in which entities are connected and using this understanding to accomplish some higher order goal.
	    
	-   Applies attention between memories at a single time step, and not across all previous representations computed from all previous observations.
	    
	-   Employs multi-head dot product attention (MHDPA), where each memory will attend over all of the other memories, and will update its content based on the attended information.


## [Neural Map: Structured Memory for Deep Reinforcement Learning](https://arxiv.org/abs/1702.08360), algorithm: Neural Map


	-   Neural map is  a memory system with an adaptable write operator that is customized to the sorts of 3D environments that DRL agents typically interact with.
	    
	-   Neural networks that utilized external memories can be distinguished along two main axis: memories with write operators and those without. Writeless external memory systems, often referred to as “Memory Networks” typically fix which memories are stored. For example, at each time step, the memory network would store the past M states seen in an environment. What is learnt by the network is therefore how to access or read from this fixed memory pool, rather than what contents to store within it.
	    
	-   The memory network introduces two main disadvantages. The first disadvantage is that a potentially significant amount of redundant information could be stored. The second disadvantage is that a domain expert must choose what to store in the memory, e.g. for the DRL agent, the expert must set M to a value that is larger than the time horizon of the currently considered task.
	    
	-   On the other hand, external neural memories having write operations are potentially far more efficient, since they can learn to store salient information for unbounded time steps and ignore any other useless information, without explicitly needing any a priori knowledge on what to store.
	    
	-   Neural Map uses an adaptable write operation and so its size and computational cost does not grow with the time horizon of the environment as it does with memory networks. Also, it imposes a particular inductive bias on the write operation so that it is 1) well suited to 3D environments where navigation is a core component of successful behaviours, and 2) uses a sparse write operation that prevents frequent overwriting of memory locations that can occur with NTMs and DNCs.
	    
	-   Combination of REINFORCE with value function baseline is commonly termed the “Actor-Critic” algorithm.


## [Neural Episodic Control](https://arxiv.org/abs/1703.01988), algorithm: NEC


	-   Agent uses a semi-tabular representation of the value function: a buffer of past experience containing slowly changing state representations and rapidly updated estimates of the value function.
	    
	-   Why learning is slow:
	    

		-   Stochastic gradient descent optimisation requires the use of small learning rates.
		    
		-   Environments with a sparse reward signal can be difficult for a neural network to model as there may be very few instances where the reward is non-zero. This can be viewed as a form of class imbalance where low-reward samples outnumber high-reward samples by an unknown number.
		    
		-   Reward signal propagation by value-bootstrapping techniques, such as Q-learning, results in reward information being propagated one step at a time through the history of previous interactions with the environment.
	    

	-   NEC is able to rapidly latch onto highly successful strategies as soon as they are experienced, instead of waiting for many steps of optimisation (e.g., stochastic gradient descent) as is the case with DQN and A3C.
	    
	-   The semi-tabular representation is an append-only memory that binds slow-changing keys to fast updating values and uses a context-based lookup on the keys to retrieve useful values during action selection by the agent.
	    
	-   Values retrieved from the memory can be updated much faster than the rest of the deep neural network.
	    
	-   The architecture does not try to learn when to write to memory, as this can be slow to learn and take a significant amount of time. Instead, they elect to write all experiences to the memory, and allow it to grow very large compared to existing memory architectures (in contrast to where the memory is wiped at the end of each episode).
	    
	-   Differentiable Neural Dictionary(DND): For each action $a \in A$, NEC has a simple memory module $Ma = (Ka, Va)$, where $Ka$  and $Va$  are dynamically sized arrays of vectors, each containing the same number of vectors. The memory module acts as an arbitrary association from keys to corresponding values, much like the dictionary data type found in programs.
	    
    -   The pixel state s is processed by a convolutional neural network to produce a key h. The key h is then used to lookup a value from the DND, yielding weights wi  in the process for each element of the memory arrays. Finally, the output is a weighted sum of the values in the DND. The values in the DND, in the case of an NEC agent, are the Q values corresponding to the state that originally resulted in the corresponding key-value pair to be written to the memory. Thus this architecture produces an estimate of $Q(s, a)$ for a single given action a.


## [Model-Free Episodic Control](https://arxiv.org/abs/1606.04460), algorithm: MFEC


	-   Addresses the question of how to emulate such fast learning abilities in a machine—without any domain-specific prior knowledge.
	    
	-   QEC (s, a): Each entry contains the highest return ever obtained by taking action a from state s. It estimates the highest potential return for a given state and action, based upon the states, rewards and actions seen.
	    
    -   Tabular RL methods suffer from two key deficiencies: firstly, for large problems they consume a large amount of memory, and secondly, they lack a way to generalise across similar states. To address the first problem, they limit the size of the table by removing the least recently updated entry once a maximum size has been reached. Such forgetting of older, less frequently accessed memories also occurs in the brain. In large scale RL problems (such as real life) novel states are common; the real world, in general, also has this property. They address the problem of what to do in novel states and how to generalise values across common experiences by taking QEC  to be a non-parametric nearest-neighbours model. For states that have never been visited, QEC  is approximated by averaging the value of the k nearest states.


## [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296), algorithm: HIRO


	-   To address efficiency, proposes to use off-policy experience for both higher- and lower-level training. This allows HIRO to take advantage of recent advances in off-policy model-free RL to learn both higher- and lower-level policies using substantially fewer environment interactions than on-policy algorithms.
	    
	-   HIRO: a hierarchical two-layer structure, with a lower-level policy $\mu_{lo}$  and a higher-level policy $\mu_{hi}$.
	    
	-   The higher-level policy operates at a coarser layer of abstraction and sets goals to the lower-level policy, which correspond directly to states that the lower-level policy attempts to reach.
	    
	-   At step t, the higher-level policy produces a goal gt, indicating its desire for the lower-level agent to take actions that yield it an observation $st+c$  that is close to $st + gt$ .
	    
	-   HRL methods to be applicable to real-world settings, they must be sample-efficient, and off-policy algorithms (often based on some variant of Q-function learning) generally exhibit substantially better sample efficiency than on-policy actor-critic or policy gradient variants.


## [FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161), algorithm: Feudal Networks


	-   Employs a Manager module and a Worker module. The Manager operates at a lower temporal resolution and sets abstract goals which are conveyed to and enacted by the Worker. The Worker generates primitive actions at every tick of the environment.
	    
	-   key contributions:
	    

		-   A consistent, end-to-end differentiable model that embodies and generalizes the principles of feudal reinforcement learning(FRL).
		    
		-   A novel, approximate transition policy gradient update for training the Manager, which exploits the semantic meaning of the goals it produces.
		    
		-   The use of goals that are directional rather than absolute in nature.
		    
		-   A novel RNN design for the Manager – a dilated LSTM – which extends the longevity of the recurrent state memories and allows gradients to flow through large hops in time, enabling effective back-propagation through hundreds of steps.
	    

	-   The top level produces a meaningful and explicit goal for the bottom level to achieve. Sub-goals emerge as directions in the latent state- space and are naturally diverse.


## [Strategic Attentive Writer for Learning Macro-Actions](https://arxiv.org/abs/1606.04695), algorithm: STRAW


	-   Proposes a new deep recurrent neural network architecture, dubbed STRategic Attentive Writer (STRAW), that is capable of learning macro-actions in a reinforcement learning setting.
	    
	-   Macro-actions enable both structured exploration and economic computation.
	    
	-   STRAW maintains a multi-step action plan. STRAW periodically updates the plan based on observations and commits to the plan between the replanning decision points.
	    
	-   One observation can generate a whole sequence of outputs if it is informative enough.
	    
	-   Facilitates structured exploration in reinforcement learning – as the network learns meaningful action patterns it can use them to make longer exploratory steps in the state space.
	    
	-   Since the model does not need to process observations while it is committed its action plan, it learns to allocate computation to key moments thereby freeing up resources when the plan is being followed.
	    
	-   Macro-action is a particular, simpler instance of options, where the action sequence (or a distribution over them) is decided at the time the macro-action is initiated.
	    
	-   STRAW learns macro-actions and a policy over them in an end-to-end fashion from only the environment’s reward signal and without resorting to explicit pseudo-rewards or hand-crafted subgoals.
	    
    -   STRAW is a deep recurrent neural network with two modules. The first module translates environment observations into an action-plan – a state variable which represents an explicit stochastic plan of future actions. STRAW generates macro-actions by committing to the action-plan and following it without updating for a number of steps. The second module maintains commitment-plan – a state variable that determines at which step the network terminates a macro-action and updates the action-plan.


## [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495), algorithm: Hindsight Experience Replay (HER)


	-   The necessity of cost engineering limits the applicability of RL in the real world because it requires both RL expertise and domain-specific knowledge.
	    
	-   The approach is based on training universal policies which takes as input not only the current state, but also a goal state.
	    
	-   The pivotal idea behind HER is to replay each episode with a different goal than the one the agent was trying to achieve.
	    
	-   Shows that training an agent to perform multiple tasks can be easier than training it to perform only one task.
	    
	-   HER: after experiencing some episode $s_0, s_1, ..., s_T$ they store in the replay buffer every transition $s_t \rightarrow s_{t + 1}$ not only with the original goal used for this episode but also with a subset of other goals.
	    
    -   Not only does HER learn with extremely sparse rewards, in their experiments it also performs better with sparse rewards than with shaped ones. These results are indicative of the practical challenges with reward shaping, and that shaped rewards would often constitute a compromise on the metric they truly care about (such as binary success/failure).


## [Learning an Embedding Space for Transferable Robot Skills](https://openreview.net/forum?id=rk07ZXZRb&noteId=rk07ZXZRb)


	-   Allows for discovery of multiple solutions and is capable of learning the minimum number of distinct skills that are necessary to solve a given set of tasks.
	    
	-   Recent RL advances learn solutions from scratch for every task. Not only this is inefficient and constrains the difficulty of the tasks that can be solved, but also it limits the versatility and adaptivity of the systems that can be built.
	    
	-   The main contribution of is an entropy-regularized policy gradient formulation for hierarchical policies, and an associated, data-efficient and robust off-policy gradient algorithm based on stochastic value gradients.
	    
	-   Desires an embedding space, in which solutions to different, potentially orthogonal, tasks can be represented.
	    
	-   Aims to learn a skill embedding space, in which different embedding vectors that are “close” to each other in the embedding space correspond to distinct solutions to the same task.
	    
	-   Given the state and action trace of an executed skill, it should be possible to identify the embedding vector that gave rise to the solution: i.e. derive a new skill by re-combining a diversified library of existing skills.
	    
	-   For the policy to learn a diverse set of skills instead of just T separate solutions (one per task), they endow it with a task-conditional latent variable z. With this latent variable, which they also refer to as “skill embedding”, the policy is able to represent a distribution over skills for each task and to share these across tasks.


## [Mutual Alignment Transfer Learning](https://arxiv.org/abs/1707.07907), algorithm: MATL


	-   Harnesses auxiliary rewards to guide the exploration for the real world agent based on the proficiency of the agent in simulation and vice versa.
	    
	-   Real world applications of RL present a significant challenge to the reinforcement learning paradigm as it is constrained to learn from comparatively expensive and slow task executions.
	    
	-   As policies trained via reinforcement learning will learn to exploit the specific characteristics of a system – optimizing for mastery instead of generality – a policy can overfit to the simulation.
	    
	-   Guides the exploration for both systems towards mutually aligned state distributions via auxiliary rewards.
	    
	-   Employs an adversarial approach to train policies with additional rewards based on confusing a discriminator with respect to the originating system for state sequences visited by the agents. By guiding the target agent on the robot towards states that the potentially more proficient source agent visits in simulation, they can accelerate training.
	    
	-   Also, the agent in simulation will be driven to explore better trajectories from states visited by the real-world policy.


## [PathNet: Evolution Channels Gradient Descent in Super Neural Networks](https://arxiv.org/abs/1701.08734), algorithm: PathNet


	-   It is a neural network algorithm that uses agents embedded in the neural network whose task is to discover which parts of the network to re-use for new tasks.
	    
	-   Agents are pathways (views) through the network which determine the subset of parameters that are used and updated by the forwards and backwards passes of the backpropagation algorithm.
	    
	-   Fixes the parameters along a path learned on task A and re-evolving a new population of paths for task B, allows task B to be learned faster than it could be learned from scratch or after fine-tuning.
	    
	-   Uses genetic algorithms to select a population of pathways through the neural network for replication and mutation.
	    
	-   PathNet also significantly improves the robustness to hyperparameter choices of a parallel asynchronous reinforcement learning algorithm (A3C).


## [The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously](https://arxiv.org/abs/1707.03300), algorithm: IU Agent


	-   Learns to solve many tasks simultaneously and faster than agents that target a single task at-a-time comparing with DDPG.
	    
	-   The architecture enables the agent to attend to one task on-policy, while unintentionally learning to solve many other tasks off-policy. Due to the fact that multiple policies are being learned at once they must necessarily be learning off-policy.
	    
	-   Consists of two neural networks. The actor neural network has multiple-heads representing different policies with shared lower-level representations. The critic network represents several state-action value functions, sharing a common representation for the observations.
	    
	-   Refers to the task whose behavior the agent follows during training as the intentional task, and to the remaining tasks as unintentional.
	    
	-   The experiments demonstrate that when acting according to the policy associated with one of the hardest tasks, they are able to learn all other tasks off-policy.


## [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397), algorithm: UNREAL


	-   Hypothesis is that an agent that can flexibly control its future experiences will also be able to achieve any goal with which it is presented.
	    
	-   Predicts and controls features of the sensorimotor stream, by treating them as pseudo- rewards for reinforcement learning.
	    
	-   Uses reinforcement learning to approximate both the optimal policy and optimal value function for many different pseudo-rewards.
	    
	-   Both the auxiliary control and auxiliary prediction tasks share the convolutional neural network and LSTM that the base agent uses to act.
	    
	-   The auxiliary control tasks are defined as additional pseudo-reward functions in the environment the agent is interacting with.
	    
	-   Changes in the perceptual stream often correspond to important events in an environment. They train agents that learn a separate policy for maximally changing the pixels in each cell of an n × n non-overlapping grid placed over the input image. They refer to these auxiliary tasks as pixel control.
	    
	-   The policy or value networks of an agent learn to extract task- relevant high-level features of the environment, they can be useful quantities for the agent to learn to control. Hence, the activation of any hidden unit of the agent’s neural network can itself be an auxiliary reward. They train agents that learn a separate policy for maximally activating each of the units in a specific hidden layer. They refer to these tasks as feature control.
	    
	-   Reward prediction auxiliary task: predicting the onset of immediate reward given some historical context.
	    
	-   The auxiliary tasks are trained on very recent sequences of experience that are stored and randomly sampled; these sequences may be prioritised(in this case according to immediate rewards).
	    
	-   The auxiliary control tasks (pixel changes and simple network features) are shown to enable the A3C agent to learn to achieve the scalar reward faster in domains where the action-space is discrete.


## [Universal Value Function Approximators](http://proceedings.mlr.press/v37/schaul15.pdf), algorithm: UVFA


	-   Introduces universal value function approximators (UVFAs) $V (s, g; \theta)$ that generalise not just over states s but also over goals g.
	    
	-   The goal space often contains just as much structure as the state space.
	    
	-   By universal, they mean that the value function can generalise to any goal g in a set G of possible goals.
	    
	-   UVFAs can exploit two kinds of structure between goals: similarity encoded a priori in the goal representations g, and the structure in the induced value functions discovered bottom-up.
	    
	-   The complexity of UVFA learning does not depend on the number of demons but on the inherent domain complexity.
	    
	-   Introduces a novel factorization approach that decomposes the regression into two stages. They view the data as a sparse table of values that contains one row for each observed state $s$ and one column for each observed goal $g$, and find a low-rank factorization of the table into state embeddings $\phi(s)$ and goal embeddings $\psi(g)$.
	    
	-   Provides two algorithms for learning UVFAs directly from rewards. The first algorithm maintains a finite Horde of general value functions $V_g(s)$, and uses these values to seed the table and hence learn a UVFA $V (s, g; \theta)$ that generalizes to previously unseen goals. The second algorithm bootstraps directly from the value of the UVFA at successor states.


## [Progressive Neural Networks](https://arxiv.org/abs/1606.04671), algorithm: Progressive Networks


	-   Progressive networks: immune to forgetting and can leverage prior knowledge via lateral connections to previously learned features.
	    
	-   Progressive networks retain a pool of pretrained models throughout training, and learn lateral connections from these to extract useful features for the new task.
	    
	-   Solves K independent tasks at the end of training.
	    
	-   Accelerates learning via transfer when possible.
	    
	-   Avoids catastrophic forgetting.
	    
	-   Makes no assumptions about the relationship between tasks, which may in practice be orthogonal or even adversarial.
	    
	-   Each column is trained to solve a particular Markov Decision Process (MDP), the k-th column thus defines a policy 
        $\pi(k)(a | s)$ 
        taking as input a state s given by the environment, and generating probabilities over actions.
	    
	-   A downside of the approach is the growth in number of parameters with the number of tasks.
	    
	-   Studies in detail which features and at which depth transfer actually occurs. They explored two related methods: an intuitive, but slow method based on a perturbation analysis (APS), and a faster analytical method derived from the Fisher Information (AFS).
	    
	-   APS: To evaluate the degree to which source columns contribute to the target task, they inject Gaussian noise at isolated points in the architecture (e.g. a given layer of a single column) and measure the impact of this perturbation on performance.
	    
	-   AFS: By using the Fisher Information matrix, they get a local approximation to the perturbation sensitivity.


## [Variational Option Discovery Algorithms](https://arxiv.org/abs/1807.10299), algorithm: VALOR


	-   Highlights a tight connection between variational option discovery methods and variational autoencoders, and introduces Variational Autoencoding Learning of Options by Reinforcement (VALOR), a new method derived from the connection.
	    
	-   In VALOR, the policy encodes contexts from a noise distribution into trajectories, and the decoder recovers the contexts from the complete trajectories.
	    
	-   Proposes a curriculum learning approach where the number of contexts seen by the agent increases whenever the agent’s performance is strong enough (as measured by the decoder) on the current set of contexts.
	    
	-   Shows that Variational Intrinsic Control (VIC) and the Diversity is All You Need (DIAYN) are specific instances of this template which decode from states instead of complete trajectories.
	    
	-   VALOR can attain qualitatively different behavior of VIC and DIAYN because of its trajectory-centric approach, and DIAYN learns more quickly because of its denser reward signal.
	    
	-   Learns a policy \pi where action distributions are conditioned on both the current state $s_t$ and a context $c$ which is sampled at the start of an episode and kept fixed throughout. The context should uniquely specify a particular mode of behavior (also called a skill). But instead of using reward functions to ground contexts to trajectories, they want the meaning of a context to be arbitrarily assigned (‘discovered’) during training.
	    
	-   VALOR is a variational option discovery method with two key decisions about the decoder:
	    

		-   The decoder never sees actions. Their conception of "interesting" behaviors requires that the agent attempt to interact with the environment to achieve some change in state. If the decoder was permitted to see raw actions, the agent could signal the context directly through its actions and ignore the environment. Limiting the decoder in this way forces the agent to manipulate the environment to communicate with the decoder.
		    
		-   Unlike in DIAYN, the decoder does not decompose as a sum of per-timestep computations.
	    

	-   VALOR has a recurrent architecture for the decoder, using a bidirectional LSTM to make sure that both the beginning and end of a trajectory are equally important.
	    
	-   Starts training with small K (where learning is easy), and gradually increase it over time as the decoder gets stronger.


## [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070), algorithm: DIAYN


	-   Learns useful skills without a reward function, by maximizing an information theoretic objective using a maximum entropy policy.
	    
	-   Unsupervised discovery of skills can serve as an effective pre-training mechanism for overcoming challenges of exploration and data efficiency in reinforcement learning.
	    
	-   A skill is a latent-conditioned policy that alters that state of the environment in a consistent way.
	    
	-   Uses discriminability between skills as an objective.
	    
	-   Learns skills that not only are distinguishable, but also are as diverse as possible.
	    
	-   Proposes a simple method for using learned skills for hierarchical RL and find this methods solves challenging tasks.
	    
	-   Because skills are learned without a priori knowledge of the task, the learned skills can be used for many different tasks.
	    
	-   First, for skills to be useful, they want the skill to dictate the states that the agent visits. Different skills should visit different states, and hence be distinguishable. Second, they want to use states, not actions, to distinguish skills, because actions that do not affect the environment are not visible to an outside observer. For example, an outside observer cannot tell how much force a robotic arm applies when grasping a cup if the cup does not move. Finally, they encourage exploration and incentivize the skills to be as diverse as possible by learning skills that act as randomly as possible.
	    
	-   Performs option discovery by optimizing a variational lower bound for an objective function designed to maximize mutual information between context and every state in a trajectory, while minimizing mutual information between actions and contexts conditioned on states, and maximizing entropy of the mixture policy over contexts.


## [Variational Intrinsic Control](https://arxiv.org/abs/1611.07507), algorithm: VIC


	-   Introduces two policy gradient based algorithms, one that creates an explicit embedding space of options and one that represents options implicitly.
	    
	-   Addresses the question of what intrinsic options are available to an agent in a given state?
	    
	-   The objective of empowerment: long-term goal of the agent is to get to a state with a maximal set of available intrinsic options.
	    
	-   The primary goal of empowerment is not to understand or predict the observations but to control the environment.
	    
	-   Learns to represent the intrinsic control space of an agent.
	    
	-   Data likelihood and empowerment are both information measures: likelihood measures the amount of information needed to describe data and empowerment measures the mutual information between action choices and final states.
	    
	-   VIC is an option discovery technique based on optimizing a variational lower bound on the mutual information between the context and the final state in a trajectory, conditioned on the initial state.


## [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894), algorithm: RND


	-   Introduces exploration bonus that is the error of a neural network predicting features of the observations given by a fixed randomly initialized neural network .
	    
	-   Flexibly combines intrinsic and extrinsic rewards.
	    
	-   Can be used with any policy optimization algorithm.
	    
	-   Efficient to compute as it requires only a single forward pass of a neural network on a batch of experience.
	    
	-   Predicts the output of a fixed randomly initialized neural network on the current observation.
	    
	-   Introduces a modification of Proximal Policy Optimization (PPO) that uses two value heads for the two reward streams, to combine the exploration bonus with the extrinsic rewards. This allows the use of different discount rates for the different rewards, and combining episodic and non-episodic returns.
	    
	-   In a tabular setting with a finite number of states one can define it  to be a decreasing function of the visitation count $nt(s)$ of the state s. In non-tabular cases it is not straightforward to produce counts, as most states will be visited at most once. One possible generalization of counts to non-tabular settings is pseudo-counts (Bellemare et al., 2016) which uses changes in state density estimates as an exploration bonus.
	    
	-   RND involves two neural networks: a fixed and randomly initialized target network which sets the prediction problem, and a predictor network trained on data collected by the agent.
	    
	-   Prediction errors can be attributed to a number of factors:
	    

		-   Amount of training data.
		    
		-   Stochasticity.
		    
		-   Model misspecification.
		    
		-   Learning dynamics.
	    

	-   The distillation error could be seen as a quantification of uncertainty in predicting the constant zero function.
	    
	-   In order to keep the rewards on a consistent scale they normalized the intrinsic reward by dividing it by a running estimate of the standard deviations of the intrinsic returns.


## [Large-Scale Study of Curiosity-Driven Learning](https://arxiv.org/abs/1808.04355)


	-   Curiosity is a type of intrinsic reward function which uses prediction error as reward signal.
	    
	-   Examples of intrinsic reward include “curiosity” which uses prediction error as reward signal, and “visitation counts” which discourages the agent from revisiting the same states. The idea is that these intrinsic rewards will bridge the gaps between sparse extrinsic rewards.
	    
	-   Studies agents driven purely by intrinsic rewards.
	    
	-   Shows that encoding observations via a random network turns out to be a simple, yet effective technique for modeling curiosity across many popular RL benchmarks. This might suggest that many popular RL video game test-beds are not as visually sophisticated as commonly thought.
	    
	-   If the agent itself is the source of stochasticity in the environment, it can reward itself without making any actual progress.
	    
	-   One important point is that the use of an end of episode signal, sometimes called a ‘done’, can often leak information about the true reward function. If they don’t remove the ‘done’ signal, many of the Atari games become too simple. For example, a simple strategy of giving +1 artificial reward at every time-step when the agent is alive and 0 on death is sufficient to obtain a high score in some games.


## [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363), algorithm: Intrinsic Curiosity Module (ICM)


	-   Formulates curiosity as the error in an agent’s ability to predict the consequence of its own actions in a visual feature space learned by a self-supervised inverse dynamics model.
	    
	-   Motivation/curiosity have been used to explain the need to explore the environment and discover novel states.
	    
	-   Measuring “novelty” requires a statistical model of the distribution of the environmental states, whereas measuring prediction error/uncertainty requires building a model of environmental dynamics that predicts the next state $s_{t+1}$ given the current state $s_t$ and the action $a_t$ executed at time t.
	    
	-   Predicts those changes in the environment that could possibly be due to the actions of their agent or affect the agent, and ignore the rest.
	    
	-   Curiosity helps an agent explore its environment in the quest for new knowledge. Further, curiosity is a mechanism for an agent to learn skills that might be helpful in future scenarios.
	    
	-   Agent is composed of two subsystems: a reward generator that outputs a curiosity-driven intrinsic reward signal and a policy that outputs a sequence of actions to maximize that reward signal.
	    
	-   Making predictions in the raw sensory space (e.g. when st  corresponds to images) is undesirable not only because it is hard to predict pixels directly, but also because it is unclear if predicting pixels is even the right objective to optimize.


## [EX2: Exploration with Exemplar Models for Deep Reinforcement Learning](https://arxiv.org/abs/1703.01260), algorithm: EX2


	-   When the reward signals are rare and sparse, function approximation methods such as deep RL struggle to acquire meaningful policies.
	    
	-   Intuitively, novel states are easier to distinguish against other states seen during training.
	    
	-   Estimate novelty by considering how easy it is for a discriminatively trained classifier to distinguish a given state from other states seen previously.
	    
	-   Trains exemplar models for each state that distinguish that state from all other observed states.
	    
	-   Given a dataset $X = \{x_1, ...x_n\}$, an exemplar model consists of a set of $n$ classifiers or discriminators $\{D_{x_1} , ....D_{x_n}\}$, one for each data point. Each individual discriminator $D_{x_i}$ is trained to distinguish a single positive data point xi, the “exemplar,” from the other points in the dataset X.
	    
	-   In GANs, the generator plays an adversarial game with the discriminator by attempting to produce indistinguishable samples in order to fool the discriminator. However, in this work, the generator is rewarded for helping the discriminator rather than fooling it, so their algorithm plays a cooperative game instead of an adversarial one.


## [#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning](https://arxiv.org/abs/1611.04717), algorithm: Hash-based Counts


	-   It is generally thought that count-based methods cannot be applied in high-dimensional state spaces, since most states will only occur once.
	    
	-   States are mapped to hash codes, which allows to count their occurrences with a hash table. These counts are then used to compute a reward bonus according to the classic count-based exploration theory.
	    
	-   Important aspects of a good hash function are, first, having appropriate granularity, and second, encoding information relevant to solving the MDP.
	    
	-   The sample complexity can grow exponentially(with state space size) in tasks with sparse rewards.
	    
	-   Discretizes the state space with a hash function and apply a bonus based on the state-visitation count.
	    
	-   The agent is trained with rewards $(r + r+)$, while performance is evaluated as the sum of rewards without bonuses.


## [Count-Based Exploration with Neural Density Models](https://arxiv.org/abs/1703.01310), algorithm: PixelCNN-based Pseudocounts


	-   Considers two questions left open by CTS-based Pseudocounts work: First, how important is the quality of the density model for exploration? Second, what role does the Monte Carlo update play in exploration?
	    
	-   Answers the first question by demonstrating the use of PixelCNN, an advanced neural density model for images, to supply a pseudo-count.
	    
	-   The mixed Monte Carlo update is a powerful facilitator of exploration in the sparsest of settings.
	    
	-   Trains the density model completely online on the sequence of experienced states.


## [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868), algorithm: CTS-based Pseudocounts


	-   In a tabular setting, agent’s uncertainty over the environment’s reward and transition functions can be quantified using confidence intervals derived from Chernoff bounds, or inferred from a posterior over the environment parameters.
	    
	-   Count-based exploration methods directly use visit counts to guide an agent’s behaviour towards reducing uncertainty.
	    
	-   In spite of their pleasant theoretical guarantees, count-based methods have not played a role in the contemporary successes of reinforcement learning. The issue is that visit counts are not directly useful in large domains, where states are rarely visited more than once.
	    
	-   Pseudo-count estimates the uncertainty of an agent’s knowledge.
	    
	-   Derives the pseudo-count from a density model over the state space to generalize count-based exploration to non-tabular reinforcement learning.
	    
	-   The model should be learning-positive, i.e. the probability assigned to a state x should increase with training.
	    
	-   It should be trained on- line, using each sample exactly once.
		    
    -   The effective model step-size should decay at a rate of $n^{−1}$.


## [VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674), algorithm: VIME


	-   Maximizes information gain about the agent’s belief of environment dynamics.
	    
	-   Modifies the MDP reward function.
	    
	-   Agents are encouraged to take actions that result in states they deem surprising, i.e., states that cause large updates to the dynamics model distribution.
	    
	-   Variational inference is used to approximate the posterior distribution of a Bayesian neural network that represents the environment dynamics.
	    
	-   Using information gain in this learned dynamics model as intrinsic rewards allows the agent to optimize for both external reward and intrinsic surprise simultaneously.


## [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864), algorithm: ES


	-   Needs to communicate scalars, making it possible to scale to over a thousand parallel workers.
	    
	-   It is invariant to action frequency and delayed rewards.
	    
	-   Tolerant of extremely long horizons.
	    
	-   Does not need temporal discounting or value function approximation.
	    
	-   Uses of virtual batch normalization and other reparameterizations of the neural network policy greatly improve the reliability of evolution strategies.
	    
	-   The data efficiency of evolution strategies was surprisingly good.
	    
	-   Exhibits better exploration behaviour than policy gradient methods like TRPO.
	    
	-   Black-box optimization methods have several highly attractive properties: indifference to the distribution of rewards (sparse or dense), no need for backpropagating gradients, and tolerance of potentially arbitrarily long time horizons.


## [Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/abs/1704.06440)


	-   Q-learning methods are not well-understood why they work, since empirically, the Q-values they estimate are very inaccurate.
	    
	-   “Soft” (entropy-regularized) Q-learning is exactly equivalent to a policy gradient method.
	    
	-   In both cases, if the return following an action at is high, then that action is reinforced: in policy gradient methods, the probability 
        $\pi (a_t |s_t)$ 
        is increased; whereas in Q-learning methods, the Q-value $Q (s_t, a_t)$ is increased.
	    
	-   Problem setting described in the paper is the bandit problem.


## [Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning](http://papers.nips.cc/paper/6974-interpolated-policy-gradient-merging-on-policy-and-off-policy-gradient-estimation-for-deep-reinforcement-learning), algorithm: IPG


	-   Examines approaches to merging on- and off-policy updates for deep reinforcement learning.
	    
	-   On-policy learning: One of the simplest ways to learn a neural network policy is to collect a batch of behavior wherein the policy is used to act in the world, and then compute and apply a policy gradient update from this data. This is referred to as on-policy learning because all of the updates are made using data that was collected from the trajectory distribution induced by the current policy of the agent.
	    

		-   Drawbacks: Data inefficient, because they only look at each data point once.
	    


	-   Off-policy learning: Such methods reuse samples by storing them in a memory replay buffer and train a value function or Q-function with off-policy updates.
	    

		-   Drawbacks: This improves data efficiency, but often at a cost in stability and ease of use.
	    


	-   Mixes likelihood ratio gradient with $\hat{Q}$, which provides unbiased but high-variance gradient estimation, and deterministic gradient through an off-policy fitted critic $Q_w$, which provides low-variance but biased gradients.


## [The Reactor: A Fast and Sample-Efficient Actor-Critic Agent for Reinforcement Learning](https://arxiv.org/abs/1704.04651), algorithm: Reactor


	-   First contribution is a new policy evaluation algorithm called Distributional Retrace, which brings multi-step off-policy updates to the distributional reinforcement learning setting.
	    
	-   Introduces the β-leave-one-out policy gradient algorithm, which improves the trade-off between variance and bias by using action values as a baseline.
	    
	-   Exploits the temporal locality of neighboring observations for more efficient replay prioritization.
	    
	-   Combines the sample-efficiency of off-policy experience replay with the time-efficiency of asynchronous algorithms.
	    
	-   The Reactor architecture represents both a policy 
        $\pi (a|x)$ 
        and action-value function $Q(x, a)$.
	    
	-   Temporal differences are temporally correlated, with correlation decaying on average with the time-difference between two transitions.
	    
	-   More samples are made in areas of high estimated priorities, and in the absence of weighting this would lead to overestimation of unassigned priorities.
	    
	-   An important aspect of the architecture: an acting thread receives observations, submits actions to the environment, and stores transitions in memory, while a learning thread re-samples sequences of experiences from memory and trains on them.


## [Combining Policy Gradient and Q-learning](https://arxiv.org/abs/1611.01626), algorithm: PGQL


	-   Combines policy gradient with off-policy Q-learning, drawing experience from a replay buffer.
	    
	-   Considers model-free RL, where the state-transition function is not known or learned.
	    
	-   In policy gradient techniques the policy is represented explicitly and they improve the policy by updating the parameters in the direction of the gradient of the performance.
	    
	-   The actor refers to the policy and the critic to the estimate of the action-value function.
	    
	-   Combines two updates to the policy, the regularized policy gradient update, and the Q-learning update.
	    
	-   Mix some ratio of on- and off-policy gradients or update steps in order to update a policy.


## [Trust-PCL: An Off-Policy Trust Region Method for Continuous Control](https://arxiv.org/abs/1707.01891), algorithm: Trust-PCL


	-   Under entropy regularization, the optimal policy and value function satisfy a set of pathwise consistency properties along any sampled path.
	    
	-   By alternatively augmenting the maximum reward objective with a relative entropy regularizer, the optimal policy and values still satisfy a certain set of pathwise consistencies along any sampled trajectory.
	    
	-   Maximizes entropy regularized expected reward while maintaining natural proximity to the previous policy.
	    
	-   Entropy regularization helps improve exploration, while the relative entropy improves stability and allows for a faster learning rate. This combination is a key novelty.
	    
	-   It is beneficial to learn the parameter $\phi$ at least as fast as $\theta$, and accordingly, given a mini-batch of episodes they perform a single gradient update on $\theta$ and possibly multiple gradient updates on $\phi$.


## [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892), algorithm: PCM


	-   Identifies a strong form of path consistency that relates optimal policy probabilities under entropy regularization to softmax consistent state values for any action sequence.
	    
	-   Use this result to formulate a novel optimization objective that allows for a stable form of off-policy actor-critic learning.
	    
	-   Attempts to minimize the squared soft consistency error over a set of sub-trajectories $E$.
	    
	-   Given a fixed rollout parameter d, at each iteration, PCL samples a batch of on-policy trajectories and computes the corresponding parameter updates for each sub-trajectory of length d. Then PCL exploits off-policy trajectories by maintaining a replay buffer and applying additional updates based on a batch of episodes sampled from the buffer at each iteration. So, PCL is applicable to both on-policy and off-policy trajectories.
	    
	-   Unified PCL optimizes the same objective as PCL but differs by combining the policy and value function into a single model.


## [The Mirage of Action-Dependent Baselines in Reinforcement Learning](https://arxiv.org/abs/1802.10031)


	-   Demonstrates that the state-action-dependent(like Q-Prop or Stein Control Variates) don't result in variance reduction over a state-dependent baseline in commonly tested benchmark domains.
	    
	-   Having bias leads to instability and sensitivity to hyperparameters.
	    
	-   The variance caused by using a function approximator for the value function or state-dependent baseline is much larger than the variance reduction from adding action dependence to the baseline.
	    
	-   "We emphasize that without the open-source code accompanying, this work would not be possible. Releasing the code has allowed us to present a new view on their work and to identify interesting implementation decisions for further study that the original authors may not have been aware of."
	    
	-   Shows that prior works actually introduce bias into the policy gradient due to subtle implementation decisions:
	    

		-   Applies an adaptive normalization to only some of the estimator terms, which introduces a bias.
		    
		-   Poorly fit value function.
		    
		-   Fitting the baseline to the current batch of data and then using the updated baseline to form the estimator results in a biased gradient.


## [Action-depedent Control Variates for Policy Optimization via Stein’s Identity](https://arxiv.org/abs/1710.11198), algorithm: Stein Control Variates


	-   The idea of the control variate method is to subtract a Monte Carlo gradient estimator by a baseline function that analytically has zero expectation.
	    
	-   Constructs a class of Stein control variate that allows to use arbitrary baseline functions that depend on both actions and states.
	    
	-   Does not change the expectation but can decrease the variance significantly when it is chosen properly to cancel out the variance of Q.
	    
	-   The gradient in is taken w.r.t. the action a, no w.r.t the parameter \theta.
	    
	-   Connect 
        $\bigtriangledown_a log \pi(a|s)$
        to 
        $\bigtriangledown_\theta log \pi(a|s)$ 
        in order to apply Stein’s identity as a control variate for policy gradient. This is possible when the policy is reparameterizable in that 
        $a \approx \pi_\theta(a|s)$ 
        can be viewed as generated by 
        $a = f_\theta(s,\xi)$ 
        where 
        $\xi$ 
        is a random noise drawn from some distribution independently of $\theta$.


## [Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](https://arxiv.org/abs/1611.02247), algorithm: Q-Prop


	-   Combines the stability of policy gradients with the efficiency of off-policy RL.
	    
	-   Uses a Taylor expansion of the off-policy critic as a control variate.
	    
	-   Reduces the variance of gradient estimator without adding bias.
	    
	-   It can be easily incorporated into any policy gradient algorithm.
	    
	-   A common choice is to estimate the value function of the state $V_\theta(s_t)$ to use as the baseline, which provides an estimate of advantage function $A_\theta(s_t , a_t)$.
	    
	-   Constructs a new estimator that in practice exhibits improved sample efficiency through the inclusion of off-policy samples while preserving the stability of on-policy Monte Carlo policy gradient.
	    
	-   A more general baseline function that linearly depends on the actions.
	    
	-   An off-policy Q critic is trained but is used as a control variate to reduce on-policy gradient variance.
	    
	-   Uses only on-policy samples for estimating the policy gradient.


## [Dopamine: A Research Framework for Deep Reinforcement Learning](https://openreview.net/forum?id=ByG_3s09KX)


	-   A new research framework for deep RL that aims to support some of the RL goals diversities.
	    
	-   Open-source, TensorFlow-based, and provides compact yet reliable implementations of some state-of-the-art deep RL agents.
	    
	-   DQN is architecture research.
	    
	-   Double DQN, distributional methods, prioritized experience replay are algorithmic research.
	    
	-   Rainbow is comprehensive study.


## [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923), algorithm: IQN


	-   Reparameterizes a distribution over the sample space, causing to yield an implicitly defined return distribution and give rise to a large class of risk-sensitive policies.
	    
	-   Learns the full quantile function, a continuous map from probabilities to returns.
	    
	-   The approximation error for the distribution is controlled by the size of the network itself, and the amount of training.
	    
	-   Provides improved data efficiency with increasing number of samples per training update.
	    
	-   Expands the class of policies to more fully take advantage of the learned distribution.
    
	-   IQN is a type of [universal value function approximator (UVFA)](http://proceedings.mlr.press/v37/schaul15.pdf).
	    
	-   Samples TD errors decorrelated and the estimated action-values go from being the true mean of a mixture of n Diracs to a sample mean of the implicit distribution.


## [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044), algorithm: QR-DQN


	-   Sampling probabilistically, randomness in the observed long-term return will be induced.
	    
	-   The distribution over returns is modeled explicitly instead of only estimating the mean, which means learning the value distribution  instead of the value function.
	    
	-   Assigns fixed, uniform probabilities to N adjustable locations.
	    
	-   Uses quantile regression to stochastically adjust the distributions’ locations so as to minimize the Wasserstein distance to a target distribution(adapts return quantiles to minimize the Wasserstein distance between the Bellman updated and current return distributions).
	    
	-   The distribution over returns plays the central role and replaces the value function.
	    
	-   Transposes parametrization of C51 approach by considering fixed probabilities but variable locations.
	    
	-   Distributional Reinforcement Learning with Quantile Regression
	    
	-   Uses a similar neural network architecture as DQN, changing the output layer to be of size $\|A\| × N$, where $N$ is a hyper-parameter giving the number of quantile targets.
	    
	-   Replaces the Huber loss used by DQN with a quantile Huber loss.
	    
	-   Replaces RMSProp with ADAM.


## [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887), algorithm: C51


	-   Studies the random return Z whose expectation is the value Q.
	    
	-   The distributional Bellman equation states that the distribution of Z is characterized by the interaction of three random variables: the reward R, the next state-action (X′, A′), and its random return Z(X′,A′).
	    
	-   Instead of trying to minimize the error between the expected values, it tries to minimize a distributional error, which is a distance between full distributions.
	    
	-   Proves that the distributional Bellman operator is a contraction in a maximal form of the Wasserstein metric between probability distributions.
	    
	-   The Wasserstein metric, viewed as a loss, cannot generally be minimized using stochastic gradient methods.
	    
	-   Approximates the distribution at each state by attaching variable(parametrized) probabilities to fixed locations.


## [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477), algorithm: TD3


	-   Tackles the problem of overestimation bias and the accumulation of error in temporal difference method, by using double Q-learning. Double DQN doesn't work.
	    
	-   Target networks are critical for variance reduction by reducing the accumulation of errors.
	    
	-   Delaying policy updates until the value estimate has converged, to overcome the the coupling of value and policy.


## [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971), algorithm: DDPG


	-   Off-policy actor-critic, model-free, deterministic policy gradient on continuous action space.
	    
	-   Actor maps states to actions, instead of outputting the probability distribution across a discrete action space.
	    
	-   Next-state Q values are calculated with the target value network and target policy network.
	    
	-   For discrete action spaces, exploration is selecting a random action(e.g. epsilon-greedy) by a given probability. For continuous action spaces, exploration is adding some noise to the action itself(using Ornstein-Uhlenbeck process).
	    
	-   Soft target update: solve the Q update divergence. The target values are constrained to change slowly, improving the stability of learning.
	    
	-   Having both a target $\mu′$  and $Q′$  was required to have stable targets $y_i$  in order to consistently train the critic without divergence, which slows learning. However, in practice this was greatly outweighed by the stability of learning.


## [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf), algorithm: DPG


	-   Deterministic policy gradient can be estimated more efficiently than the usual stochastic policy gradient.
	    
	-   Uses an off-policy actor-critic algorithm to learn a deterministic target policy from an exploratory behaviour policy to address the exploration concerns.
	    
	-   Proves that deterministic policy gradient exists and has a simple model-free form.
	    
	-   Stochastic vs Deterministic policy gradient:
	    

		-   stochastic case: policy gradient integrates over both state and action spaces.
		    
		-   deterministic case: policy gradient integrates over the state space.


## [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290), algorithm: SAC


	-   Entropy is a quantity which basically determines how random a random variable should be.
	    
	-   Includes entropy regularization in the RL problem: policy function, value function, Q function.
	    
	-   Maximum entropy RL alters the RL objective, though the original objective can be recovered using a temperature parameter.
	    
	-   The maximum entropy formulation provides a substantial improvement in exploration and robustness.


## [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224), algorithm: ACER


	-   Introduces truncated importance sampling with bias correction, stochastic dueling network architectures, and efficient trust region policy optimization.
	    
	-   Matches the state-of-the-art performance of deep Q-networks with prioritized replay on Atari, and substantially outperforms A3C in terms of sample efficiency on both Atari and continuous control domains.
	    
	-   ACER may be understood as the off-policy counterpart of the A3C method.
	    
	-   Retrace and off- policy correction, SDNs, and trust region are critical: removing any one of them leads to a clear deterioration of the performance.
	    
	-   Mix some ratio of on- and off-policy gradients or update steps in order to update a policy.


## [Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation](https://arxiv.org/abs/1708.05144), algorithm: ACKTR


	-   Optimizes both the actor and the critic using Kronecker-factored approximate curvature (K-FAC) with trust region.
	    
	-   K-FAC: approximates the curvature by Kronecker factorization, reducing the computation complexity of the parameter updates.
	    
	-   Learns non-trivial tasks in continuous control as well as discrete control policies directly from raw pixel inputs.
	    
	-   ACKTR substantially improves both sample efficiency and the final performance of the agent in the Atari environments and the MuJoCo tasks compared to the state-of-the-art on-policy actor-critic method A2C and the famous trust region optimizer TRPO.


## [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286), algorithm: PPO-Penalty


	-   Investigates learning complex behavior in rich environments.
	    
	-   Locomotion: behaviors that are known for their sensitivity to the choice of reward.
	    
	-   Environments include a wide range of obstacles with varying levels of difficulty causing the implicit curriculum learning to the agent.
	    
	-   Uses TRPO, PPO and parallelism(like A3C).


## [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347), algorithm: PPO-Clip,  PPO-Penalty


	-   Compares to TRPO, simpler to implement, more general, and better sample complexity.
	    
	-   Algorithm:
	    

		-   Each iteration, each of N parallel actors collect T timesteps of data.
		    
		-   Then they construct the surrogate loss on these NT timesteps of data.
		    
		-   Optimize it with minibatch SGD for K epochs.
	    

	-   PPO is a robust learning algorithm that requires little hyper-parameter tuning.


## [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438), algorithm: GAE


	-   Uses value functions to estimate how the policy should be improved.
	    
	-   Reduces variance while maintaining a tolerable level of bias, called Generalized Advantage Estimator(GAE).
	    
	-   Trains value functions with trust region optimization.


## [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), algorithm: TRPO


	-   Uses the expected advantage value of function A to improve the current policy.
	    
	-   Guarantees to increase the policy performance, or leave it constant when the expected advantage is zero everywhere.
	    
	-   Applies lower bounds on the improvements of the policy to address the issue on how big of a step to take.
	    
	-   Uses a hard constraint rather than a penalty because it is hard to choose a single value of à that performs well across di erent problems.
	    
	-   A major drawback is that such methods are not able to exploit off-policy data and thus require a large amount of on-policy interaction with the environment, making them impractical for solving challenging real-world problems.


## [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783), algorithm: A3C


	-   Uses asynchronous gradient descent for optimization of deep neural network controllers: run many agents on many instances of the environment.
	    
	-   Experience replay drawbacks: uses more memory and computation per real interaction, and requires off-policy learning algorithms.
	    
	-   Parallelism decorrelates agents data.
	    
	-   A3C works well on 2D and 3D games, discrete and continuous action spaces, and can train feedforward or recurrent agents.
	    
	-   Has threads over one CPU to do the learning.
	    
	-   Multiple learners increase the exploration probability.
	    
	-   A3C trains agents that have both a policy (actor) distribution \pi and a value (critic) estimate V \pi .


## [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298), algorithm: Rainbow DQN


	-   Combines 6 extensions to the DQN algorithm: [double DQN](https://arxiv.org/abs/1509.06461), [prioritize experience replay](https://arxiv.org/abs/1511.05952), [dueling network architecture](https://arxiv.org/abs/1511.06581), [multi-step bootstrap targets](https://arxiv.org/abs/1901.07510), [distributional Q-learning](https://arxiv.org/abs/1707.06887), and [noisy DQN](https://arxiv.org/abs/1706.10295).
	    
	-   SOTA, both data efficiency wise and performance wise.
	    
	-   Does an ablation study over the 6 extensions.


## [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952), algorithm: PER


	-   Agents can remember and reuse experiences from the past.
	    
	-   Increases the replay probability of experience tuples that have a high expected learning progress.
	    
	-   Greedy TD-error prioritization: the amount the RL agent can learn from a transition in its current state.
	    
	-   Greedy TD-error prioritization problems: only update the transitions that are replayed, sensitive to noise spikes, and focuses on a small subset of experiences.
	    
	-   To overcome problems, use a stochastic sampling method to interpolate greedy prioritization and uniform random sampling.


## [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461), algorithm: Double DQN


	-   Reduces the observed overoptimism of the DQN: decomposing the max operation in the target into action selection and action evaluation.
	    
	-   Good side of being optimistic: helps exploration. Bad side: leads to a poorer policy.
	    
	-   The weights of the second network are replace with the weights of the target network for the evaluation of the current greedy policy.
	    
	-   Evaluates the greedy policy according to the online network, but using the target network to estimate its value.


## [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581), algorithm: Dueling DQN


	-   Uses two separate estimators: one for state value function(V), one for state-dependent action advantage function(A).
	    
	-   Two streams representing V and A function, but using same convolutional neural network.
	    
	-   Two streams are combined using an aggregation layer.
	    
	-   Intuition: learning valuable states, w/o having to learn the effect of each action for each state.
	    
	-   Clips gradients to have their norm less than or equal to 10.


## [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527), algorithm: Deep Recurrent Q-Learning


	-   Adds RNNs to the DQNs.
	    
	-   Replaces the first post-convolutional fully-connected layer with an LSTM.
	    
	-   DQN agents are unable to perform well on games that require the agent to remember events from far past(k frame skipping).
	    
	-   Uses only one single game state.
	    
	-   Is able to handle partial observability.
	    
	-   Partial observability: with probability p = 0.5, the screen is either fully obscured or fully revealed.
	    
	-   Uses "Bootstrapped Sequential Updates" and "Bootstrapped Random Updates" to update RNN weights.


## [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), algorithm: DQN


	-   Uses convolutional neural network to extract features directly from the input images, which are game states.
	    
	-   No adjustments of the hyperparameters for each game.
	    
	-   Model-free: no explicit construction of environment.
	    
	-   Off-policy: learning policy and behavior policy are different, same as Q-learning.
	    
	-   Uses experience replay: better data efficiency, decorrelate data, avoiding oscillations or divergence in parameters.
	    
	-   Scales scores of different games to -1, 0, 1.
	    
	-   Uses frame skipping: agent sees and selects actions of every k frame.
	    
	-   Freezes the parameters of of the target network for a fixed time while updating the online network by gradient descent.