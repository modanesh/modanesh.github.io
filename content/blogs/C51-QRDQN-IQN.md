---
title: "Distributional Reinforcement Learning"
description: Presenting some of the most fundamental works on distributional RL.
date: 2021-03-03
author: ["Mohamad H. Danesh"]
showToc: true
disableAnchoredHeadings: false
---



## Distributional RL

In common RL approaches, we have a value function which returns a single value for each action. This single value is the expectation of a true distribution which in the distributional RL, we seek to return that for each action. In common RL, value function is defined using the Bellman's equation:

$$Q(x,a) = \mathbb{E} R(x,a) + \gamma \mathbb{E} Q(X', A')$$

In distributional RL, we need to drop those expectations and so the distributional Bellman's equation would look like (onte that this equality sign means that the sides are random variables drawn from the same probability distribution law):

$$Z(x,a) = R(x,a) + \gamma Z(X', A')$$

According to this equation, there are three random variables that affect the value distribution: reward function, transition function, and next state-action value distribution. Distributional RL has studied before, but it was limited to specific cases like risk-sensitive policies or theoretical analysis.


## Prerequisites

KL divergence: sum of log probability ratios of two probability distributions. For two distributions with non-overlapping supports, it'll be infinite.

Total variation: how much probability mass two distributions disagree. It's done by subtracting all the mass where they predict the same outcome. Shifting two distributions (e.g. multiplying by discount factor), total variation won't change so it's not a good distance.

Wasserstein: The area (integral) between the CDFs of two random variables. The three properties of the Wasserstein metric is very useful for the RL case.


## C51 [1]
In common RL, the Bellman operator has the alpha-contraction property which basically means that by applying the Bellman operator, our estimation of the value function gets closer to the optimal one. But in dist RL, that's not as straight forward as common RL. In policy evaluation, the distributional Bellman operator is an alpha-contraction only in a maximal form of the Wasserstein metric. In policy improvement, things are messier and it requires more investigation. An example is provided further in the paper that discusses this issue. Using value distributions over value functions offers a few benefits like preserving the multimodality of returns, and decreasing the bad effects of learning from a non stationary policy, which all leads to more stability in learning.

MDP setting definition is:

$$(X, A, R, P, \gamma)$$

The common definition for value function which estimates a value for state action pair based on policy $\pi$ is:

$$Q^\pi(x,a) = \mathbb{E} R(x,a) + \gamma \mathbb{E}_{P, \pi} Q^{\pi}(x', a')$$

We have the definition for optimal value function which has this max operator for taking the next action:

$$Q^{*}(x,a) = \mathbb{E} R(x,a) + \gamma \mathbb{E}_{P} max\_{a' \in A} Q^{*}(x', a')$$

This equation has a unique fixed point which is very important. It means that starting from any arbitrary value function, by iteratively updating the value function this way, we would end up having the optimal unique value function.

And the definition for Bellman operator and optimality Bellman operator. They basically take a value function as input and return an updated value function which can describe the expected behavior of the learning algorithm:

$$\mathrm{T}^{\pi} Q(x,a) := \mathbb{E} R(x,a) + \gamma \mathbb{E}_{P, \pi} Q(x',a')$$

$$\mathrm{T} Q(x,a) := \mathbb{E} R(x,a) + \gamma \mathbb{E}_{P} max\_{a'\in A} Q(x',a')$$

The Wasserstein metric which for two CDFs (probability that $U$ will have a value less than or equal to $y$) is defined as:

$$d_p (F,G) := \inf_{U,V} ||U-V||_p$$

Infimum is actually the greatest lower bound. Next we can remove the inf and put inverse CDFs because inf is implicit in its definition and we get a LP norm of a vector. And having $p < \infty$ we arrive to this equation:

$$d_p (F,G) = \left( \int_{0}^{1} |F^{-1}(u) - G^{-1}(u)|^p du \right)^{1/p}$$

And eventually we get to this equation which basically means that Wasserstein is a metric wrt to these random variables:

$$d_p (U,V) := \inf_{U,V} ||U-V||_p$$

This Wasserstein metric has three important properties that are useful and discussed in the paper.

The maximal form of the Wasserstein metric for value distributions are (supremum is least upper bound):

$$d_p (Z_1, Z_2) := sup_{x,a} d_p(Z_1(x,a),Z_2(x,a))$$

Policy evaluation: distributional Bellman operator is a gamma contraction in $d_p$. It can be concluded that the sequence of value functions under policy $\pi$ converges to $Z_\pi$ in $d_p$ (for $1 \leq p \leq \infty$). It's important that it only works for Wasserstein metric, not any other metrics. But further in the paper for practical usage they switch to kl divergence.

Policy improvement: in common RL where we have value functions, all optimal policies get to the same $Q^*$ which is the unique fixed point, but in distributional RL, there may be many optimal value distributions for different optimal policies. And that's what makes it a bit messy, meaning that this operator is not a contraction in any metric. In the best case, it converges to the non stationary optimal value distribution.

Now to approximate the value distribution, they propose a method to have a discrete distribution with $N$ atoms at fixed locations in a range of $v_min$ to $v_max$. Having a discrete distribution causes the Bellman update and our estimation to have disjoint supports meaning that they mostly don't overlap. In this case, KL-JS-total variation metrics are problematic but Wasserstein metric could be very useful as we also saw it's theoretically strong. But, since transitions are sampled, the Wasserstein loss and the sample loss are not exactly the same (proposition 5). Why Wasserstein metric hasn't been used in the paper: biased Wasserstein gradients. There's an approximation which make things difficult.

To solve this issue, the Bellman update is projected into the sampled discrete distribution (fig 1 $\phi$ operator). This projection step is simply a linear interpolation to the closest neighbor. And now since we have overlapping distributions, we can use KL divergence as the metric (and loss):

$$D_{KL} (\Phi \hat{\mathrm{T}} Z_{\tilde{\theta}} (x,a) || Z_\theta (x,a)) $$

Which is in contrast to what they've talked about in most of the paper.

In experiments: there are sources of stochasticity even in deterministic Atari envs: partial observability, non stationary policies, and approximation errors. Authors say that having 51 atoms results in better performance compared to other numbers of atoms and they don't know why is that. And that's where their algorithm got it's name from. Nothing surprising in their experiments, they have better sample efficiency and overall performance compared to DQN. The only exciting thing is that it performs very good in games with sparse rewards (VENTURE and PRIVATE EYE). This suggests that value distributions are better able to propagate rarely occurring events.

The distinction we wish to make is that learning distributions matters in the presence of approximation.


## QR-DQN [2]
Using kl divergence in C51 left a theory-practice gap open which in this paper they talk about and try to address. One interesting point they raise is that since they didn't follow the theory they provided, it's hard to justify the improvements in performance they reported which actually is a fair point if you think about it.

The contributions of this paper are: first, instead of having a set of fixed location atoms, they parameterize the value distribution using a set of fixed probabilities with adjustable locations. Doing this they will be able to use Wasserstein metric and so close the gap. Second, in order to have Wasserstein metric, they use quantile regression. Third, they provide the contraction property for their overall algorithm.

They raise the concern about C51's performance saying that although it still were acting by maximizing expectations, it's SOTA performance was the main interesting thing in the paper. And I also think it is since in C51, in policy improvement, actions were taken based on the max operator and so it basically is DQN. Anyone has an idea to explain this concern?

Up to here they talk about the Wasserstein metric and bellman operator's contraction property under this metric. But Wasserstein metric cannot be used according to theorem 1: let $\hat{y}$ be the empirical distribution draw from $B$. And let $b_\mu$ be a distribution parameterized by $\mu$. Then, the minimum of the expected sample loss is in general different from the minimum of the true Wasserstein loss.

Figure 2: it shows the Wasserstein metric between two random variables. There are 4 quantiles q1, q2, q3, q4. And their median is named as $\hat{\tau}_1, \hat{\tau}_2, \hat{\tau}_3, \hat{\tau}_4$. The sum of the red shaded regions is the wasserstein metric. If $n$ were bigger, the wasserstein metric would decrease because our estimation would be closer to the real distribution.

Further, they introduce their method. Having $N$ quantiles, each with a weight of $1/N$. So If the atoms are $N$ then the $i$-th atom corresponds to a quantile of $\tau_i$.

Benefits of quantile distribution: no need to have pre-specified bounds for discrete distribution's support. No need to have a projection step since we'll have the wasserstein metric here so the non-overlapping supports won't make any problems. Take advantage of wasserstein metric, using quantile regression.

(quantile regression) when we want to understand the relations between variable outside of the mean, or when we have non-normally distribution data, quantile regression can be very useful. Because it penalizes estimates differently depending on which quantile they belong to and whether they are too high or too low. For low quantiles, it penalizes overestimates more heavily, while for high quantiles it penalizes underestimates more heavily.

Quantile huber loss: the idea of huber loss is to make L1 norm's gradient better behaved around 0, by turning it into an L2 norm. So that's exactly the motivation behind combining quantile regression and huber loss, to make quantile regression well-behaved around 0.

Proposition 2 proves that the QR-DQN algorithm has a contraction property.

Comparing QR-DQN with plain DQN: everything is the same except two things: output size is changed so as to return values according to the number of quantiles and actions. Next, the loss function is changed. Also, the rmsprop is replaced by Adam for optimization.

Their experimental results are good, nothing surprising reported.

## IQN [3]
Compared to QR-DQN, there are two major differences: instead of using the quantile divided by the same probability, the probabilities are randomly sampled implicitly, and also the network architecture is different. IQN also uses the quantile regression technique as QR-DQN. As an example: let's say the number of quantiles is 4. In this case, the quantile value of QR-DQN is [0.25, 0.5, 0.75, 1], and QR-DQN derives a support corresponding to the median of the quantile to minimize the Wasserstein distance. That is, we estimate the supports corresponding to [0.125, 0.375, 0.625, 0.875].
However, in IQN, the quantile value $\tau$ is randomly sampled between 0 and 1. If there are 4 quantiles, let's say that 4 randomly extracted values between 0 and 1 are [0.12, 0.32, 0.78, 0.92]. Sampling like this from the value distribution gives the opportunity to measure the level of risk the policy takes.

Advantages of IQN over QR-DQN are that the approximation error won't be controlled by the number of quantiles. It has better sample efficiency. It is also capable of learning a wider range of policies.

Risk in RL: let's consider two actions each with a normal value distribution, but different means and standard deviations. Let's say action 1 has smaller mean and std than action 2. Now, since action 1 that has a smaller variance, the probability of receiving a return close to the average is high. But the expected value of the value distribution is less than action 2. In Distributional RL we compare the expected value of the distribution and select an action, so in this case action 2 will be selected. For a2, it is a distribution with a very large variance. In this distribution, very small returns may be derived or very high returns may be derived in some cases. In Distributional RL, such a large variance is said to be "high risk" when confidence in the result is low. Conversely, in the case of a1, which has a relatively high confidence in the result, it can be said that the "risk is small" compared to a2. When learning is performed through sampling and selecting an action, it is possible to select an action according to this risk. There are two types of risk sensitive policies: risk-averse policy and risk-seeking policy.

Implicit Quantile Networks (implementation): Looking at the figure 1, the result obtained through the convolution function ($\psi$) and the result obtained through the embedding function ($\phi$) for $\tau$ are element-wise multiplied. And by performing a fully-connected layer operation on the result, we finally get the value distribution for action $a$.

Embedding function ($\phi$) (equation 4): The role of this function is to return an embedding sampled $\tau$ as a vector. The function for embedding tau is an $n$ cosine basis function .

Quantile huber loss (equation 3): Quantile Huber Loss used in QR-DQN is used as it is. But, in this paper, the equation is expressed as eq 3. The only point about this equation is that $N$ is the number of $\tau$ samples sampled for the online network, and $N'$ is the number of $\tau$ samples sampled for the target network. The rest is the same as QR-DQN.

(Under eq 4): they hypothesized that $N$ would affect the sample complexity, and $N'$ would affect the variance of gradients. But results show that initially $N'$ is effective but as the training goes on, it loses its impact.

Types of risks (fig 3): Risk-averse > Neutral > Risk-seeking. About this, the longer the game lasts, the higher the rewards would be, so it can be considered that the risk-averse policy of choosing a stable reward shows good performance even though it is less than the risk-seeking policy of choosing a high reward while being uncertain.

One interesting point is that in table 1, the mean and median of scores to compare different algorithms are based on different experiment setups. I mean for DQN, PER, and c51 scores are only for 1 seed but for IQN report are for 5 seeds. Which is not fair (?).


## References

[1] Marc G Bellemare, Will Dabney, and Rémi Munos. A distributional perspective on reinforcement learning. In Proceedings of the 34th International Conference on Machine Learning-Volume 70,  pages 449–458. JMLR. org, 2017.

[2] Will Dabney, Mark Rowland, Marc G Bellemare, and Rémi Munos. Distributional reinforcement learning with quantile regression. In Thirty-Second AAAI Conference on Artificial Intelligence, 2018.

[3] Will Dabney, Georg Ostrovski, David Silver, and Remi Munos. Implicit quantile networks for distributional reinforcement learning. In International Conference on Machine Learning, pages 1104–1113, 2018.
