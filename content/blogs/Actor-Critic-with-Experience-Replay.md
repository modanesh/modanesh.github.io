---
title:  "Actor-Critic with Experience Replay"
description: A brief overview of the ACER RL algorithm is provided.
date: 2021-01-29
author: ["Mohamad H. Danesh"]
showToc: true
disableAnchoredHeadings: false
---


## ACER

It is an off-policy actor-critic model with experience replay, greatly increasing the sample efficiency and decreasing the data correlation.
The reason for doing that is because ACER is off-policy and to control the stability of the off-policy estimator:
- It has multiple workers (as A2C);
- It uses replay buffer (as in DQN);
- It uses Retrace Q-value estimation;
- It truncates the importance weights with bias correction;
- It applies TRPO.

Deep Q-learning methods are most sample efficient techniques. However, they have two important limitations. **First**, the deterministic nature of the optimal policy limits its use in adversarial domains. **Second**, finding the greedy action with respect to the Q function is costly for large action spaces.
ACER matches SOTA of DQN with PER on Atari, and substantially outperforms A3C in terms of sample efficiency on both Atari and continuous control domains.

In the calculation of the gradient of the policy wrt to theta (following equation), we can replace $A$ with state-action value $Q$, the discounted return $R$, or the temporal difference residual, without introducing bias:

$$g = \mathbb{E}_ {x_{0:\infty}, a_{0:\infty}} [\sum_{t \geq 0} A^{\pi} (x_t, a_t) \bigtriangledown_{\theta} log \pi_\theta (a_t, x_t)]$$


But since we use deep networks thus introducing additional approximation errors and biases. Comparing the expressions, we can have instead of $A$, the discounted return will have higher variance and lower bias but the estimators using function approximation will have higher bias and lower variance. In ACER, they combine $R$ with the current value function approximation to minimize bias while maintaining bounded variance.


ACER is A3C's off-policy counterpart. It uses parallel agents. It also uses a single neural net to estimate policy and value function. Gradient of policy with importance sampling:

$$\hat{g}^{imp} \left( \prod_{t=0}^{k} \rho_t \right) \sum_{t=0}^{k} \left( \sum_{i=0}^{k} \gamma^i r_{t+i} \right) \bigtriangledown_\theta log \pi_\theta (a_t, x_t)$$

It uses discounted rewards to reduce bias, but has a very high variance since it involves a product of many potentially unbounded importance weights. Truncating this product can prevent it from exploding. But this also adds bias. Then Degris suggested to use marginal value functions over the limiting distributions (The limiting distribution can be used on small, finite samples to approximate the true distribution of a random variable) to have this gradient expression:

$$g^{marg} = \mathbb{E}\_{x_t \sim \beta , a_t \sim \mu} [\rho_t \bigtriangledown\_\theta log \pi\_\theta (a_t, x_t) Q^\pi (x_t, a_t)]$$

It depends on $Q_\pi$ and not on $Q_\mu$, consequently we must be able to estimate $Q_\pi$. And instead of having a long product, it has marginal importance weight which lowers the variance. In ACER, they use Retrace to estimate $Q_\pi$.

To have off-policy samples ($Q_\pi$), there's a need for importance sampling:

$$\Delta Q^{imp} (S_t, A_t) = \gamma^t \prod_{1 \leq \tau \leq t} \frac{\pi(A_tau, S_tau)}{\beta(A_tau, S_tau)}\delta_t$$

Importance sampling is a technique of estimating the expected value of $f(x)$ where $x$ has a data distribution $p$. However, instead of sampling from $p$, we calculate the result from sampling $q$:

$$\mathbb{E}_p [f(x)] = \mathbb{E}_q (\frac{f(x)p(x)}{q(x)})$$

Retrace Q-value estimation method modifies $\Delta Q$ to have importance weights truncated by no more than a constant $c$:

$$\Delta Q^{ret} (S_t,A_t) = \gamma^t \prod_{1 \leq \tau \leq t} min(c, \frac{\pi(A_tau, S_tau)}{\beta(A_tau, S_tau)}\delta_t)$$

ACER uses $Q^{ret}$ as the target to train the critic by minimizing the L2 error term: $(Q^{ret}(s,a)−Q(s,a))^2$.

Finally, the multi-step estimator $Q^{ret}$ has two benefits: to reduce bias in the policy gradient, and to enable faster learning of the critic, which further reduces bias.

Truncate the importance weights with bias correction: To reduce the high variance of the policy gradient, ACER truncates the importance weights by a constant $c$, plus a correction term.

Furthermore, ACER adopts the idea of TRPO but with a small adjustment to make it more computationally efficient: rather than measuring the KL divergence between policies before and after one update, ACER maintains a running average of past policies and forces the updated policy to not deviate far from this average.

Continuous case:
It is not easy to integrate over $Q$ to derive $V$ in continuous action spaces. To overcome this issue, they propose a network architecture similar to dueling DQN which estimates both $V_\pi$ and $Q_\pi$ off-policy (SDNs). In addition to SDNs, they also construct (equation 14) novel target for estimating value function derived via the truncation and bias correction trick.

