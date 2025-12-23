---
title: "Domain Randomization"
description: Bridging the Reality Gap - A Survey of Domain Randomization and Future Horizons
date: 2025-12-23
author: ["Mohamad H. Danesh"]
showToc: true
disableAnchoredHeadings: false
---

# Bridging the Reality Gap: A Survey of Domain Randomization and Future Horizons

**Date:** January 2025
**Category:** Sim-to-Real, Reinforcement Learning, Robotics

Simulation is the bedrock of modern robotic learning. It allows us to train agents safely and parallelize data collection at scales impossible in the physical world. However, the "Reality Gap", the discrepancy between simulated physics and the real world, remains a formidable barrier. Policies trained in perfectly deterministic simulations often fail catastrophically when deployed on physical hardware.

The dominant solution to this problem is **Domain Randomization (DR)**. By randomizing the physical parameters of the simulation (friction, mass, damping, etc.) during training, we force the agent to learn robust strategies that can generalize to the real world as just another "variation" of the simulation.

But standard DR is not a silver bullet. In this literature review, I’ve been exploring three critical questions:

1.  **Sample Efficiency:** Does blind randomization lead to inefficiency and delayed convergence?
2.  **Safety:** Some adaptive DR methods require deploying policies mid-training. How can we avoid unsafe real-world testing?
3.  **Parameter Selection:** How do we choose which parameters to randomize and by how much?

In this post, I will categorize the landscape of DR research and propose a new direction using generative modeling.

---

## 1. The Optimization Perspective: Guiding the Randomization

The standard approach to DR involves sampling parameters uniformly from a fixed range. However, this often wastes computation on trivial scenarios or impossible ones. To address this, several works have framed randomization as an optimization or curriculum problem.

**AutoAugment** (Cubuk et al., 2018) and **"Learning to Simulate"** (Ruiz, 2019) introduced the idea of treating data generation as a Reinforcement Learning (RL) problem. Much like Neural Architecture Search (NAS), a policy is trained to select randomization parameters that maximize performance on a validation set. While AutoAugment focused on image classification, the principle applies to robotics: we want to find the "hardest" or "most informative" training samples.

This concept is formalized in **Active Domain Randomization (ADR)** (Mehta et al., 2019) and **DeceptionNet** (Zakharov et al., 2019).
*   **ADR** searches for environment variations that cause the most discrepancy compared to a reference environment. It uses a discriminator to distinguish between randomized and reference trajectories, using the difficulty of prediction as a reward to encourage diversity (Stein Variational Policy Gradient).
*   **DeceptionNet** uses an adversarial approach, maximizing the difference between prediction and labels to learn the most "confusing" visual distortions.

Similarly, **AutoDR** (OpenAI, 2019) applies a curriculum: it starts with a narrow environment and only widens the randomization ranges (e.g., friction, object size) once the policy reaches a success threshold. This "expand-when-ready" loop drives robust transfer without needing real-world tuning.

## 2. Closing the Loop: Real-Data Guided Adaptation

While the methods above optimize for *difficulty* or *diversity*, they don't necessarily optimize for *reality*. A major branch of research focuses on using real-world data to anchor the simulation distribution.

**SimOpt** (Chebotar et al., 2019) minimizes the discrepancy between simulated and real trajectories ($D(\tau_{sim}, \tau_{real})$). It propagates gradients through the simulator to adjust the randomization distribution $\phi$ so that the simulation behaves like reality.

**BayRN** (Muratore et al., 2021) and **NPDR** (Muratore et al., 2021) take a Bayesian approach:
*   **BayRN** uses Bayesian Optimization to adapt the distribution of simulator parameters based on real-world returns.
*   **NPDR** uses neural likelihood-free inference (normalizing flows) to maintain a posterior distribution over simulator parameters.
*   *Crucially*, both methods require the policy to be deployed on the real robot **during training** to gather data. This allows for high precision but raises safety concerns regarding mid-training deployment.

Conversely, methods like **SIPE** (Mehta et al., 2020) focus on system identification to calibrate the simulator offline, though their experiments were primarily validated in Sim-to-Sim transfer settings.

## 3. Zero-Shot Transfer and Robustness

To avoid the risks of mid-training deployment, many researchers focus on Zero-Shot transfer, training a policy so robust that it works on the real robot immediately.

*   **RMA** (Kumar et al., 2021) is a standout work here. It trains an adaptation module entirely in simulation. By randomizing terrain and physics, the agent learns to estimate extrinsics from its own state-history, allowing it to adapt to the real world on the fly without real-world fine-tuning.
*   **DORAEMON** (Tiboni et al., 2024) formulates a constrained optimization problem. It attempts to maximize the entropy of the parameter distribution (keeping it as wide as possible) while maintaining a minimum success constraint. This achieves robustness without ever seeing real data during training.
*   **GenLoco** (Feng et al., 2022) takes this a step further by randomizing **morphology**. By procedurally generating diverse robot bodies during training, the controller learns a generalized locomotion strategy that works across different robot shapes and dynamics zero-shot.

Theoretical underpinnings by **Chen et al. (2022)** suggest that memory is key. They model DR as a latent MDP and prove that history-dependent policies (like LSTMs or Transformers) are essential for successful sim-to-real transfer, a concept practically applied in **LocoTransformer** (Yang et al., 2022) which fuses proprioception and depth using Transformers.

## 4. The Visual Gap: Generative Adversarial Networks

For vision-based tasks, the gap isn't just physical; it's visual. **RCAN** (James et al., 2019) combines Domain Adaptation and DR using cGANs. It translates randomized simulation images into a "canonical" version and does the same for real images, ensuring the agent sees a consistent input regardless of the domain.

---

## Future Direction: Diffusion for Domain Randomization

Despite these advances, a core issue remains: **Initialization.**
Creating randomizable simulation environments is time-intensive. We rely on expert knowledge to set the initial "mean" and "variance" of parameters (friction, damping, etc.). If these priors are wrong, even adaptive methods struggle.

Gaussian or Uniform distributions are often insufficient to capture the complex, multimodal correlations of the real world.

### A Proposal
I am currently exploring a method to automate the generation of simulation environments using **Diffusion Models**.

> **Hypothesis:** instead of relying on simplistic Gaussian priors, we can leverage diffusion models to learn complex, realistic parameter distributions initialized from limited real-world data.

By treating the simulator parameters (and perhaps even terrain/morphology profiles) as samples from a diffusion process, we can capture nuanced correlations that manual tuning misses. This "Real-to-Sim-to-Real" pipeline could drastically reduce the workload of setting up simulations and provide a much stronger initialization for Sim-to-Real transfer.

### Selected References
*   *Chebotar et al. (2019). Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience.*
*   *Kumar et al. (2021). RMA: Rapid Motor Adaptation for Legged Robots.*
*   *Muratore et al. (2021). Data-efficient Domain Randomization with Bayesian Optimization.*
*   *Tiboni et al. (2024). DORAEMON: Domain Randomization with Entropy Maximization.*