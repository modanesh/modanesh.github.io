---
title:  "What Should a Robot Dream About?"
description: Notes on world models for robotics.
date: 2026-06-14
author: ["Mohamad H. Danesh"]
showToc: true
disableAnchoredHeadings: false
---



# What Should a Robot Dream About?

*Notes on world models for robotics*

![Image generated with Gemini](project_assets/robot_dreaming.png)

---

Somewhere in the last eighteen months, "world model" stopped being a research term and became a land grab. Yann LeCun walked out of Meta and raised a billion-dollar seed round in Paris to build them. DeepMind shipped Genie 3 and then opened it to the public as Project Genie: type a sentence, walk around inside the result at 24 frames per second. Waymo bolted a driving-specialized world model on top of Genie 3 so its cars could rehearse tornadoes and elephants. NVIDIA went from Cosmos to Cosmos 3 in eighteen months and now calls world models the operating layer of "physical AI." Tesla folds its cars and its humanoid into one neural simulator. 1X calls its world model the cognitive core of a home robot. NVIDIA's GEAR lab released a 14B parameter video model that doubles as a zero-shot robot policy and half the field declared it robotics' GPT-2 moment. Even the theorists piled on: an ICML paper out of DeepMind proved that any agent competent at long-horizon goals *must* contain a predictive model of its environment, whether its designers intended one or not.

So the argument about whether robots need world models is, functionally, over. What's left, and what almost nobody is arguing about carefully, is *what kind*. "World model" now names at least five different design commitments that get bundled together because they share a noun. I work with and train these models for my PhD, for robots with arms and legs, and the longer I do it the more I think the field's real disagreements live along five fault lines. This post is my attempt to pull them apart and describe where I stand on each.

---

## The cartographers are arguing

Before the fault lines, it's worth listening to how the field's most prominent voices have drawn the map, because they disagree in ways that are easy to miss under the shared vocabulary, and a robotics-from-the-trenches view ends up siding with each of them on some axes and against them on others.

**Fei-Fei Li** has written one of the field's most-read manifesto for the *pixels-and-geometry* program. [From Words to Worlds](https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence) argues that spatial intelligence is AI's next frontier and that LLMs are "wordsmiths in the dark, eloquent but inexperienced". Her follow-up, [A Functional Taxonomy of World Models](https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models) is the single cleanest carving of the term I've read. Every so-called world model is a projection of the old POMDP loop, and the projections sort into **renderers** (output observations/pixels), **simulators** (output state, geometry and physics a program can compute on), and **planners** (output actions). I think this taxonomy is correct and underused, and most of my five fault lines are arguments *within* her planner box. Where I part company is emphasis. Her bet is that the simulator is the linchpin, master geometry-and-physics state and you can project down into pixels for humans or actions for robots. For the specific job of closing a control loop, I'd argue the opposite ordering: you don't need a metrically faithful simulator of the whole scene to act well, you need a predictor at the resolution of decisions, and chasing explicit 3D state can be capacity spent where a controller doesn't need it. She's describing the model that serves every downstream consumer, while I'm describing the model that serves the gripper. Both can be right about different artifacts, which is, in fact, her own concluding move toward a unified model.

**Yann LeCun** is the patron saint of the other camp, and the bet got literal this year with his departure from Meta and raising a billion-dollar seed for a company built on the thesis that prediction belongs in representation space, not pixels. I'm temperamentally on his side of fault line 1, and the JEPA lineage is the spine of the lean control-loop stack. But the strong form of his view, that reconstruction is a near-dead end, sits awkwardly next to the fact that the most capable robot world models of 2026 are video models, and that internet-scale pixel pretraining is exactly what a robot lab can't reproduce. The honest position isn't LeCun-vs-Fei-Fei rather it's that the frontier system borrows from both, and the open question is what appearance-trained encoders quietly fail to encode about contact and force.

**Richard Sutton and David Silver**, in [Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf) supply the deepest *why* under all of this: human data is a ceiling, and superhuman competence will come from agents learning on streams of grounded experience with rewards that come from consequences rather than human opinion. Crucially for us, they name the world model as the mechanism that lets an agent "ground thinking in the external world", plan in terms of its own actions and their causal effects. I find the diagnosis bracing and largely right. My friction is the reward clause. In real robot manipulation, "grounded reward from consequences" is doing enormous unspoken work, because the thing that tells you whether the cup was placed is either a human or another learned model with its own blind spots which is precisely the honesty problem that fault line 5 is about. Experience is the right substrate. *Graded* experience is the hard part nobody's manifesto has costed.

**Jonathan Richens and colleagues at DeepMind** turned the slogan into a [theorem](https://arxiv.org/abs/2506.01622). Any agent that generalizes across long-horizon goals must have learned a predictive model of its environment, one you can in principle extract from its policy. I lean on this in fault line 3, because it dissolves the "model-based vs. model-free" framing. The model is always there. The only question is whether it's queryable on purpose or fossilized inside a reflex. If I have a quarrel it's with how the result gets cited: "contains a world model" is a long way from "contains a world model accurate enough to plan novel behavior against," and the theorem's own error bounds say the sparse, common-transition model an agent secretes is not the high-resolution model a planner needs. The theorem ends the existence debate. It doesn't end the engineering one.

Then there's the most useful piece of pushback the field has produced recently, a position paper out of Stanford, ETH, TU Darmstadt and collaborators with the deliberately deflating title [Robots Need More Than VLAs & World Models](https://arxiv.org/abs/2606.06556). Its claim is that everyone is fighting over the policy and the predictor while the actual bottleneck is *grounding*: the world is full of human video, simulation rollouts, and interaction logs, almost none of which arrives with the action labels, contacts, and reward structure a robot can learn from. World models, in their telling, don't solve this. They *relocate* it, because a visually plausible rollout that ignores contact and friction is not yet supervision. I think this is the correct cold shower, and it reframes my own fault lines as necessary-but-insufficient. Getting the dynamics head, the resolution, and the search right still leaves you needing a machine that turns messy physical experience into something a robot can train on. The one place I'd resist is the implied modesty about world models specifically, for the *evaluation* and *multimodality* problems I care about, the world model isn't a component awaiting grounding, it's part of the grounding apparatus itself.

And then the companies, who vote with engineering rather than essays. **1X** put the bluntest version of fault line 5 in writing back in 2024, and it has aged well. The real reason to build a robot world model, they argued, is *evaluation*. You cannot tell whether a new model is better across a thousand tasks when the same checkpoint degrades over fifty days from nothing but a lighting change, so you learn a simulator from raw sensor data and score policies inside it. They were refreshingly candid about the failure modes too (objects that vanish, plates suspended in air, no self-recognition in a mirror), which is exactly the honesty the imagined-evaluation thread needs. I'll register one disagreement with the broader industry mood the company now sits in. The line, repeated across NVIDIA, Tesla, and the humanoid labs, that scaling a video world model is the "GPT moment" for robotics. The physics-law evidence (below) says scaled video models generalize in-distribution and fail out of it, behaving like case-based mimics. In my opinion, a GPT moment requires extrapolation those models haven't yet shown. The bet may pay off, but it isn't won.

With those positions on the table, here are the five places I think the real arguments live.

## Fault line 1: The resolution of the dream


The generative camp, Genie 3, Cosmos, GAIA-2, the Waymo World Model, the interactive simulators, predicts the future at the resolution of the camera. Every blade of grass, every specular highlight, every cloud that has no bearing on whether the grasp succeeds. The latent camp, V-JEPA 2 and its action-conditioned variant, DINO-WM, PLDM, the LeJEPA line, predicts the future at the resolution of a learned embedding, on the bet that most pixels are dynamics-irrelevant and that modeling them is a tax paid in capacity, data, and inference time. LeCun has been making this argument since his 2022 position paper. The difference now is that both camps have real robots and real receipts.

My position is that the fight is misframed, because the right resolution is set by *who consumes the prediction*. In Fei-Fei Li's vocabulary this is the renderer-vs-simulator question, and the answer depends on the consumer: humans consume pixel, so if the job is synthetic data generation, rehearsing edge cases, or letting an engineer eyeball a rollout, render away. That's exactly where Cosmos and the Waymo model earn their keep. But a control loop consumes costs, distances, and constraints. A planner that evaluates tens of thousands of candidate futures per executed action cannot afford for each future to be a film. It needs the map, not the territory, and a good map is defined by what it omits.

The two camps even share the shape of their objective. The entire disagreement fits in a subscript:

$$\mathcal{L}_{pixel}=\big\lVert \hat{o}_{t+1}-o_{t+1}\big\rVert^{2}\ \ over\ \mathbb{R}^{3\times H\times W}, \qquad \mathcal{L}_{latent}=\big\lVert g_{\phi}(z_{t},a_{t})-f_{\theta}(o_{t+1})\big\rVert^{2}\ \ over\ \mathbb{R}^{d}.$$

The left spends a million output dimensions repainting clouds. The right spends a few thousand on whatever the encoder decided was worth keeping.

The honest counterargument is that pixels are a supervision signal that's hard to game: a decodable model can be debugged by looking at it, and internet video gives generative models priors that no robot lab's dataset will ever match. That's real, and the strongest recent systems hedge accordingly. V-JEPA 2 pretrains a latent model on a million hours of internet video, then post-trains an action-conditioned predictor on about sixty hours of unlabeled robot video and plans pick-and-place zero-shot on Franka arms in labs it never saw. Latents for the loop, scale from the internet. That recipe, not either purism, is the actual frontier. The lingering question for my own camp is what frozen, appearance-trained encoders fail to encode about contact and force, and whether dynamics-aware perception, trained to notice what control needs noticed, closes that gap. I suspect it matters more than the leaderboards currently show.

## Fault line 2: The number of futures

Here is the dirty secret of most latent world models actually deployed on hardware: their dynamics head is a regression mean-squared or L1 error to the next embedding. This is not a neutral choice, rather a theorem about what the model will become:

$$g^{\star}=\arg\min_{g}\ \mathbb{E} \big\lVert g(z_{t},a_{t})-z_{t+1}\big\rVert_{2}^{2}   \Longrightarrow   g^{\star}(z_{t},a_{t})=\mathbb{E}\big[ z_{t+1}\mid z_{t},a_{t} \big].$$

The optimal regressor is the conditional mean. And when the conditional $p(z_{t+1}\mid z_{t},a_{t})$ is multimodal, its mean sits in a probability desert, a point the world almost never visits.

Robot data, especially the human-collected demonstrations that increasingly feed manipulation learning, is saturated with multimodality. A nudged object goes left or it goes right. A grasp slips or it holds. The average of those futures is not a future at all. It's a ghost, a half-open hand around a half-fallen cup, and a controller steering toward a ghost is optimizing toward a state the world cannot produce. The generative camp internalized this long ago. Video models are diffusion or flow models precisely because the future is a distribution, and the interactive-simulator line picked its samplers on exactly this argument. Dreamer-class models keep a stochastic latent state inside the recurrence for the same reason. But the lean, frozen-encoder, latent-MPC stack that actually ships on arms today still mostly predicts a single mean next state, and the blur it induces falls exactly on the interaction-dependent structure a planner needs most, the difference between outcomes, not their centroid.

The reason this hasn't been fixed by simply transplanting diffusion into the latent space is arithmetic. A sampling-based planner's appetite is multiplicative:

$$model\ queries\ per\ executed\ action = \underbrace{N}_{candidates}\times\underbrace{H}_{horizon}\times\underbrace{I}_{iterations}\times\underbrace{K}_{sampler\ steps}.$$

A TD-MPC2-style loop already sits on the order of $10^{4}$ dynamics evaluations per *control step* at $K=1$, hand the head a fifty-step sampler and the robot stands still, thinking. So today you are quietly asked to choose: a model that is honest about uncertainty, or a model cheap enough to query ten thousand times before the gripper moves. I think this is the most consequential unforced error in the deployed stack, and I'll note only that the generative-modeling community has spent three years collapsing thousand-step samplers into a handful of steps, while a few probabilistic JEPA formulations have started to appear on the other side of the aisle. The two literatures are circling each other. Whoever closes the loop gets a world model that respects the multiplicity of futures without forfeiting the planning budget, and I'd bet on that combination over either purism.

## Fault line 3: When the dreaming happens

There are two times a robot can use its imagination: at training time, and during inference. Dreamer made the first mode famous, roll the model forward in latent space, train an actor-critic inside the dream, deploy the resulting reflex. It put model-based RL in *Nature*, mined diamonds in Minecraft from scratch, and in its DayDreamer incarnation taught a real quadruped to walk in about an hour. The second mode keeps the model in the loop at decision time: TD-MPC2, DINO-WM, V-JEPA 2-AC, the MCTS-flavored visual planners, sample candidate action sequences, roll each through the model, act on the best one, re-plan. In symbols, the deliberative mode solves, at every single step,

$$a^{\star}_{t:t+H-1} = \arg\max_{a_{t:t+H-1}}\ \sum_{k=0}^{H-1}\gamma^{k} \hat{r}\big(\hat{z}_{t+k},a_{t+k}\big) + \gamma^{H} V_{\psi}\big(\hat{z}_{t+H}\big),$$

executes $a^{\star}_{t}$, and throws the rest away. The amortized mode is the degenerate case $H=0$: trust the proposal completely, spend nothing on search, and let the entire future live inside $V_{\psi}$, or inside the policy distilled from it.

The fashionable way to frame this is through the test-time-compute lens borrowed from language models: amortized policies are System 1, planners are System 2, and 2025 was the year LLMs taught everyone that spending inference compute on hard instances buys capability that training compute alone doesn't. I mostly buy the analogy, but the DeepMind theorem sharpens it in a way the discourse hasn't absorbed. If every competent goal-directed agent provably contains a world model, then "model-based vs. model-free" was never the question. The question is whether the model is somewhere you can *query it on purpose*. Whether the knowledge is load-bearing at decision time or fossilized inside a reactive mapping. A policy is a plan that has been compiled. Compilation is a wonderful technology, right up until the input distribution drifts past what the compiler saw.

What frustrates me is how rarely the field tests this distinction cleanly. Planner-vs-policy comparisons in the literature are almost always confounded. Different backbones, different training data, different perception stacks, and search budgets that go unreported, as if the planner's compute were free. Meanwhile the few clean signals we have are tantalizing: hierarchical latent planning recently took a real-robot task that defeats greedy goal-chasing from 0% to 70% success using the *same class* of world model, just by reorganizing how the search spends its budget, and on the other end of the spectrum people are seriously asking whether world-action models need test-time imagination at all once the policy is good enough. Both can't be the whole story. My instinct, and it is the instinct underneath much of what I'm working on, is that the interesting structure lives in *which problems* deliberation changes, not in average margins on benchmarks that reactive policies were already tuned for. I'll leave it there.

## Fault line 4: Whose body is it?

A model trained on one robot learns that robot's physics the way a tailor learns one client's body. Swap the motors, lengthen a limb, hand it a different machine, and the fit is gone, the field's dynamics models are, almost universally, one-body affairs. The mainstream answer to this is brute diversity on the policy side: procedurally generate thousands of bodies in simulation, randomize everything, and train one reactive controller across all of them. It works startlingly well, embodiment-scaling studies show generalization to unseen robots improving steadily with the number of training morphologies, single policies now drive dozens of real quadrupeds and humanoids, and long-context controllers adapt to new bodies on the fly like a temporary system-identification routine.

But notice what that recipe asks the network to do: interpolate in *behavior* space across bodies. And here's my heterodox intuition, physics is polite to interpolation. Optimal behavior is not. Perturb a mass here, a length there, and tomorrow's accelerations change by a little: the dynamics are a continuous function of the body. The best behavior is under no such obligation,

$$\mu  \mapsto  f(s,a;\mu)\ is\ smooth,\qquad \mu  \mapsto  \pi^{\star}_{\mu}=\arg\max_{\pi} J(\pi;\mu)\ need\ not\ be,$$

because $\arg\max$ is not a continuous operator. A centimeter of geometry can snap the best gait, contact schedule, or recovery reflex to something qualitatively different. Functions that vary smoothly are the ones neural networks interpolate gracefully, which suggests that knowledge of the body wants to live in the *dynamics model*, with behavior re-derived against it, rather than baked directly into a policy that must generalize a rougher function. The cross-embodiment world-model results trickling in. Particle-space dynamics that plan across robot hands they never trained on, world-action models adapting to a new robot from half an hour of play data, latent actions serving as an embodiment-agnostic interface, read to me like early confirmation that the model-based route inherits generalization the policy route has to fight for. I'd call this a conjecture the field hasn't tested at the scale it deserves: not "can one policy serve many bodies", which is clearly a yes, but *where the knowledge of the body should be stored* so that the next body is cheap. I know which way I'd bet.

## Fault line 5: The dream as a courtroom

The quietest, fastest-moving corner of this field has nothing to do with control. It's evaluation.

Anyone who has run a real-robot experiment knows the shame of the denominator: the headline number in most manipulation papers rests on a few dozen hand-reset, hand-judged trials, because real rollouts are slow, expensive, unreproducible, and physically risky. Evaluation, not data, is the bottleneck nobody tweets about. So a wave of work now asks the obvious question: if the world model can predict what a policy's actions will do, can it *grade* policies before they ever touch hardware? WorldEval ranks real policies and checkpoints of the same policy entirely in imagination. WorldGym turns a world model into an evaluation environment outright. Google rolled Gemini Robotics policies inside a Veo-based simulator to score them. Benchmark suites like WorldArena now treat policy evaluation as a first-class downstream task for world models, alongside the better-advertised uses in RL post-training and data generation.

I think this is the sleeper application. The first place world models become economically indispensable rather than scientifically fashionable, precisely because the bar is lower in the right way. An evaluator doesn't need to be right about every future. But it needs its *ranking* of policies to correlate with reality, which is a far cheaper contract, the whole thing fits in one statistic,

$$\rho = \operatorname{corr}_{rank}\big(\hat{s}_{imagined},  s_{real}\big),$$

estimable from a handful of real rollouts, which makes the judge auditable in a way the driver never is. But it comes with a built-in honesty problem that deserves more respect than it gets: the judge shares training distribution, and therefore blind spots, with the defendants. A policy that exploits the model's optimism will ace the imagined exam and faceplant on hardware, and the current crop of benchmarks already documents the gap between imagined and realized success. The discipline that makes this trustworthy is unglamorous, measure the rank correlation against real evaluations before you believe anything, watch how prediction error grows with horizon, treat a policy that drives the model off-distribution as a red flag rather than a high score,  and it's the same discipline as fault line 2 wearing a different hat: a model that knows what it doesn't know is worth more than a model that renders beautifully. Use the dream for triage, spend the real robot on the finalists, and never let the courtroom forget it is not the world.

---

## The bets I'm placing

If I compress all of this into a creed, it reads: predict at the resolution of decisions, not of cameras. Respect the multiplicity of futures, the average of two outcomes is often neither. Keep some of the thinking for the moment of action, because a compiled reflex is only as good as its compiler's training set. Teach the model the body, not just the scene, and store what varies across deployments in the dynamics, where physics keeps things smooth. And before the dream drives anything, let it judge, carefully, with its honesty audited against reality.

I'd add one meta-bet about how this field should argue with itself. The 2026 wave is rich in capability demos and poor in controlled comparisons. The papers that move me are the ones that change a single component, hold every confound fixed, name the mechanism they believe is responsible, and state in advance what result would falsify them, including the compute bill, because a planner's search budget is part of the comparison, not an implementation detail. World models are finally important enough to deserve experiments worthy of them. That standard is the one I try to hold my own work to, and the projects I'm being cagey about in this post are, in the end, just these five positions converted into falsifiable form.

---

## A reading map

Not a survey, but a partisan's shelf, organized by fault line, one opinionated line per entry.

### Foundations and theory

<details>
<summary>Click to expand</summary>

- **[World Models](https://arxiv.org/abs/1803.10122)** (Ha & Schmidhuber): the modern starting gun: learn the dynamics, train the behavior inside the dream.
- **[A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf)** (LeCun): the manifesto for predicting in representation space. Agree or not, the whole latent camp descends from it.
- **[General agents contain world models](https://arxiv.org/abs/2506.01622)** (Richens et al.): the theorem that ended the existence debate: competent generalist agents provably contain extractable world models. Reframes everything as a question of access, not possession. Though "contains a model" and "contains a model good enough to plan against" are not the same sentence.
- **[Mastering Diverse Control Tasks Through World Models](https://arxiv.org/abs/2301.04104)** (Hafner et al.): DreamerV3's victory lap. Imagination-trained reflexes, from Atari to Minecraft diamonds.

</details>


### Manifestos worth arguing with

The position-statements the field is actually reacting to. Read them as primary sources. The opinions in the body of this post are largely responses to these.

<details>
<summary>Click to expand</summary>

- **[From Words to Worlds](https://drfeifei.substack.com/p/from-words-to-worlds-spatial-intelligence)** and **[A Functional Taxonomy of World Models](https://drfeifei.substack.com/p/a-functional-taxonomy-of-world-models)** (Fei-Fei Li): the case that spatial intelligence is AI's next frontier, and the renderer/simulator/planner carving that makes "world model" a precise word again. I borrow the taxonomy wholesale and disagree only about which box is the linchpin.
- **[Welcome to the Era of Experience](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)** (Silver & Sutton): the manifesto for grounded, experiential learning over human data. Names the world model as the organ of grounded planning. Bracing diagnosis,  under-specifies who writes the reward.
- **[Robots Need More Than VLAs & World Models](https://arxiv.org/abs/2606.06556)** (Karcini et al.): the cold shower: the bottleneck isn't the policy or the predictor, it's *grounding* messy physical experience into robot-usable supervision. The most useful piece of pushback in the field. Reframes everything else as necessary-but-insufficient.
- **[1X World Model](https://www.1x.tech/discover/1x-world-model)** (1X): the company that said the quiet part first: the real prize is *evaluation*, learned from raw sensor data, with the failure modes shown honestly. The founding document of the model-as-judge thread.

</details>


### The latent camp (JEPA and friends)

<details>
<summary>Click to expand</summary>

- **[V-JEPA 2](https://arxiv.org/abs/2506.09985)** (Assran et al.): internet-scale latent video pretraining. Its action-conditioned variant plans zero-shot on real Franka arms from ~62 hours of unlabeled robot video. The reference point for latent MPC at scale.
- **[DINO-WM](https://arxiv.org/abs/2411.04983)** (Zhou et al.): freeze a foundation encoder, learn only the predictor, plan with CEM. Proof that the encoder and the dynamics can be decoupled.
- **[PLDM: Learning from Reward-Free Offline Data](https://arxiv.org/abs/2502.14819)** (Sobal et al.): the case for planning with latent dynamics when rewards never existed.
- **[LeJEPA](https://arxiv.org/abs/2511.08544)** (Balestriero & LeCun): anti-collapse via a provable isotropic-Gaussian regularizer. The heuristic-free path to end-to-end JEPA training.
- **[stable-worldmodel](https://arxiv.org/abs/2602.08968)** (Maes et al.): reproducible infrastructure for latent world-model research. The field needs more of this and less of one-off codebases.
- **[Navigation World Models](https://arxiv.org/abs/2412.03572)** (Bar et al.): latent prediction meets navigation.
- **[Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208)** (Zhang et al.): multi-scale latent planning takes a real non-greedy manipulation task from 0% to 70%. The cleanest recent evidence that *how* you search a model matters as much as the model.
- **[Causal-JEPA](https://arxiv.org/abs/2602.11389)** (Nam et al.): object-level latent masking for interaction-aware, counterfactually robust dynamics. The object-centric future of this camp.

</details>

### Imagination as a training ground (the Dreamer / MBRL lineage)

<details>
<summary>Click to expand</summary>

- **PlaNet to Dreamer v1–v3** (Hafner et al.): the RSSM dynasty, stochastic latents inside the recurrence were the original answer to fault line 2.
- **[DayDreamer](https://arxiv.org/abs/2206.14176)** (Wu et al.): world models on physical robots, including a quadruped that learned to walk in roughly an hour.
- **[TD-MPC2](https://arxiv.org/abs/2310.16828)** (Hansen et al.): the workhorse of decision-time planning in latent space. Also an honest look at what MPPI costs per control step.
- **[WMP: World-Model-Based Perception for Visual Legged Locomotion](https://arxiv.org/abs/2409.16784)** (Lai et al.): the world model as a perception organ for legs, sim-to-real, no privileged teacher.
- **[WorldPlanner](https://arxiv.org/abs/2511.03077)** (Khorrambakht et al.): MCTS and MPC over action-conditioned visual world models. Search algorithms are back.

</details>

### Generative world simulators

<details>
<summary>Click to expand</summary>

- **Genie 1 to 3** (DeepMind): from 2D latent-action curiosities to publicly explorable, promptable 720p worlds at 24 fps. The cultural event of the wave.
- **Cosmos to Cosmos 3** (NVIDIA): world foundation models as an industrial platform: synthetic data, Sim2Real transfer, and now unified generation–reasoning–action.
- **GAIA-1 and GAIA-2** (Wayve): the trailblazers of generative driving worlds.
- **The Waymo World Model**: Genie 3 post-trained to emit synchronized camera and lidar. Rehearsing edge cases the fleet has never seen. World models as institutional safety infrastructure.
- **[Interactive World Simulator](https://www.yixuanwang.me/interactive_world_sim/texts/main.pdf)** (Wang et al.): consistency-model dynamics for stable, action-conditioned rollouts at interactive rates; the pixel camp's answer to the sampling-cost problem.

</details>

### World models meet VLAs (the WAM convergence)

<details>
<summary>Click to expand</summary>


- **[DreamZero: World Action Models are Zero-shot Policies](https://dreamzero0.github.io/)** (Ye et al.): a 14B video-diffusion backbone jointly predicting futures and actions, squeezed to 7 Hz closed-loop control. Transfers to a new robot from 30 minutes of play data. Whatever else it is, it's a statement.
- **Unified Video Action / UWM / WorldVLA** (2025): coupling video and action generation so the model can skip the rendering when it only needs to act.
- **[Fast-WAM: Do World Action Models Need Test-Time Future Imagination?](https://arxiv.org/abs/2603.16666)** (Yuan et al.): asks fault line 3's question from the opposite direction. Read it next to the hierarchical-planning result and feel the tension.
- **[VLA-JEPA](https://arxiv.org/abs/2602.10098)** (Sun et al.): latent world-modeling objectives as pretraining for VLA policies. The camps are interbreeding.
- **[World-Value-Action models](https://arxiv.org/abs/2604.14732)** (Li et al.): implicit planning: fold the search into the architecture and let inference concentrate on high-value futures.

</details>


### The model as judge

<details>
<summary>Click to expand</summary>


- **[1X World Model](https://www.1x.tech/discover/1x-world-model)** (1X): the industrial precursor (full entry under Manifestos): evaluation as the real prize, learned from sensor data.
- **[WorldEval](https://arxiv.org/abs/2505.19017)** (Li et al.): rank policies, and checkpoints of one policy, entirely in imagination. The founding research document of the evaluator thread.
- **[WorldGym](https://arxiv.org/abs/2506.00613)** (Quevedo et al.): the world model *as* the evaluation environment.
- **[Evaluating Gemini Robotics Policies in a Veo World Simulator](https://arxiv.org/abs/2512.10675)** (DeepMind): frontier-lab endorsement of imagined evaluation.
- **[WorldArena](https://world-arena.ai/)** (Shang et al.): benchmarks the judges themselves, and documents how far imagined verdicts still sit from real ones. Required reading before trusting any imagined success rate.
- **[World models with calibrated uncertainty](https://arxiv.org/abs/2512.05927)** (Mei et al.): the discipline that makes the judge trustworthy: a model that knows when it doesn't know can flag its own off-distribution verdicts and catch a VLA's failures before hardware does.

</details>

### Bodies

<details>
<summary>Click to expand</summary>

- **[Towards Embodiment Scaling Laws in Robot Locomotion](https://arxiv.org/abs/2505.05753)** (Ai et al.): ~1,000 training morphologies, zero-shot to real Go2 and H1. The strongest case for the brute-diversity route.
- **[Multi-Embodiment Locomotion at Scale](https://arxiv.org/abs/2509.02815)** (Bohlinger & Peters): embodiment randomization pushed to millions of bodies per run.
- **[LocoFormer](https://arxiv.org/abs/2509.23745)** (Liu et al.): long-context adaptation as implicit system identification. The policy-side answer to "whose body is it."
- **[Scaling Cross-Embodiment World Models for Dexterous Manipulation](https://arxiv.org/abs/2511.01177)** (He et al.): particle-space states and actions unify hands of different morphologies under one dynamics model planned by MPC. The model-based route, vindicated early.

</details>

### Actions without labels

<details>
<summary>Click to expand</summary>

- **LAPO / Genie / LAPA / UniVLA**: the latent-action lineage: discover an action space from video alone, then ground it.
- **[Learning Latent Action World Models in the Wild](https://arxiv.org/abs/2601.05230)** (Garrido et al.): latent actions as a universal interface for planning on uncurated video. The label bottleneck is dissolving.

</details>

### The skeptics' shelf

<details>
<summary>Click to expand</summary>

- **[How Far Is Video Generation from World Model: A Physical Law Perspective](https://arxiv.org/abs/2411.02385)** (Kang et al.): scaled video models nail in-distribution physics and fail out-of-distribution, behaving like case-based mimics rather than law-learners. The single most clarifying negative result in this literature.
- **PhyGenBench / VACT / WorldArena's functional metrics**: the growing toolkit for catching beautiful, physically wrong dreams.
- **[Is Sora a World Simulator?](https://arxiv.org/abs/2405.03520)** (Zhu et al.): realism is not reality. A survey-length meditation on the difference.

</details>

### Maps of the territory

<details>
<summary>Click to expand</summary>


- **[World Model for Robot Learning: A Comprehensive Survey](https://arxiv.org/abs/2605.00080)** (Hou et al.): the current best aerial photograph of the field: world models for policy, as simulators, and as video generators, current through early 2026. Read it for coverage, read essays like this one for opinions.

</details>

---

*If you disagree with any of the five positions above, good. Three of them are conjectures wearing confident clothing, and the fastest way to find out which is to build the experiment that breaks them. That's roughly what I'm doing. More soon.*