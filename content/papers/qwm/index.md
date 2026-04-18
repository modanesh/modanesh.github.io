---
title: "Toward Hardware-Agnostic Quadrupedal World Models via Morphology Conditioning"
date: 2026-04-09
author: ["Mohamad H. Danesh", "Chenhao Li", "Amin Abyaneh", "Anas Houssaini", "Kirsty Ellis", "Glen Berseth", "Marco Hutter", "Hsiu-Chin Lin"]
description: "QWM enables a single neural world model to control diverse quadrupedal robots without retraining, via explicit morphology conditioning on engineering specifications."
tags: ["robotics", "world models", "quadrupeds", "locomotion", "zero-shot generalization"]
showToc: false
disableAnchoredHeadings: true
hideMeta: true
---

<style>
.qwm-page {
  max-width: 860px;
  margin: 0 auto;
  font-family: inherit;
}
.qwm-title {
  text-align: center;
  font-size: 1.6rem;
  font-weight: 700;
  line-height: 1.3;
  margin-bottom: 0.6rem;
}
.qwm-venue {
  text-align: center;
  font-size: 1rem;
  color: var(--secondary);
  margin-bottom: 1.2rem;
}
.qwm-authors {
  text-align: center;
  font-size: 0.95rem;
  line-height: 1.8;
  margin-bottom: 0.4rem;
}
.qwm-affiliations {
  text-align: center;
  font-size: 0.82rem;
  color: var(--secondary);
  margin-bottom: 1.4rem;
}
.qwm-links {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  justify-content: center;
  margin-bottom: 2.5rem;
}
.qwm-link-btn {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.4rem 1rem;
  border: 1px solid var(--border);
  border-radius: 4px;
  font-size: 0.88rem;
  text-decoration: none !important;
  color: inherit !important;
  transition: background 0.15s;
}
.qwm-link-btn:hover {
  background: var(--entry);
}
.qwm-link-btn.disabled {
  opacity: 0.45;
  pointer-events: none;
}
.qwm-section {
  margin-bottom: 2.5rem;
}
.qwm-section h2 {
  font-size: 1.15rem;
  font-weight: 700;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0.3rem;
  margin-bottom: 1rem;
}
.qwm-abstract {
  font-size: 0.95rem;
  line-height: 1.7;
  text-align: justify;
}
.qwm-fig {
  margin: 1.2rem 0;
  text-align: center;
}
.qwm-fig img {
  max-width: 100%;
  border-radius: 4px;
}
.qwm-fig figcaption {
  font-size: 0.82rem;
  color: var(--secondary);
  margin-top: 0.5rem;
  line-height: 1.5;
}
.qwm-video-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}
.qwm-video-placeholder {
  background: var(--entry);
  border: 1px dashed var(--border);
  border-radius: 6px;
  padding: 2.5rem 1rem;
  text-align: center;
  font-size: 0.85rem;
  color: var(--secondary);
}
.qwm-video-placeholder .qwm-vid-icon {
  font-size: 2rem;
  display: block;
  margin-bottom: 0.5rem;
}
.qwm-results-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 1rem;
}
@media (max-width: 600px) {
  .qwm-results-grid { grid-template-columns: 1fr; }
}
.qwm-highlight {
  background: var(--entry);
  border-left: 3px solid var(--primary);
  padding: 0.8rem 1.1rem;
  border-radius: 0 4px 4px 0;
  font-size: 0.9rem;
  margin: 1rem 0;
}
.qwm-bibtex {
  background: var(--entry);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 1rem 1.2rem;
  font-family: monospace;
  font-size: 0.82rem;
  line-height: 1.6;
  overflow-x: auto;
  white-space: pre;
}
.qwm-divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 2rem 0;
}
</style>

<div class="qwm-page">

<div class="qwm-title">Toward Hardware-Agnostic Quadrupedal World Models<br>via Morphology Conditioning</div>
<div class="qwm-venue">arXiv 2026</div>

<div class="qwm-authors">
  Mohamad H. Danesh&nbsp;&nbsp;·&nbsp;&nbsp;Chenhao Li&nbsp;&nbsp;·&nbsp;&nbsp;Amin Abyaneh&nbsp;&nbsp;·&nbsp;&nbsp;Anas Houssaini<br>
  Kirsty Ellis&nbsp;&nbsp;·&nbsp;&nbsp;Glen Berseth&nbsp;&nbsp;·&nbsp;&nbsp;Marco Hutter&nbsp;&nbsp;·&nbsp;&nbsp;Hsiu-Chin Lin
</div>
<div class="qwm-affiliations">
  McGill University &nbsp;·&nbsp; ETH Zürich &nbsp;·&nbsp; Université de Montréal / Mila
</div>

<div class="qwm-links">
  <a class="qwm-link-btn" href="https://arxiv.org/abs/2604.08780" target="_blank">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
    arXiv
  </a>
  <a class="qwm-link-btn" href="https://arxiv.org/pdf/2604.08780" target="_blank">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
    PDF
  </a>
  <a class="qwm-link-btn disabled" href="#" title="Code coming soon">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
    Code (coming soon)
  </a>
</div>

<hr class="qwm-divider">

<div class="qwm-section">
<h2>Abstract</h2>
<p class="qwm-abstract">
World models trained on one quadrupedal platform typically fail on different hardware due to morphological differences in mass, link dimensions, and kinematic configuration. We present <strong>QWM</strong>, a framework that enables a single neural dynamics model to generalize across diverse quadrupedal robots without retraining. The key innovation is to <em>explicitly condition</em> the generative dynamics on the robot's engineering specifications — extracted directly from URDF/USD files — rather than inferring physical properties implicitly from interaction history. A Physical Morphology Encoder (PME) derives a compact embedding from kinematic, geometric, dynamic, and actuation features, which is injected into every recurrent step of a DreamerV3-based world model. An Adaptive Reward Normalizer (ARN) handles heterogeneous reward scales across platforms. We further introduce <strong>Hetero-Isaac</strong>, an extension to NVIDIA Isaac Lab enabling true heterogeneous training across different morphologies in parallel. QWM achieves zero-shot locomotion on unseen robots — including real-world deployment on Unitree Go1 and ANYmal-D — with performance approaching per-robot specialists, while eliminating the dangerous adaptation lag of implicit system identification approaches.
</p>
</div>

<div class="qwm-section">
<h2>Overview</h2>

<figure class="qwm-fig">
  <img src="https://arxiv.org/html/2604.08780/2604.08780v1/x1.png" alt="QWM framework overview">
  <figcaption><strong>Figure 1.</strong> Overview of the QWM framework. Left (WM Learning): A single world model is trained across diverse morphologies. The Physical Morphology Encoder (PME) derives a static embedding μ from each robot's USD file, which explicitly conditions both the encoder and the recurrent dynamics. Right (Policy Learning): A policy is trained purely in imagination and deployed zero-shot on real hardware.</figcaption>
</figure>

<figure class="qwm-fig">
  <img src="https://arxiv.org/html/2604.08780/2604.08780v1/x2.png" alt="Heterogeneous robot cohort">
  <figcaption><strong>Figure 2.</strong> The heterogeneous morphology cohort used in experiments, illustrating the variance in physical scale and configuration. QWM is trained on seven robots while holding out one for zero-shot evaluation.</figcaption>
</figure>
</div>

<div class="qwm-section">
<h2>Method</h2>

<p>QWM extends DreamerV3 with three targeted architectural changes to handle cross-morphology generalization:</p>

<div class="qwm-highlight">
  <strong>Physical Morphology Encoder (PME)</strong> — Extracts normalized features across four categories: kinematics &amp; topology (hip offset, limb lengths, knee configuration), geometry (stance dimensions), dynamics (log-scaled mass), and actuation (torque density). Processed by a dedicated 2-layer MLP that runs parallel to the proprioceptive encoder, preventing static context from being overwhelmed by dynamic signals.
</div>

<div class="qwm-highlight">
  <strong>Morphology-Conditioned Recurrent Dynamics</strong> — The morphology embedding μ is injected at every recurrent step: <em>h<sub>t</sub> = f(h<sub>t−1</sub>, z<sub>t−1</sub>, a<sub>t−1</sub>, μ)</em>. This allows the recurrent state to focus on dynamic execution while explicit conditioning handles static embodiment properties.
</div>

<div class="qwm-highlight">
  <strong>Adaptive Reward Normalizer (ARN)</strong> — Quantile-based scaling using exponential moving averages tracks per-robot reward distributions, dynamically normalizing heterogeneous reward signals so no single morphology dominates training.
</div>

</div>

<div class="qwm-section">
<h2>Real-World Experiments</h2>

<p>Both Unitree Go1 and ANYmal-D were <em>held out during training</em>. By injecting the correct morphology embedding, the frozen policy achieves stable locomotion on both platforms with zero falls across 20 trials (10 per platform, 60 seconds each).</p>

<figure class="qwm-fig">
  <img src="https://arxiv.org/html/2604.08780/2604.08780v1/x5.png" alt="Real-world deployment on Unitree Go1 and ANYmal-D">
  <figcaption><strong>Figure 5.</strong> Real-world deployment on Unitree Go1 and ANYmal-D. Both robots were held out during training. The frozen policy achieves stable zero-shot locomotion by simply injecting the correct morphology embedding μ.</figcaption>
</figure>

<p>Videos from real-world experiments:</p>

<div class="qwm-video-grid">
  <div class="qwm-video-placeholder">
    <span class="qwm-vid-icon">▶</span>
    <strong>ANYmal-D Deployment</strong><br>
    Zero-shot locomotion on held-out platform
    <br><br><em>(video coming soon)</em>
  </div>
  <div class="qwm-video-placeholder">
    <span class="qwm-vid-icon">▶</span>
    <strong>Unitree Go1 Deployment</strong><br>
    Zero-shot locomotion on held-out platform
    <br><br><em>(video coming soon)</em>
  </div>
  <div class="qwm-video-placeholder">
    <span class="qwm-vid-icon">▶</span>
    <strong>Multi-Robot Training</strong><br>
    Hetero-Isaac: 8 robots training in parallel
    <br><br><em>(video coming soon)</em>
  </div>
  <div class="qwm-video-placeholder">
    <span class="qwm-vid-icon">▶</span>
    <strong>Open-Loop Imagination</strong><br>
    Long-horizon dynamics prediction rollouts
    <br><br><em>(video coming soon)</em>
  </div>
</div>
</div>

<div class="qwm-section">
<h2>Results</h2>

<figure class="qwm-fig">
  <img src="https://arxiv.org/html/2604.08780/2604.08780v1/x3.png" alt="Learning curves on heterogeneous robot cohort">
  <figcaption><strong>Figure 3.</strong> Learning curves comparing QWM against world model baselines (DreamerV3, PWM, TWISTER) trained simultaneously on the full heterogeneous cohort. QWM achieves significantly faster convergence and higher stability. Shaded regions are standard deviation across 5 seeds.</figcaption>
</figure>

<figure class="qwm-fig">
  <img src="https://arxiv.org/html/2604.08780/2604.08780v1/x4.png" alt="Long-horizon dynamics prediction">
  <figcaption><strong>Figure 4.</strong> Long-horizon dynamics prediction. Left: Open-loop imagination rollouts vs. ground truth physics. QWM maintains tight synchronization across diverse scales. Right: Normalized Mean Squared Error (NMSE) over a 45-step horizon — QWM consistently outperforms baselines with minimal error accumulation.</figcaption>
</figure>

<div class="qwm-results-grid">
  <div>
    <figure class="qwm-fig">
      <img src="https://arxiv.org/html/2604.08780/2604.08780v1/x7.png" alt="Ablation study">
      <figcaption><strong>Figure 7.</strong> Ablation study isolating contributions of PME, ARN, and conditioning locations.</figcaption>
    </figure>
  </div>
  <div>
    <figure class="qwm-fig">
      <img src="https://arxiv.org/html/2604.08780/2604.08780v1/x8.png" alt="PCA of QWM latent states">
      <figcaption><strong>Figure 8.</strong> PCA of QWM latent states — morphology clusters (a) vs. dynamic state gradients (b–e), showing the model cleanly separates embodiment identity from locomotion dynamics.</figcaption>
    </figure>
  </div>
</div>

<figure class="qwm-fig">
  <img src="https://arxiv.org/html/2604.08780/2604.08780v1/x6.png" alt="Morphological feature distance matrix">
  <figcaption><strong>Figure 6.</strong> Morphological Feature Distance Matrix. Euclidean distances between z-score standardized PME features, showing the encoder correctly identifies physical families (e.g., ANYmal variants cluster together).</figcaption>
</figure>
</div>

<div class="qwm-section">
<h2>Zero-Shot Generalization</h2>

<p>QWM is evaluated in two generalization regimes:</p>

<ul>
  <li><strong>Morphological Interpolation</strong> (within training distribution): Unitree Go1 achieves 974.4 ± 6.2 episode length vs. 996.1 ± 1.1 for the specialist. ANYmal-D achieves 948.6 ± 12.1 vs. 981.3 ± 4.2.</li>
  <li><strong>Real-World Transfer</strong>: ANYmal-D reaches 0.30 m/s linear tracking error vs. 0.28 for specialist; Go1 reaches 0.34 vs. 0.31. Zero falls across all 20 trials.</li>
  <li><strong>Morphological Extrapolation</strong> (out of distribution): Performance degrades for geometric outliers (e.g., Unitree B2), confirming QWM acts as a distribution-bounded interpolator — a universal physics engine requires cohorts that span the full parameter space.</li>
</ul>
</div>

<hr class="qwm-divider">

<div class="qwm-section">
<h2>BibTeX</h2>
<div class="qwm-bibtex">@misc{danesh2026qwm,
  title     = {Toward Hardware-Agnostic Quadrupedal World Models
               via Morphology Conditioning},
  author    = {Danesh, Mohamad H. and Li, Chenhao and Abyaneh, Amin
               and Houssaini, Anas and Ellis, Kirsty and Berseth, Glen
               and Hutter, Marco and Lin, Hsiu-Chin},
  year      = {2026},
  eprint    = {2604.08780},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url       = {https://arxiv.org/abs/2604.08780}
}</div>
</div>

</div>
