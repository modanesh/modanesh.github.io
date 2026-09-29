---
title: "Papers"
description: "Preprints and articles by Professor Dr von Igelfeld."
---

<style>
.scholar-callout {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  padding: 0.85rem 1.1rem;
  margin: 0 0 1.6rem;
  background: rgba(106, 123, 162, 0.12);
  border: 1px solid var(--lightcolor);
  border-left: 4px solid var(--darkcolor);
  border-radius: var(--radius);
  font-size: 0.95rem;
  line-height: 1.5;
}
.scholar-callout svg {
  flex-shrink: 0;
  color: var(--darkcolor);
}
.scholar-callout a {
  font-weight: 700;
  color: var(--darkcolor) !important;
  box-shadow: 0 1px 0 var(--darkcolor);
}
.dark .scholar-callout {
  background: rgba(106, 123, 162, 0.25);
  border-color: var(--darkcolor);
}
.dark .scholar-callout svg,
.dark .scholar-callout a {
  color: var(--lightcolor) !important;
}
.dark .scholar-callout a {
  box-shadow: 0 1px 0 var(--lightcolor);
}
/* external (off-site) paper cards: reset the generic .post-content
   tag rules that would otherwise leak in, since these cards live
   inside the page's own Markdown content rather than the themeʼs
   auto-generated list loop */
#ext-papers .post-entry { margin-bottom: var(--gap); }
#ext-papers h2 { margin: 0; font-size: 20px; line-height: var(--lineheight); }
#ext-papers p { margin: 0; }
#ext-papers img { margin: 0; }
#ext-papers a { box-shadow: none; color: currentColor; }
</style>

<div class="scholar-callout">
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 10L12 5 2 10l10 5 10-5z"/><path d="M6 12v5c0 1.7 2.7 3 6 3s6-1.3 6-3v-5"/></svg>
  <span>These are my most recent papers as first or corresponding author. For the full list, check my <a href="https://scholar.google.com/citations?user=AsqcJtAAAAAJ&hl=en">Google Scholar</a>.</span>
</div>

<div id="ext-papers">
<article class="post-entry">
  <figure class="entry-cover"><img loading="lazy" src="/papers/driftql-fig2.png" alt="Overview of DriftQL: sampling states and noise, forming positives and negatives, computing the conditional drift field, and the drift plus Q-learning loss"></figure>
  <header class="entry-header">
    <h2 class="entry-hint-parent">Drift Q-Learning</h2>
  </header>
  <div class="entry-venue">NeurIPS 2026</div>
  <div class="entry-content">
    <p>DriftQL learns a single drift field, balancing attraction toward the dataset with repulsion for diversity, so an offline policy can act in one forward pass, no denoising chain or ODE solver required. It matches diffusion and flow policies on D4RL and OGBench and stays far more robust when the data is corrupted.</p>
  </div>
  <a class="entry-link" aria-label="post link to Drift Q-Learning" href="https://driftql.github.io/" target="_blank" rel="noopener"></a>
</article>
</div>
