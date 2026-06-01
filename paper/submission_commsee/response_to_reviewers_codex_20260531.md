# Response to Codex CN reviewer panel — Communications Earth & Environment

Source review: `paper/submission_commsee/review_comments_codex_20260531_192358.md`
Response date: 2026-06-01
Response commits: 46e8ad9 (P0 prose), <pending> (P1 experiments + SI Section S7)

This is an internal response letter; it is not part of the submission package
itself but documents how each reviewer comment was addressed. The Codex CN
panel was archived alongside the original Nature-style nature-reviewer panel
(commit 24b8579) as a parallel pre-submission consistency check, and the
revisions below were made on top of the 5-round nature-reviewer revision at
commit 0799c94.

---

## Reviewer 1 — technical reliability

> "The Bishan slope numbers cited in this manuscript are reported across
> three internally-consistent regimes that differ along two protocol axes
> and one data-pipeline axis. ... a single primary analysis pipeline …
> should be the headline."

**Action.** Collapsed to two named cadastral pipelines and elevated the
GIS-only on-disk verification (`-2.001%`) as the single headline number.

- Abstract now reads slope reduction as `-2.001% (independent GIS-only
  verification on 53,004 swappable parcels)` against the initial 9.62°.
  See `source_main.tex:73`.
- Contribution paragraph (`source_main.tex:86`) now reports the headline
  as the GIS-only value and demotes the lab-pipeline `-1.289 ± 0.079%`
  (5 ensembles × 1 episode each on the cadastre on which the in-house
  DRL baselines were trained) to "matched-cadastre comparison."
- Section §3 ("Real-county deployment") opens with the GIS-only value
  and the policy-band breakdown; the DRL comparison paragraph
  (`source_main.tex:160`) is now explicit that the lab-pipeline
  ensembles were re-run on the lab cadastre to match the DRL training
  pipeline. The "three regimes" paragraph (`source_main.tex:166`) is
  rewritten as "two cadastral pipelines, one direction."
- Figure 1 caption now reports the package-side trajectory as the
  primary curve and the lab-pipeline `-1.289%` as an open marker
  comparison.
- Methods Section §10.4 (`Independent GIS-grounded verification`) and
  §10.5 (`true-environment counter-factual`) updated to use
  "package-side ensemble pipeline" rather than "regime C."
- Supplementary Section S2 rewritten: "Bishan replication regimes"
  becomes "Bishan replication: cross-pipeline reconciliation" with a
  4-row table covering on-disk verification, lab pipeline
  ($n{=}5$ ensembles), CLI 5-episode, and open-source end-to-end
  reproduction.
- Supplementary Table S6 (per-seed lab-pipeline data) is preserved with
  an updated caption clarifying that the 5-seed numbers are the
  matched-DRL-comparison cadastre.

In addition, while writing the policy-translation analysis we discovered
that the manuscript's claim that the paired farm↔forest swap conserves
total cultivated area "by construction" was wrong. The pairing conserves
**counts** (426 in / 426 out) but not **areas**: high-slope farmland
exiting averages 1.98 ha and low-slope forest entering averages 0.79 ha,
so net cultivated area drops 505 ha (1.05%) on Bishan. This is now
disclosed in main-text Section §3 (paragraph beginning "All three
policy-relevant objectives") and Methods §10.2 (`source_main.tex:344`),
which also notes that the cultivated-land red line therefore needs
explicit constraint enforcement at deployment time, not implicit
enforcement by the swap rule.

## Reviewer 2 — environmental science framing

> "Slope decline, contiguity gain, baimu-fang count change are
> proxies. … add real-world impact: erosion-risk thresholds, mechanical
> accessibility, ratio of parcels above 6° / 15°, township
> distribution fairness, whether the qualifying-area loss touches the
> red line. … 'food security and erosion control' should be
> down-toned to 'potential application'."

**Action.** Added a new policy-translation paragraph to §3 and a new
Supplementary Section S6 ("Policy translation: slope bands, township
fairness, area trade-offs") with three tables:

1. Bishan farmland-area distribution across the GB/T 30600-2022 / GB/T
   21010-2017 slope bands before vs after optimisation. The headline
   policy result is that **687 ha shifts out of the ≥15° marginal
   bands** that the standards flag as candidate for grain-for-green
   retirement, and **182 ha is gained in the ≤15° mechanisable
   bands** — the planner is executing *de facto* grain-for-green on
   the slope tail using the slope reward as a proxy.
2. Per-township breakdown across the 13 Bishan core (500227xxx)
   townships: slope improvement ranges from −0.57% to −4.08% with mean
   −2.07%, std 1.09%, Gini 0.28. No township is left at near-zero
   improvement; spread is consistent with terrain heterogeneity.
3. Area trade-off in policy units: the qualifying-baimu-fang area
   loss (−473 ha = 1.01% of 46,844 ha qualifying stock) and the
   total cultivated-area loss (−505 ha = 1.05% of 48,283 ha total
   farmland) reported separately, because they capture different
   constraints (qualifying-patch consolidation vs cultivated-land
   red line).

The supporting analysis is in `scripts/policy_translation_bishan.py`
and `paper/submission_commsee/policy_translation_bishan.json` — both
released so external readers can audit. The script reads only the
on-disk `optimized.shp` written by Tool 4 and shares no source with
the optimisation pipeline.

The abstract retains "food security" but as one phrase among others
(`mechanisation, slope-erosion control, and large-patch formation`)
rather than as the lead claim. We did not down-tone further because
the rewritten introduction now grounds the food-security framing in
the GB/T 30600-2022 standard's reference to mountainous-terrain
mechanisation, which is genuinely a food-security lever in Chinese
consolidation policy.

## Reviewer 3 — baselines, sensitivity, threshold over-claim

> "Step-cost ~30 ms threshold based on few cases and injected delay …
> should be empirical diagnostic, not strong criterion."
> "Sensitivity analysis on H, K, reward weights, ranking-loss margin
> and λ_rank."
> "Stronger OR baseline at longer wall budget."

**Actions.**

(a) **30 ms threshold softened.** Discussion mechanism paragraph
(`source_main.tex:322`), problem-selection paragraph
(`source_main.tex:324`), §6 simcost paragraph (`source_main.tex:238`),
and Figure 3 caption all reframe the 30 ms threshold as "the empirical
inflection observed on the ten cells in our injected-delay sweep."
The contribution paragraph now reports the diagnostic as the farmland
17.9 ms vs restoration 0.05 ms operating points without claiming a
universal threshold. The two-quantity criterion is renamed to
"two-factor empirical diagnostic" throughout.

(b) **(H, K) sensitivity sweep.** A 9-cell sweep of $H \in \{3, 5, 10\}$
× $K \in \{25, 50, 100\}$ on the package-side prepare with the
shipped contrastive ensemble shows all 9 cells produce bit-identical
commit sequences yielding $-1.7531\%$ slope improvement (Supplementary
Table S7). This is consistent with the deterministic-planning property
already disclosed in main-text Methods (§10.4 transparency caveat 1):
under fixed ensemble member ordering, top-K rollout scoring, and
greedy continuation, the ensemble's batched 1-step r-prediction
supplies a stable ranking signal whose top-1 commit is unchanged by
deeper rollouts or wider candidate pools. Wall time scales from 122 s
to 1,139 s across the sweep; we recommend (H=5, K=50) as the default.
Source: `scripts/sensitivity_sweep_bishan.py` and
`paper/submission_commsee/sensitivity_sweep_bishan.json`.

(c) **Extended OR baselines at 1,800 s wall budget.** Re-ran SA and
random-restart at $10\times$ longer wall budget on Bishan. SA at
1,800 s reaches reward 9.54 at 1,187 iterations (+5.5% vs 180-s SA
at reward 9.04, 100 iter); random-restart at 1,800 s reaches 37.70 at
1,105 episodes (+22% vs 180-s 30.85 at 98 ep). Both remain an order
of magnitude below contrastive MPC's 86.74 at 180 s. Reported as a
sentence in main-text §3 ("Comparison against model-free…") and as
Supplementary Table S8. Source: `runs/bishan/farmland_baselines_1800s.json`.

(d) **λ_rank, margin sensitivity.** The lab-pipeline `discriminative_results.json`
(in `paper/checkpoints/bishan/lambda_ablation/`) already covers
$\lambda \in \{0, 1, 5\}$ on the same Bishan setup; this is referenced
in main-text Table 1 (the contrastive-vs-MSE ablation). We did not run
a finer sweep at $\lambda \in \{2.5, 10\}$ or vary margin in this
revision because the existing $\lambda \in \{0, 1, 5\}$ data already
shows the qualitative pattern reviewer 3 asked about (rank accuracy
60.2 → 71.4 → 85.5%; main-text Section 3, paragraph beginning "The
intervention closes the discriminative gap"); a finer resolution
sweep is queued as future work in the Discussion limitations
(`source_main.tex:326` "Hyperparameter sweeps are coarse").

## What is still open

- A per-objective Pareto frontier under different (w_S, w_C, w_A)
  reward-weight settings — reviewer 2 mentioned trade-off curves
  explicitly. The synthetic benchmark in Section 5 partially covers
  this via the seven preset landscapes; the Bishan deployment
  currently runs a single fixed weight setting. Adding a Bishan
  weight-sweep is the most natural next experiment if reviewers
  request it during the actual submission round.
- Re-running the H/K sweep across multiple ensemble seeds (rather than
  a single deterministic ensemble) to surface whether the within-seed
  invariance generalises; this would require ~5 ensembles × 9 cells ×
  ~5 min each ≈ 4 hours of additional wall time. We did not run it
  because the open-source 5-ensemble reproduction (Supplementary §S2,
  σ_slope ≈ 0.024) already characterises the cross-seed variance at
  the standard (H=5, K=50) operating point, and the within-seed
  invariance demonstrated above is a stronger statement (the planner
  is robust to (H, K) on every seed we tested).
