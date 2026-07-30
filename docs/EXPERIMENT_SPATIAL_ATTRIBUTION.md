# Experiment: Does weakly-supervised attention recover spatial biology?

**Status:** v2 redesign, revised 2026-07-30.

Corpus is complete: **421/421 human Visium slides** have CTransPath features and
depth-aware pathway targets (`format_version 3`); H-Optimus-0 features exist for
the 73 Bowel slides only. Pipeline validated against the published
HEST-Benchmark. §2 records the v1 assumptions that measurement falsified; §6
records why the backbone choice is load-bearing.

**Not yet run:** the model matrix itself. Outstanding build items are in §10.

## 1. The question

> Can a *weakly-supervised* (slide/bag-level) model resolve the **actual
> spatial distribution of pathway activity** through its attention — and does
> **direct spatial supervision** improve on this?

Weakly-supervised pathology models (MIL; SURVPATH for survival) routinely
visualise attention spatially and argue it highlights the biologically relevant
regions. None of them could *verify* that claim: they had no spatially-resolved
ground truth. We do. This connects to the *Attention is/is-not Explanation*
debate (Jain & Wallace 2019; Wiegreffe & Pinter 2019) by grounding attention
validity in a biological signal rather than a proxy task.

## 2. What v1 got wrong (measured, not assumed)

Five findings from the review. Each is reproducible via the scripts named.

**(a) The shared component is sequencing depth, not "tissue density."**
`|r(PC1, log total_counts)| = 0.93` median across slides (0.91 vs genes
detected). Depth is itself spatially autocorrelated (Moran's I 0.76), which is
*why* every pathway inherits a high raw Moran's I (~0.61). The old scorer's
docstring claimed it was "depth-normalised via the prior CP10k step" — CP10k
fixes the per-spot *total*, but the mean over a gene subset is still driven by
dropout, so that claim was false.

**(b) PC1 removal is the wrong instrument.** Compared against regressing out
the *measured* depth covariate:

| | variance destroyed | residual depth correlation |
| --- | --- | --- |
| PC1 removal | 63.0% | 0.077 |
| depth regression | 51.3% | **0.000** |

Depth regression removes the confound completely while preserving ~12
percentage points more variance. PC1 is an unsupervised direction that merely
*correlates* with depth, so it also destroys biology aligned with it — and
since tumour/stroma tracks cell density tracks depth, that plausibly includes
the dominant biological axis.

**(c) The v1 headline contrast was close to tautological.** v1 compared
`stf_dense[prediction]` (a directly supervised dense output) against
`stf_weak[attention]` (an unsupervised internal mechanism). The supervised
output was always going to win; that conflates *supervision* with *readout*.
See §5 for the corrected contrasts.

**(d) Absolute fidelity numbers are uninterpretable on their own.** Three
distinct quantities must not be conflated:

| quantity | value | meaning |
| --- | --- | --- |
| measurement reliability | 0.67 | how well the target reproduces itself (split-half, `measure_target_reliability.py`) |
| **achievable** | **0.28 within-study / 0.19 cross-study** | what a strong linear baseline reaches (`baseline_pca_ridge.py`, H-Optimus-0) |
| observed | — | what a given model scores |

Split-half reliability (binomial UMI split, both halves scored, Spearman-Brown
corrected) is **0.85 for raw maps and 0.67 for depth-residualised maps** — so
roughly a third of the pathway-specific variance is sampling noise. But
reliability is an upper bound on *measurement*, not on *predictability from
morphology*: the achievable figure is far lower. **Normalise reported fidelity
by achievable, not by reliability**, and quote the baseline alongside every
model number.

**(e) The selected pathways are ~6 axes, not 14.** Participation ratio 6.1;
EMT↔MYOGENESIS r=0.78, IFN-γ↔IFN-α r=0.77, E2F↔G2M r=0.64. Averaging over 14
as if independent overstates precision.

**Also corrected:** `pathway_morans_i` stored in each `.h5` is the **raw,
zero-clipped** Moran's I, not the residual — it cannot shortcut pathway
selection (`scripts/select_residual_pathways.py` recomputes properly).

## 3. Scope: all human Visium, not Bowel-only

v1 used 73 Bowel/Visium slides across 10 studies — 5 of them singletons, with
COLON MAP at 56% of the corpus. That is structurally underpowered: a two-sided
Wilcoxon on *n* paired folds cannot return p below 2^-(n-1), so with n<6 folds
significance is unreachable *regardless of effect size*.

**v2 uses all human Visium: 421 samples, 61 studies, 16 organs.** LOSO becomes
n=61 (floor ~0), COLON MAP drops to 10% of the corpus, and cross-organ
generalisation becomes claimable rather than assumed. Acquisition needs 348
downloads (~178 GB; storage is not a constraint) via
`scripts/download_hest_visium.py`, which fetches only `st/` and `patches/` —
feature extraction reads pre-extracted patches, so the 25 GB of whole-slide
images per ~95 samples is skipped.

Excluded: Xenium (73), Visium HD (21), Spatial Transcriptomics (163) — different
chemistries and panels, so pathway-target semantics are not comparable.

## 4. Targets: depth-robust scoring + explicit QC

The v1 target (`mean log1p CP10k over member genes`) is ~65% sequencing depth
and is not, strictly, "pathway activity" — it is gene-set mean expression.
p53 activity is post-translational; averaging P53_PATHWAY transcripts does not
measure it. This is a framing exposure as much as a technical one, and it is
why PROGENy uses fitted downstream-responsive genes instead of nominal
membership.

**Decision (revised after tuning, `scripts/tune_aucell.py`): keep the mean
log1p CP10k score and regress out measured depth. Do not adopt AUCell.**

AUCell was the initial choice because within-spot ranking is depth-robust
*and* preserves slide-stationarity. Tuning falsified the premise:

| config | depth `\|r\|` | reliability (ceiling) |
| --- | --- | --- |
| mean log1p (raw) | 0.751 | — |
| **mean log1p + depth-regression** | **0.000** | **0.668** |
| AUCell R=100 (raw) | 0.274 | — |
| AUCell R=100 + depth-regression | 0.000 | 0.531 |

Depth regression removes the confound *completely*, so AUCell's rank-robustness
buys nothing while its discarding of magnitude costs 0.14 of ceiling. Switching
residualisation from PC1 to depth-regression also lifts the ceiling from 0.49
to **0.668** on its own — the single largest improvement available here.

**The slide-stationarity objection dissolves under inspection.** The property
was never actually achieved: median depth varies **83×** across studies and the
score correlates with depth at 0.93, so identical biology already yields wildly
different target values on different slides. Per-slide depth regression
*restores* cross-slide comparability rather than breaking it — and the primary
metric is within-slide anyway (§7), where per-slide transforms are harmless.

**Tuning also exposed a trap worth recording.** Optimising
`reliability − depth_leakage` naively selects `max_rank=25`, which looks
excellent (depth `|r|` 0.09) because **98% of scores are exactly zero and only
5 of 50 pathways retain any variance** — a near-constant correlates with
nothing. `tune_aucell.py` now rejects configurations failing a
non-degeneracy guard (`--min-live-frac`, `--max-frac-zero`). Best *viable*
AUCell setting was common-universe `max_rank=100` (all 50 pathways live), which
is what the table above uses.

A second slide-stationarity hazard surfaced en route: HEST gene universes range
**17,943–36,601** genes, so any *fractional* rank cutoff means a different
absolute cutoff per slide. Irrelevant now that AUCell is dropped, but it would
bite any future rank-based scorer.

Remaining requirements:

- **QC-filter on reliability, not just depth.** Slides whose targets barely
  reproduce against themselves contribute noise: MISC69 (607 genes/spot) has
  residual reliability 0.255 vs 0.550 for MISC73. Drop below a preregistered
  threshold.
- **Apply the residualisation to training targets too**, so training,
  checkpoint selection and evaluation all optimise the same quantity. v1
  trained and selected on raw targets while grading on the residual, which
  would have made a weak `stf_dense` result ambiguous between "dense
  supervision fails" and "we never optimised for it."

## 5. Design: hold the readout fixed, vary the supervision

The corrected primary contrasts — same readout on both sides, so only
supervision differs:

| Contrast | Isolates |
| --- | --- |
| `stf_dense[attention]` vs `stf_weak[attention]` | **Primary.** Same mechanism, different supervision. |
| `stf_dense[prediction]` vs `stf_weak[dense-head]` | **Primary.** Same head, different supervision. |
| `stf_dense[prediction]` vs `stf_weak[attention]` | Reported for continuity with prior work, flagged as readout-confounded. |

Both ingredients were already being computed in v1 — this is a choice-of-contrast
fix, not new data collection.

## 6. Model matrix

Identical features (**H-Optimus-0**, `--backbone h_optimus_0`), study-grouped
splits, pathway set, loss and epochs.

**Backbone choice is load-bearing, not incidental.** CTransPath is second-worst
of the eleven encoders in the HEST-Benchmark, and on this task it is the
difference between an experiment and a null result:

| backbone | within-study | cross-study (LOSO) |
| --- | --- | --- |
| CTransPath | 0.122 | **-0.026** |
| H-Optimus-0 | **0.280** | **+0.192** |

With CTransPath, cross-study transfer collapses to nothing and every arm of the
matrix would score ~0 — the study-grouped LOSO protocol of §3 would be
unrunnable. With H-Optimus-0 cross-study *exceeds* CTransPath's within-study
figure, so LOSO is viable. UNI / GigaPath / Virchow need institutional access
and are unavailable; Phikon is reachable but only ~6% better than CTransPath on
average. **Spatial PE enabled on both STF arms** — the experiment is about
spatial biology, so a weak-attention result must not be attributable to a
missing positional encoding; PE-on remains architecture-matched across the pair.

| ID | `--model` | Supervision | Per-pathway signal(s) |
| --- | --- | --- | --- |
| `stf_dense` | `interaction` | dense per-spot | dense prediction; pathway→patch attention; gradient saliency |
| `stf_weak` | `interaction --weak-supervision` | bag | **dense-head output** (never spot-supervised); attention; saliency |
| `attn_mil` | `attention_mil --weak-supervision` | bag | shared attention (single map); saliency |
| `transmil` | `transmil --weak-supervision` | bag | shared CLS→patch attention; saliency |
| `random` | `interaction` (untrained) | — | attribution floor |

MIL shared attention is one map broadcast across pathways, so its flat
per-pathway profile is a *structural property*, not an empirical finding —
state it as such. Gradient saliency is what gives MIL a genuinely per-pathway
map.

## 7. Metric: fidelity as a fraction of the achievable ceiling

Per held-out slide, per pathway, produce an `(N, P)` signal map and compare to
ground truth:

- **Primary — residual fidelity, reported against the linear baseline:**
  Pearson between depth-residualised signal and depth-residualised truth, quoted
  *alongside* `baseline_pca_ridge.py` run on the identical split (§2d). A model
  scoring 0.19 cross-study has merely matched Ridge; beating it is the claim
  worth making. Split-half reliability is reported as context, not as the
  denominator.
- **Secondary — raw fidelity**, for comparability with prior work, explicitly
  flagged as depth-inflated.
- **Report by cluster** (§2e): aggregate within the ~6 empirical pathway
  clusters before averaging, so no axis is counted five times.
- Per-slide z-scoring before correlating; Spearman alongside Pearson.
- **Statistics at study level**, paired across held-out studies (n=61).

## 8. Controls

- **Negative:** untrained `random` → fidelity ≈ 0.
- **Ceiling:** split-half reliability per slide and per pathway.
- **Sanity:** dense-prediction fidelity must match engine `validate()`
  per-pathway PCC (implemented, agrees to 1.4e-08).
- **Batch shortcut:** `dataset_title` is 98.5% predictable from CTransPath
  features (vs 60.3% majority baseline), and median depth varies **83×** across
  studies — so "predict depth from H&E" partly reduces to study recognition.
  Study-grouped LOSO handles test leakage but not within-training shortcutting;
  report this alongside results. Note this was measured on CTransPath, and much
  of it appears to be a CTransPath weakness: per-slide feature standardisation
  is essential there (-0.026 → +0.044) but nearly irrelevant for H-Optimus-0
  (+0.192 → +0.198), whose embeddings are already far more study-invariant.
  Worth re-measuring the shortcut on H-Optimus-0 before quoting it.
- **Two attributions** (attention + gradient) so conclusions aren't artefacts of
  one explainer.

## 9. Expected outcomes

| Observation | Interpretation |
| --- | --- |
| `stf_dense[attention]` ≫ `stf_weak[attention]` | Dense supervision improves the *same* mechanism — supports the thesis. |
| `stf_dense[attention]` ≈ `stf_weak[attention]` | Attention localises comparably regardless of supervision. |
| `stf_weak[dense-head]` > 0 | Bag supervision localises implicitly, without spot labels. |
| All arms ≈ 0 **and** ceiling ≈ 0 | Targets are noise-dominated; report as a measurement-limit result, not a modelling one. |
| Raw fidelity high, residual ≈ 0 | Model is riding depth, not resolving pathway-specific biology. |
| Shared-attention MIL flat across pathways | Structural, not empirical (§6). |

## 10. Build status

Reusable from v1: `attribution.py`, dataset-grouped splitting + CLI,
`evaluate_spatial_attribution.py` (incl. §8 sanity check),
`run_attribution_experiment.py`, `diagnose_batch_shortcut.py`, tests.
New in v2: `select_residual_pathways.py`, `measure_target_reliability.py`,
`download_hest_visium.py`.

Still to build: AUCell scorer in `compute_pathway_activities.py` (with
`max_rank` tuned per §4 and a `format_version` bump), depth-residualised
training targets, reliability-normalised metric and cluster aggregation in the
eval script, and the corrected primary contrasts in the orchestrator.

## 11. Sequencing

1. **Acquire** (running) — 348 human-Visium samples.
2. **Fix targets on the existing 73** — implement AUCell, tune `max_rank`
   against depth correlation, re-measure reliability. Validate small before
   scaling: recomputing 421 samples with a bad scorer is the expensive mistake.
3. **Extract features + score pathways** at full scale.
4. **Pilot** COLON-MAP-vs-rest to validate the pipeline end-to-end.
5. **Full LOSO** (n=61) for the headline result.
