# Evaluation harness

The measurement instrument for every change after Phase 3. Without it,
"did the pipeline rewrite help?" and "which model should we use?" are both
unanswerable, and model selection degenerates into swapping until something
returns parseable output.

## Running it

```bash
# Deterministic pipeline only — no API key needed, no cost
python evals/run.py

# Include the LLM evaluation (spends credits)
python evals/run.py --with-llm

# Machine-readable, for diffing two runs
python evals/run.py --json > baseline.json

# Compare candidate models
python evals/bakeoff.py --models MODEL_A MODEL_B
```

## What is measured

| Metric | What it tells you |
|---|---|
| **nDCG@5 / @10** | Whether the right candidate is ranked first. The standard ranking metric in the published resume-matching literature, so the numbers are comparable outside this repo. |
| **RBO** | Rank agreement, top-weighted — disagreement at rank 1 costs more than at rank 10. Extrapolated form, so identical lists score exactly 1.0 at any length. |
| **Spearman** | Whether the continuous score tracks human grades at all. |
| **skill_f1** | Skill extraction against hand-labelled expectations. |
| **missing_skill_f1** | Whether reported gaps are real gaps. This is what the Phase 0 normalization bug destroyed. |
| **Parser field recall** | Which of name/email/phone/linkedin/github survive extraction, broken down by layout. Two-column resumes are where this falls apart. |
| **Schema conformance** | Share of LLM calls returning valid JSON. The metric that should decide model choice. |

## Stress cases

`seed.jsonl` is the smoke test — clean separation, which the pipeline scores
1.0 on. `stress_a.jsonl` and `stress_b.jsonl` are the real benchmark: six
groups built to break specific parts of the pipeline.

| Stress | What it probes |
|---|---|
| `vocabulary` | Two byte-equivalent candidates, one writing `K8s`/`TF`/`Postgres` and one writing them out. Any score gap is surface-form sensitivity. |
| `keyword_stuffing` | A 48-item skills list with zero supporting evidence, against a genuine engineer with fewer keywords. |
| `career_change` | A teacher whose Projects section meets every requirement, against someone whose *title* says Software Engineer but whose work does not. |
| `overqualification` | A principal engineer with 15 years applying to a junior role — maximal keyword overlap, wrong role. |
| `near_tie` | Two ML engineers with genuinely contestable ordering, plus a researcher the JD explicitly discounts. |
| `mangled_extraction` | The same candidate twice: clean text, and the interleaved text a two-column PDF actually produces. |

### Equivalence pairs

Some candidates are declared `equivalent_pairs` — they describe substantively
the same fit, so they must receive near-identical scores. **This catches
defects the ranking metrics hide**: if the order happens to come out right, a
12-point penalty for writing "K8s" instead of "Kubernetes" is invisible to
nDCG but obvious here.

## Current baseline (deterministic pipeline, no LLM)

Committed as `evals/baseline.json`, regenerated after the Phase 1 cleanup.
Re-cut it whenever the numbers below stop matching a fresh run, and say in the
commit what moved.

```
ndcg@5   0.9984      everything except "general"   1.0000
ndcg@10  0.9984      general                       0.9951  <-- weakest
rbo      0.9668
spearman 0.9609

equivalence   worst gap 2.72   (pv-abbrev 83.0 vs pv-spelled 85.8)
skill_f1 0.9030      missing_skill_f1 0.7379
parser recall 0.8333 (single-column 0.67; linkedin 0.50, email 0.75)
```

### What this tells us now

1. **The benchmark is saturated.** Six of the seven stress families score a
   perfect 1.0000 and the seventh scores 0.9951. Keyword stuffing, career
   change, overqualification, near ties and mangled extraction are all solved
   *as posed here*. That is not a claim that the system handles them in the
   world — it is a statement that these 42 self-labelled candidates can no
   longer tell a good change from a bad one.
2. **So ranking metrics have almost no power left.** A change that improves
   the product cannot show up in a number already at 1.0. Neither can a
   regression, until it is large enough to flip an order. Adding more cases of
   the same kind does not fix this: they would be labelled by the same person
   who tuned the system.
3. **Vocabulary sensitivity is nearly closed but not closed.** Writing
   `K8s`/`TF`/`Postgres` instead of the full words still costs 2.72 points of
   a possible 100, down from 12.53. The residual is small enough that the
   invariance gate sits at 5 with a documented ratchet to 3 once it closes.
4. **Skill extraction is where the headroom is.** `missing_skill_f1` at 0.7379
   means roughly a quarter of the reported gaps are wrong — the number a user
   acts on directly, and the one no ranking metric reflects.
5. **The parser is the other weak point.** 0.67 recall on single-column
   layouts, with LinkedIn found half the time. Item 9 rebuilds this path.

The honest conclusion is the one that shaped Phase 2: the instrument needs
external ground truth — a public dataset and human graders — because no amount
of additional self-labelled stress cases can restore its power.

## Public dataset — the result

```bash
python evals/public_dataset.py --checkpoint run.jsonl        # ~25 min, resumable
python evals/public_dataset.py --checkpoint run.jsonl --summarize-only
```

Metrics fixed in advance in `PREREGISTRATION.md`, committed before the first
scorer ran. Full output in `public_dataset_result.json`.

`cnamuangtoun/resume-job-description-fit`, test split, all 1,759 rows across 70
job descriptions, revision `08978e217149`. 20,501 within-job label-differing
pairs.

| Scorer | Pairwise | Spearman | 95% CI | Macro AUC |
|---|---:|---:|---:|---:|
| TF-IDF cosine | 0.5356 | 0.1335 | [0.0876, 0.1796] | 0.5083 |
| Embedding | 0.5501 | 0.1045 | [0.0594, 0.1516] | 0.5074 |
| Coverage alone | 0.5244 | 0.0374 | [-0.0090, 0.0845] | 0.5016 |
| **CareerIQ** | **0.5239** | **0.0988** | [0.0531, 0.1453] | 0.5053 |

**This is a negative result, and it is the pre-registered one.** Both primary
thresholds are missed: pairwise accuracy 0.5239 against a 0.60 bar, where 0.50
is chance; Spearman 0.0988 against a 0.30 bar. The confidence interval excludes
zero, so the association is real rather than noise — it is just very small.

Three things follow, and they should be reported together because any one of
them alone is misleading.

**1. CareerIQ does not beat the 30-line TF-IDF baseline.** Paired permutation
test on Spearman, 10,000 resamples: difference −0.0347, p = 0.42. Not
significant in either direction. Everything the pipeline does beyond term
overlap — embeddings, evidence weighting, level fit, clarity — buys nothing
measurable against these labels.

**2. It does beat coverage alone**, +0.0614, p = 0.027. So the machinery on top
of keyword coverage is doing *something*. That is the one positive finding here,
and it is a modest one.

**3. Every scorer is near chance, including the literature baseline.** TF-IDF
manages 0.5356. This is not a CareerIQ-specific failure, which cuts both ways:
the pipeline is not uniquely bad, and it also has no demonstrated advantage over
the simplest thing anyone would try.

### Why the score barely moves

The clearest number is the mean score per label (post-hoc, below): CareerIQ
gives 34.7 to `No Fit`, 35.8 to `Potential Fit`, 37.0 to `Good Fit`. The
ordering is right. The range is 2.4 points out of 100. A user moving from a
resume a human calls "no fit" to one a human calls "good fit" sees the number go
from 35 to 37.

The embedding scorer is not even monotonic — 60.6 / 62.1 / 61.7, with
`Potential Fit` scoring above `Good Fit`.

### Exploratory, post-hoc

Added after seeing the pre-registered result, and labelled as such in the code
and the output. It does not replace anything above.

| Scorer | Per-job median pairwise | Good vs No Fit only | Mean by label |
|---|---:|---:|---|
| TF-IDF | 0.6079 | 0.5340 | 5.9 / 6.6 / 6.7 |
| Embedding | 0.5993 | 0.5500 | 60.6 / 62.1 / 61.7 |
| Coverage | 0.5343 | 0.5476 | 17.7 / 17.4 / 20.1 |
| CareerIQ | 0.5967 | 0.5589 | 34.7 / 35.8 / 37.0 |

Two observations worth following up rather than concluding from:

- **The per-job median (0.5967) is well above the pooled figure (0.5239).**
  Pooling weights by pair count, so a job with 89 rows contributes 3,916 pairs
  and one with 5 contributes 10. Of 49 jobs with at least 10 pairs, 23 score
  above 0.6 and 5 below 0.4. The pipeline does better on a typical job than on a
  typical comparison, and a few large groups it handles badly dominate the
  headline. Which average matters depends on what is being claimed; the
  pre-registered one stands as primary because it was chosen blind.
- **Dropping the contested middle class barely helps** — 0.5239 to 0.5589. The
  `Potential Fit` label is not where the difficulty lives.

### What this does not license

It does not license re-tuning weights until this number improves and then
reporting the improvement. The pre-registration rules that out explicitly: any
change to `compute_fit` made after seeing these figures invalidates them and
requires a second, separately reported measurement. The stress-set numbers
cannot rescue this either — they are 0.9984 because their labels were written
by the person who tuned the system, and that is precisely the gap this
experiment existed to expose.

The honest one-line summary for the paper: *on 1,759 independently labelled
resume–job pairs, the system's score correlates with human judgement at
rho = 0.099 (95% CI 0.053–0.145), statistically distinguishable from zero and
not distinguishable from a TF-IDF baseline.*

## Component weight sensitivity

```bash
python evals/weight_sensitivity.py --checkpoint run.jsonl \
    --component-cache components.json
```

`compute_fit` blends semantic 0.45 / coverage 0.40 / clarity 0.15, and those
numbers were chosen by judgement. The sweep walks all 231 triples on a 0.05
simplex and re-scores everything at each one. It is exact rather than
approximate: components are computed once per candidate and `compute_fit`
itself is then called with the module constants rebound, so no copy of the
blending formula lives in the sweep and it cannot drift from the product.

Full grid in `weight_sensitivity_result.json`.

| | Shipped | Best achievable | Worst | Spread |
|---|---:|---:|---:|---:|
| Stress nDCG@5 | 0.9984 | 1.0000 | 0.9760 | 0.024 |
| Public Spearman | 0.0988 | 0.1198 | 0.0445 | 0.075 |
| Public pairwise | 0.5239 | 0.5411 | 0.5113 | 0.030 |

Three things come out of this, and the third is the one that matters.

**1. The stress set cannot distinguish weightings.** Every triple in the grid
scores between 0.976 and 1.000 on nDCG@5 — including `[0.0, 1.0, 0.0]`, pure
keyword coverage with the embeddings switched off entirely, which scores 0.999.
A benchmark on which deleting the semantic component costs nothing cannot be
used to justify keeping it. This is the saturation problem stated as sharply as
it can be stated.

**2. No weighting rescues the public dataset.** The best triple anywhere in the
grid reaches Spearman 0.1198 and pairwise 0.5411, against pre-registered bars of
0.30 and 0.60. The gap is not a tuning problem. Roughly two thirds of the
distance from chance to the target would still be missing after picking the
single best weighting in hindsight.

**3. The two datasets disagree about which way to tune.**

| Weights | Stress nDCG@5 | Public Spearman |
|---|---:|---:|
| `[0.45, 0.40, 0.15]` shipped | 0.9984 | 0.0988 |
| `[0.35, 0.65, 0.00]` stress-optimal | **1.0000** | 0.0875 |
| `[0.95, 0.00, 0.05]` public-optimal | 0.9791 | **0.1198** |

The hand-labelled set wants coverage-heavy weights. The independently labelled
set wants almost pure semantic similarity, with coverage at zero. Each optimum
is worse than shipped on the other dataset. Tuning against the stress set would
have moved the weights *away* from what third-party labels prefer — which is
the concrete form of the problem with grading your own homework, and a better
argument for external ground truth than any amount of saying so.

Both optima also drive clarity to ~0. It survives in the shipped blend on the
reasoning in `services/scoring.py` (writing quality should not decide a
comparison between two candidates who both match the role), not because either
benchmark asks for it.

### This is analysis, not tuning

Adopting `[0.95, 0.00, 0.05]` because the table above says it scores best would
be exactly the move `PREREGISTRATION.md` rules out: weights fitted to the test
split, reported as a result on that split. The weights are unchanged. If they
are ever changed on the strength of this, the public dataset has to be re-run
and reported as a second, separate measurement, and the fitted number is not
evidence about anything.

## Human grading study

```bash
python evals/grading_study.py prepare --checkpoint run.jsonl --out study/
# three people fill in grades_a/b/c.csv
python evals/grading_study.py score --dir study/
```

**Status: built, waiting on three graders.** This is the one item in the plan
that cannot be written — it needs people to read resumes. Everything around
them is here and tested, so the remaining step is finding them.

Why it is worth the week it costs. The public split gave rho = 0.099, but those
labels arrive with no rubric, no record of who wrote them and no measure of how
much the labellers agreed with each other. A correlation against labels of
unknown reliability has an unknown ceiling: if the labels are themselves noisy,
a low correlation may be measuring the noise rather than the system. Grading a
sample locally, with a stated rubric and a measured agreement figure, is what
puts a number on that ceiling.

`prepare` writes 50 pairs sampled from the public split, stratified by CareerIQ
score decile so the study spans the range rather than clustering where the
system puts most resumes. Each grader gets the same pairs in a different order,
from a different seed, so drift in strictness over a session does not line up
across graders and get measured as agreement. The packets contain the resume,
the job description and an empty grade column — no score, no dataset label.
`_key.csv` holds those and is not shared.

`score` reports:

- **Krippendorff's alpha, ordinal** — not Fleiss' kappa, because a 0 graded
  against a 3 should count as worse disagreement than a 0 against a 1, and
  because alpha tolerates a grader skipping a pair. Convention: 0.80+ reliable,
  0.667–0.80 tentative only, below that too noisy to conclude from.
- **Exact and within-one-grade agreement**, because those are the numbers a
  reader can check by eye.
- **CareerIQ vs the mean human grade**, Spearman with a bootstrap CI.
- **The human ceiling** — each grader against the mean of the other two. This
  is the number that makes the rest interpretable. If three people agree with
  each other at 0.6, a system scoring 0.5 is doing well; the same 0.5 against a
  ceiling of 0.95 is not. Reporting the system's correlation without it is the
  usual way this measurement gets overstated.
- **A cross-check against the public dataset's own labels.** If the graders
  disagree with those labels about as much as CareerIQ does, the labels are the
  noisy part and the rho = 0.099 result understates the system. If the graders
  agree with them and CareerIQ does not, it does not.

The scoring path is tested end to end against simulated graders whose agreement
is known by construction — including the cases that matter when real people are
involved: a blank grade is an abstention rather than a zero, a pair graded once
is excluded rather than counted as perfect agreement, two returned packets are
enough and one is refused.

## Hidden-text attack

```bash
python evals/hidden_text_attack.py
python evals/hidden_text_attack.py --json > attack.json
```

Takes every candidate in the stress set graded 1 or 0 — the ones an inflated
score would wrongly promote — and rebuilds their resume as a PDF with the job
description's own skill list pasted in where a reader cannot see it. Three
variants: white on white, 0.5pt, and positioned below the page edge. The
measurement is **score inflation**, the attacked resume's score minus the same
resume's without the hidden block. Zero is the only good answer.

60 attacked documents (20 candidates x 3 modes):

| Scorer | Mean inflation | Worst | Documents moved |
|---|---:|---:|---:|
| TF-IDF cosine | +11.51 | +21.89 | 60 / 60 |
| Embedding similarity | +3.04 | +7.65 | 57 / 60 |
| CareerIQ, all extractable text | +26.77 | +54.92 | 60 / 60 |
| **CareerIQ, visible text only** | **+0.00** | **+0.00** | **0 / 60** |

Reading the table: the pipeline was the *most* exploitable of the three before
the filter, not the least. That is a consequence of doing more with the text —
keyword coverage, evidence weighting and skill extraction all read the hidden
block and all agreed it was real experience, so their contributions compounded.
A candidate graded "not a fit" gained up to 54.92 points of 100. Afterwards no
hidden term reaches the scored input in any of the 60 documents.

The three modes score identically because the payload is the same string in
each; what differs is only how it is concealed from a human, and extraction
recovers it either way. That is the point worth making about the attack: it
costs nothing to vary, so a detector that catches one variant catches nothing.

The baselines are included because "our system resists this" means little
without them. TF-IDF is the standard non-neural baseline in the resume-matching
literature; it moves less than the full pipeline only because it does less. The
embedding model is the most robust of the three unfiltered — a hidden skill list
is a bag of nouns with no sentence structure, and the encoder discounts it —
but +7.65 at worst is still a candidate moving up a ranking for free.

## Grades are provisional — review them

Every grade in the stress files is **my proposed judgement, not ground truth**.
Cases marked `REVIEW` in their notes are the contestable ones. Reasonable
people would disagree about `nt-research` (does a PhD with one product
transfer beat a pure shipper?) and about `jo-mid` versus `jo-graduate`.

Change them. The labels are the ground truth the whole harness rests on, and
they should reflect *your* judgement about who you would interview.

## Expanding further

Public starting points:

- HuggingFace: `cnamuangtoun/resume-job-description-fit`
- Kaggle: resume + job-description pair collections

## Adding cases

One JSON object per line in `evals/data/*.jsonl`:

```json
{
  "id": "unique-case-id",
  "job_title": "Senior Backend Engineer",
  "jd_text": "full job description text",
  "candidates": [
    {
      "id": "candidate-1",
      "resume_text": "full resume text",
      "relevance": 3,
      "expected_skills": ["python", "docker"],
      "expected_missing": ["kubernetes"],
      "notes": "why this grade"
    }
  ]
}
```

Relevance grades: **3** strong fit · **2** good fit · **1** stretch · **0** not a fit.

Every case needs at least one grade-3 and one grade-0, or the ranking metrics
have nothing to discriminate — `tests/test_eval_metrics.py` enforces this.

## Workflow

1. `python evals/run.py --json > before.json` **before** changing anything.
2. Make the change.
3. Re-run and diff against `before.json`. If the number did not move, either
   the change did nothing or the benchmark cannot see it — find out which
   before continuing, because at current saturation the second is likely.
4. When a change is meant to move the baseline, regenerate `evals/baseline.json`
   in the same commit.

The old exit criterion — "+10 nDCG points over the Phase 0 baseline" — has been
met and then some, and is no longer a useful target: nDCG@5 is 0.9984 and there
are 0.0016 points left to win. Phase 2 replaces it with criteria that can still
discriminate: agreement with human graders, performance on a public dataset,
and the invariance gates.
