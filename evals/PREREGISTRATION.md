# Pre-registration — public dataset evaluation

Written and committed **before** any scorer was run against this dataset. The
point of doing it in this order is that the metric cannot be chosen after
seeing which one flatters the system, and the commit history is the evidence.
Everything below is fixed. If something has to change, the change goes in a
later commit that says what changed and why, and the original stays here.

## Dataset

`cnamuangtoun/resume-job-description-fit`, the `test.csv` split: 1,759
resume / job-description pairs, each with a human-assigned label.

Inspected before writing this (structure only, no scoring):

- Labels: `No Fit` 857, `Potential Fit` 444, `Good Fit` 458.
- 71 distinct job descriptions across the 1,759 rows, 477 distinct resumes.
  Group sizes run from 2 to 89 rows per job description.
- 61 of the 71 groups contain more than one distinct label, giving 18,726
  within-job pairs whose labels differ.

The snapshot revision is recorded in the report, so the numbers can be
regenerated against the same bytes.

## Label treatment

The labels are **ordinal**, and are mapped `No Fit` → 0, `Potential Fit` → 1,
`Good Fit` → 2.

This is the decision the plan required to be stated in advance, because a
three-class label set admits several framings and they do not give the same
answer. The ordinal reading is chosen because the three labels describe degrees
of one quantity and the system under test emits a single continuous 0–100
score. Collapsing to binary would discard the `Potential Fit` distinction,
which is the boundary a recruiter actually argues about.

## Primary metrics

1. **Within-job pairwise accuracy.** Over every pair of rows sharing a job
   description whose labels differ, the fraction where the scorer gives the
   higher-labelled resume the higher score. A tie in score counts as 0.5.
   Pairs are restricted to within-job because that is the decision the product
   makes — ranking candidates for one opening — and because comparing scores
   across different jobs measures something the product never does.

2. **Spearman rank correlation** between the score and the ordinal label,
   across all 1,759 rows, with a 95% confidence interval from 10,000 bootstrap
   resamples over rows.

## Secondary metric

3. **ROC-AUC, macro one-vs-rest.** For each of the three labels, that label
   against the other two, ranking by the continuous score; the reported figure
   is the unweighted mean of the three. The binary `Good Fit` vs rest AUC is
   reported alongside it, because that is the single number closest to a
   shortlisting decision. Secondary because AUC over a pooled multi-class set
   rewards separating jobs from each other as much as candidates within a job.

## Scorers

All four are run on identical rows:

| Name | What it is |
|---|---|
| `tfidf` | TF-IDF cosine, `evals/tfidf.py` — the standard non-neural baseline |
| `embedding` | the sentence embedding similarity, `services/similarity.py` |
| `careeriq` | `compute_fit(...).overall`, the number the product shows |
| `coverage` | weighted skill coverage alone, to show what the rest of the pipeline adds |

The LLM evaluation is **not** included: it costs credits per row and 1,759 rows
of a free tier is not a repeatable measurement. That exclusion is a limitation
of this experiment, not a claim about the model.

## Fixed in advance

- **No threshold tuning, no weight tuning, and no prompt changes** against this
  split. Any change to `compute_fit` made after seeing these numbers invalidates
  them and requires a re-run reported as a second, separate measurement.
- **Sample**: the full test split, all 1,759 rows. If the embedding scorer
  cannot complete the full split within the compute budget, the fallback is a
  uniform random subsample drawn with seed `20260918`, and the report states the
  size and that it was a subsample.
- **Text**: `resume_text` and `job_description_text` used verbatim. No
  hidden-text filtering — these are plain text fields with nothing concealed in
  them, so the filter is inert here by construction.
- **No per-class reweighting** and no removal of any group, including the
  89-row one.

## What a negative result looks like

Stated now so it cannot be renegotiated later. Either of these means the score
does not track third-party human judgement, and it goes in the paper that way:

- Within-job pairwise accuracy below 0.60 (0.50 is chance), or
- Spearman below 0.30, or a 95% CI that includes 0.

A result where `careeriq` does not beat `tfidf` is also reported plainly. The
comparison exists to find that out, and the stress-set numbers cannot answer it
— those labels were written by the same person who tuned the system, and this
dataset's were not. That independence is the whole reason for running it.
