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
