# A1: Lexical & Semantic Retrieval on EB-NeRD and MIND

**Course**: CS4.406 — Information Retrieval & Extraction  
**Due**: August 27, 2026 | **Individual**

---

## One-Command Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline (download → preprocess → split → feature store)
python build_pipeline.py

# Or for a specific dataset:
python build_pipeline.py --dataset mind
python build_pipeline.py --dataset ebnerd

# Skip download if data already exists:
python build_pipeline.py --skip-download
```

---

## Run Retrieval

```bash
# BM25 lexical retrieval (recall@50, 100, 200)
python -m src.retrieval.bm25 --dataset mind
python -m src.retrieval.bm25 --dataset ebnerd

# Semantic embedding retrieval
python -m src.retrieval.semantic --dataset mind
python -m src.retrieval.semantic --dataset ebnerd
```

---

## Run Evaluation

```bash
# Offline eval harness — re-ranks val impressions with the chosen retriever and
# reports AUC, MRR, nDCG@5, nDCG@10, diversity, novelty, coverage + slicing + CIs
python -m src.evaluation.metrics --dataset mind   --retriever bm25
python -m src.evaluation.metrics --dataset mind   --retriever semantic
python -m src.evaluation.metrics --dataset ebnerd --retriever bm25
python -m src.evaluation.metrics --dataset ebnerd --retriever semantic

# Slice by head/tail articles instead of the default cold-start/warm users
python -m src.evaluation.metrics --dataset mind --retriever bm25 --slice head_tail
```

---

## Generate Predictions (Codabench Submission)

```bash
python -m src.submission.generate_preds --dataset mind   --retriever bm25
python -m src.submission.generate_preds --dataset ebnerd --retriever semantic
```

By default this scores `impressions_competition_test.parquet` — the official,
unlabeled MINDlarge_test / ebnerd_testset bundle produced by `build_pipeline.py
--large` — which is what actually gets submitted to Codabench. Pass `--split
test` instead to score our own internal (labeled) temporal test split, useful
as a prediction-format dry run.

Prediction files are saved to `predictions/`.

**`--unique-ids`** (EB-NeRD): emit one line per *distinct* `impression_id` rather than
one per row. EB-NeRD's test set has 13,536,710 rows but only 13,336,711 distinct
`impression_id`s, because its 200,000 `is_beyond_accuracy` rows all carry
`impression_id = 0`. The course-provided `ebnerd_analysis.ipynb` groups by
`impression_id` and writes 13,336,711 lines, stating the file must cover all unique
impression IDs; this flag matches that. No-op for MIND, which has no duplicate IDs.

### A note on prediction files and the "no large files" rule

Q7 lists prediction files as a deliverable, and Q8 forbids committing large files.
These conflict here: the submission archives are 96–230 MB each (MIND 2.37M lines,
EB-NeRD 13.3M lines). They are therefore **deliberately gitignored, not omitted** —
`predictions/` is tracked via `.gitkeep`, and any archive is reproducible byte-for-byte
from the commands above, since scoring is deterministic given the same index and split.
Regenerating all four takes roughly 30 minutes total.

---

## Engineering Benchmarks (Q6 analysis)

```bash
# ANN index choice: exact FlatIP vs IVFFlat (nprobe 1/8/32) vs HNSW (efSearch 32/128).
# Reports build time, on-disk size, single-query latency, p95, batch throughput, and
# recall measured AGAINST THE EXACT RESULT.
python -m src.evaluation.benchmark_index --dataset mind --n-queries 1000

# Scoring-path cost: BM25 vs semantic latency/throughput, plus a controlled
# before/after for each optimisation (targeted vs full-corpus scoring, token
# memoisation, buffer reuse, vectorised gather) and artifact storage sizes.
python -m src.evaluation.benchmark_retrieval --dataset mind --n 1000
```

These produce the tables in the design note. They are the evidence for *why* the exact
index is shipped, rather than an assertion that it is fine.

---

## Evaluating on a sample

Every eval CLI accepts `--sample-n N --seed S` for a seeded, reproducible subsample:

```bash
python -m src.evaluation.metrics --dataset ebnerd --retriever bm25 --sample-n 250000
python -m src.retrieval.semantic --dataset ebnerd --k 50 100 200 --sample-n 250000
```

EB-NeRD's val split is ~2M impressions and a full Q4 pass costs ≈1 h per retriever;
full-corpus recall@K there is multi-hour. The sampling was **validated, not assumed** —
the same configuration measured at N=250,000 and at N=1,997,301 differs by 0.0003 AUC,
inside both bootstrap intervals (see `report.txt`).

---

## Run Tests

```bash
pytest tests/ -v
```

The `test_no_leakage.py` test asserts there is no future-click leakage between splits.

---

## Project Structure

```
ire-a1/
├── data/                   # gitignored — raw and processed data
│   ├── mind/raw/           # MIND-small raw files from HuggingFace
│   ├── mind/processed/     # parquet feature store, FAISS index, BM25 index
│   ├── ebnerd/raw/         # EB-NeRD raw zip/parquet files
│   └── ebnerd/processed/
├── src/
│   ├── pipeline/
│   │   ├── download.py         # Q1: download datasets
│   │   ├── preprocess.py       # Q1: parse into unified schema
│   │   ├── split.py            # Q1: temporal train/val/test split
│   │   └── feature_store.py    # Q1: article + user feature store
│   ├── retrieval/
│   │   ├── bm25.py             # Q2: BM25 lexical retrieval
│   │   └── semantic.py         # Q3: embedding ANN retrieval
│   ├── evaluation/
│   │   ├── metrics.py                # Q4: eval harness (AUC/MRR/nDCG, CIs, slices)
│   │   ├── benchmark_index.py        # Q6: ANN index choice vs alternatives
│   │   └── benchmark_retrieval.py    # Q6: scoring-path latency + optimisations
│   └── submission/
│       └── generate_preds.py   # Q5: Codabench prediction files
├── tests/
│   └── test_no_leakage.py      # Q9: temporal split + behaviour-window tests
├── predictions/                # output prediction files (gitignored — see note above)
├── notebooks/                  # EDA notebooks
├── mind_analysis.ipynb         # course-provided dataset analysis
├── ebnerd_analysis.ipynb       # course-provided dataset analysis
├── build_pipeline.py           # Q1: one-command entry point
├── design_note.tex             # Q6: design note source (pdflatex design_note.tex)
├── report.txt                  # full results log: every number + its command
├── requirements.txt
├── ai_usage_log.md             # Q7: required AI usage log
└── README.md
```

---

## Deliverables map

| Assignment item | Where |
|---|---|
| Q1 reproducible pipeline | `build_pipeline.py`, `src/pipeline/` |
| Q2 BM25 lexical retrieval | `src/retrieval/bm25.py` |
| Q3 semantic retrieval + ANN | `src/retrieval/semantic.py` |
| Q4 evaluation harness | `src/evaluation/metrics.py` |
| Q5 Codabench predictions | `src/submission/generate_preds.py` |
| Q6 design note + engineering analysis | `design_note.tex`, `src/evaluation/benchmark_*.py` |
| Q9 anti-gaming / leakage tests | `tests/test_no_leakage.py` |
| All measured results | `report.txt` |

---

## Competitions

| Competition | Link |
|---|---|
| MIND (Codabench) | https://www.codabench.org/competitions/13967/ |
| RecSys 2024 Challenge (Codabench) | https://www.codabench.org/competitions/2469/ |

---

## Datasets

| Dataset | Source |
|---|---|
| MIND | https://huggingface.co/datasets/yjw1029/MIND |
| EB-NeRD | https://recsys.eb.dk/dataset/ |
| EB-NeRD Starter Code | https://github.com/jppol-ai/ebnerd-benchmark |
