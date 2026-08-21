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
│   │   └── metrics.py          # Q4: eval harness
│   └── submission/
│       └── generate_preds.py   # Q5: Codabench prediction files
├── tests/
│   └── test_no_leakage.py      # Q9: anti-gaming / leakage test
├── predictions/                # output prediction files (gitignored)
├── notebooks/                  # EDA notebooks
├── build_pipeline.py           # Q1: one-command entry point
├── requirements.txt
├── ai_usage_log.md             # Q7: required AI usage log
└── README.md
```

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
