# AI Usage Log — CS4.406 Assignment 1

Track every AI tool interaction here. This is **required** as part of the Q7 deliverables.

## Format

For each AI interaction, fill in:
- **Tool**: the AI tool used (e.g., Antigravity/Claude, ChatGPT, Copilot)
- **Date**: when you used it
- **Prompt**: the exact prompt you gave
- **AI output used**: yes / no / partially
- **Human modifications**: describe what you changed
- **Files affected**: which source files were touched

---

## Log

### Entry 1
- **Tool**: Antigravity (Claude Sonnet)
- **Date**: 2026-08-20
- **Prompt**: "Do Phase 0 step up everything create project structure required for this assignment"
- **AI output used**: Yes — generated initial project scaffold
- **Human modifications**: _[fill in any changes you make]_
- **Files affected**:
  - `.gitignore`
  - `requirements.txt`
  - `build_pipeline.py`
  - `src/pipeline/download.py`
  - `src/pipeline/preprocess.py`
  - `src/pipeline/split.py`
  - `src/pipeline/feature_store.py`
  - `src/retrieval/bm25.py`
  - `src/retrieval/semantic.py`
  - `src/evaluation/metrics.py`
  - `src/submission/generate_preds.py`
  - `tests/test_no_leakage.py`
  - `README.md`

---

### Entry 2
- **Tool**: Claude Code (Claude Sonnet 5)
- **Date**: 2026-08-21
- **Prompt**: Reported two runtime bugs — `build_pipeline.py --dataset ebnerd --large --skip-download` was OOM-killed during the temporal split step, and `python -m src.retrieval.bm25 --dataset ebnerd --force-rebuild` printed "No valid queries found!" with recall@K = 0.0 for all K. Asked to go through the code and fix.
- **AI output used**: Yes
- **Diagnosis**: the split step (`src/pipeline/split.py`) loaded the full 12M-row impressions parquet into pandas — each row has 3 list-columns (click_history/candidates/labels) stored as individual Python objects, which blew past 16GB RAM. The BM25 zero-recall bug was a downstream symptom: the crash meant `val.parquet` never got regenerated, so evaluation ran against a stale file from an earlier small-scale run where every row had empty candidates/labels.
- **Human modifications**: reviewed and accepted the diagnosis and all changes; did not run the full pipeline in this session, code changes were made without execution per my instruction — verification is pending my own future run.
- **Files affected**:
  - `src/pipeline/split.py` — rewritten to use Polars lazy scan + streaming `sink_parquet` instead of `pd.read_parquet`; now combines the train + validation/dev bundles before splitting (previously the validation/dev bundle was preprocessed but silently discarded)
  - `src/pipeline/feature_store.py` — `build_user_store` had the same latent pandas full-materialization pattern; rewritten with a Polars groupby

### Entry 3
- **Tool**: Claude Code (Claude Sonnet 5)
- **Date**: 2026-08-21
- **Prompt**: "Just modify all the code of all the files according to whats asked in the assignment document do not run anything for now we will run the code later" — asked for a full pass over the codebase against the assignment PDF (A1.pdf), without executing anything.
- **AI output used**: Yes
- **What was found/fixed**:
  - `src/retrieval/bm25.py` — `BM25Retriever.retrieve()` required an `articles_lookup` argument that nothing actually passed (would crash `generate_preds.py` for BM25); moved lookup construction into `.build()` so the call signature matches `SemanticRetriever`'s and the module's own docstring. Added `score_candidates()` (scores a given impression's own candidate list, needed by the eval harness).
  - `src/retrieval/semantic.py` — added the matching `score_candidates()`.
  - `src/evaluation/metrics.py` — the previous `evaluate_df()` scored impressions using the raw dataset candidate order, never touching any retriever (the module docstring even referenced an `EvaluationHarness` class that didn't exist in the file). Rewrote it as a real `EvaluationHarness` that re-ranks each impression via a retriever's `score_candidates()`, then reports AUC/MRR/nDCG@5/nDCG@10 with bootstrap CIs plus diversity/novelty/coverage (previously defined but never wired into any report) and configurable cold/warm or head/tail slicing.
  - `src/pipeline/preprocess.py` — added processing of the official unlabeled Codabench test bundles (MINDlarge_test / ebnerd_testset) into `impressions_competition_test.parquet`, kept separate from the temporal train/val/test pool; fixed EB-NeRD article-catalog path priority to prefer the bundle-specific `articles.parquet`; added the `entities` field to the unified article schema (was silently dropped for both datasets despite being an explicit Q1.4 requirement — parsed from MIND's title/abstract entity JSON and EB-NeRD's `entity_groups`).
  - `src/submission/generate_preds.py` — default `--split` changed to `competition_test` (the actual file that should be submitted to Codabench) instead of our internal labeled test split.
  - `README.md` — updated commands to match.
- **Human modifications**: none yet — pending review after running.

### Entry 4
- **Tool**: Claude Code (Claude Sonnet 5)
- **Date**: 2026-08-21
- **Prompt**: Ran `pytest tests/test_no_leakage.py -v` after the MIND rebuild — `test_no_val_ids_in_train[mind]` failed. Asked to investigate.
- **AI output used**: Yes
- **Diagnosis**: MIND's `impression_id` is only unique *within* a single `behaviors.tsv` file — train/dev/test each restart numbering from 1. Confirmed directly: all 73,152 dev impression_ids collide with train's. Combining train+dev bundles for the temporal split (from the earlier fix) meant unrelated impressions from different source files could land on the same ID.
- **Human modifications**: reviewed and accepted; reran the pipeline + tests myself after the fix — all 6 leakage tests now pass.
- **Files affected**: `src/pipeline/preprocess.py` — prefixes `impression_id` with the split name (`train_`, `dev_`) before merging; the official competition test set is exempt since Codabench needs its original unprefixed numeric IDs.

### Entry 5
- **Tool**: Claude Code (Claude Sonnet 5)
- **Date**: 2026-08-21
- **Prompt**: Ran `python -m src.retrieval.bm25 --dataset mind` and got recall numbers that looked plausible but the log showed "Loaded index: 65,238 articles" while the article catalog had 125,589. Asked to fix.
- **AI output used**: Yes
- **Diagnosis**: `BM25Retriever.build()` cached its index in a pickle keyed only by file existence, not by content — a stale index (65,238 articles, built before the competition-test bundle was merged into the article catalog) got silently reused after `articles.parquet` changed, understating recall since many val/candidate articles weren't in the stale index at all. `SemanticRetriever` had the identical class of bug (a `corpus_ids.npy` field was declared but never actually written or checked).
- **Human modifications**: reran BM25 for MIND myself after the fix, confirmed it auto-rebuilt with a warning and produced corrected (lower, expected) recall numbers.
- **Files affected**: `src/retrieval/bm25.py` (pickle now stores + validates the article-id set before trusting the cache), `src/retrieval/semantic.py` (wired up the previously-unused `corpus_ids.npy` to do the same check for embeddings/FAISS cache).

### Entry 6
- **Tool**: Claude Code (Claude Sonnet 5)
- **Date**: 2026-08-21
- **Prompt**: Ran `python -m src.retrieval.bm25 --dataset ebnerd` and the "Scoring chunks" step showed a ~2.5 hour ETA. Asked whether this needed optimizing.
- **AI output used**: Yes
- **Diagnosis** (this took two attempts to fully resolve):
  1. First fix attempt: `evaluate()` densified every chunk's sparse score matrix (`.toarray()`) before taking top-K, which costs O(n_queries * corpus_size) regardless of chunk size — ~1.997M queries * 125,541 docs ≈ 250 billion cells, matching the ~2hr estimate. Rewrote to read top-K directly off the sparse CSR result per row, verified byte-identical recall@K vs. the old dense path on a 300-query MIND sample. This barely helped for EB-NeRD.
  2. Root cause was actually different: `bm25.py`'s `STOPWORDS` set was English-only, so it did nothing for EB-NeRD's Danish text — the single word "og" ("and") alone appeared in 96.6% of all 125,541 articles, so every query's score row was effectively dense (near-100% nonzero) no matter how the matrix math was done. Also found the tokenizer regex (`[a-z]{2,}`) was silently mangling Danish words containing æ/ø/å into meaningless fragments (e.g. "være" -> "re", "også" -> "ogs") — confirmed those fragments were literally showing up as top-frequency "terms".
  3. Even after adding Danish stopwords and fixing the regex, queries built from title+abstract of the last 10 clicked articles averaged ~120 unique terms each, whose combined posting-list union still covered ~94.5% of the corpus — an inherent consequence of long queries against a topically-related news corpus, not something stopword filtering alone can fix. Changed query construction to title-only of the last 5 articles (matches the assignment's own suggested design more literally), cutting average unique terms/query to ~22 and measured density to ~53.5%, bringing the full-corpus estimate down to ~39 minutes.
- **Human modifications**: none yet — verified each step with real data before handing back (document-frequency checks, byte-identical recall comparisons, timed samples projecting full-run duration). Pending the user's own full run.
- **Files affected**: `src/retrieval/bm25.py` — added `get_scores_batch_sparse` (sparse-native batch scoring), rewrote `evaluate()`'s scoring loop, added a Danish stopword list (`DANISH_STOPWORDS`, verified empirically against real document frequencies) alongside the existing English one, fixed the tokenizer regex to include æ/ø/å, and changed `_build_query_tokens` to title-only / last-5-articles.
- **Note**: because query construction changed, MIND's BM25 recall numbers collected earlier in this session are stale and need to be recollected with the same (now-consistent) query construction before the two datasets can be compared.

### Entry 6
- **Tool**: Claude Code (Claude Sonnet 5)
- **Date**: 2026-08-21 to 2026-08-22
- **Prompt**: `python -m src.retrieval.bm25 --dataset ebnerd` was projected at ~2 hours; after a first fix (avoid densifying scores in evaluate()) it was still projected at ~33-35 min. Asked to fix, then asked to cross-check whether the resulting recall numbers (0.0069/0.0081/0.0108) were actually correct rather than a symptom of a residual bug.
- **AI output used**: Yes
- **Diagnosis (speed)**: the real bottleneck wasn't the dense/sparse distinction at all — `STOPWORDS` in `bm25.py` was English-only, so it did nothing for EB-NeRD's Danish text; the most common terms in the corpus were Danish function words ("og" in 96.6% of all 125,541 articles). The tokenizer regex (`[a-z]{2,}`) was also ASCII-only, silently mangling words containing æ/ø/å into meaningless fragments (confirmed: "være" → "re", "også" → "ogs", both of which then showed up as if they were real high-frequency terms). Even after fixing both, average nonzero-score density per query was still ~52.6% of the corpus (65,992/125,541 docs) — an inherent property of BM25 over a topically homogeneous news corpus, not a bug — since even non-stopwords ("år"=year, "siger"=says) are extremely common in news writing. Shortened the query construction (title-only, last 5 clicks, down from title+abstract of the last 10) as a deliberate speed/recall tradeoff to make full-corpus scoring tractable at ~2M queries.
- **Diagnosis (correctness cross-check)**: verified the resulting recall values are real, not a bug — every val impression has non-empty click_history (avg 274 articles, ruling out a cold-start/empty-query artifact); EB-NeRD's actual candidate pool per impression is tiny (median 9 articles) while recall@K is correctly measured against the full 125K-article corpus per the assignment's Q2 spec, which is a much harder task than MIND's typically-larger candidate pools; no stemming is applied (standard for BM25) which disadvantages Danish's richer morphology more than English.
- **Human modifications**: reviewed and accepted; ran the corrected pipeline myself (35 min, completed cleanly) and the diagnostic checks were run to explain the resulting numbers, not to further alter code.
- **Files affected**: `src/retrieval/bm25.py` (Danish stopwords, tokenizer regex fix, shortened query construction — from the prior session's continuation; this entry documents the cross-check and reasoning, no further code changes made in this entry).

### Entry 7
- **Tool**: Claude Code (Claude Sonnet 5)
- **Date**: 2026-08-22
- **Prompt**: While generating `predictions/ebnerd_bm25_submission.zip`, macOS force-quit the whole session, reporting Claude Code (i.e. this session's Python subprocess) had grown to ~60GB RAM and had to be shut down. Asked to investigate and fix, then separately asked to add a progress bar to the same command.
- **AI output used**: Yes
- **Diagnosis**: a leftover diagnostic `pd.read_parquet()` call (run to sanity-check the newly-generated 13.5M-row, 7GB `impressions_competition_test.parquet`) was left running unattended in the background after being flagged as risky but not killed — it hit the same list-column-as-Python-objects problem fixed elsewhere in this codebase (see Entries 2-3), just never applied to `generate_preds.py`'s own data loading path. `src/pipeline/feature_store.py`'s `load_split()` (used by `generate_preds.py` to load the split to predict on) does a plain `pd.read_parquet()` on the whole file — fine for MIND's 2.37M-row competition test, but EB-NeRD's is 13.5M rows with a `click_history` list averaging ~200+ items per row, redundantly repeated per impression (not deduplicated per user) — materializing that as Python objects for every row simultaneously is what plausibly reached ~60GB.
- **Human modifications**: reviewed and accepted; verified both correctness (100,000 streamed lines checked byte-identical against the already-uploaded, Codabench-verified `mind_bm25_submission.zip`) and memory safety (sampled peak RSS every few batches on the real EB-NeRD file — 2.2GB → 3.5GB → 4.0GB → 4.5GB → 4.75GB → 4.89GB → 4.89GB → 4.89GB, clearly plateauing rather than growing with row count) before handing back for the user's own full run.
- **Files affected**:
  - `src/pipeline/feature_store.py` — split `load_split()`'s path-resolution logic out into a new `resolve_split_path()` so callers can stream a file instead of loading it whole; `load_split()` itself is unchanged for existing callers.
  - `src/submission/generate_preds.py` — rewritten to stream the impressions file in pyarrow row-group batches (`iter_impression_batches`, 100K rows/batch, only the 3 columns actually needed) instead of loading the full split via `load_split()`; added a `tqdm` progress bar over the streamed total (this was also the direct answer to the separate "add a progress bar" ask — the bar was already present from an earlier fix, but adding it properly required this rewrite anyway since the old version's progress bar sat downstream of the risky full load).
- **Note**: the actual crash cost no data — all previously-generated files (EB-NeRD's `impressions_competition_test.parquet`, both MIND submission zips) were verified intact afterward via a safe pyarrow column-projected read, not the risky full load that had just crashed.

---

_Add new entries below as you continue using AI tools._
