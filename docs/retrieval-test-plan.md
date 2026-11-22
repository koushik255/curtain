# Tests after excluding the three bad catalog sources

The remaining failures should stay in the benchmark. A retrieval failure is
not evidence that a movie's footage should be excluded. Exclusions require
independent confirmation of wrong footage, as for the three audited entries.

## Completed development diagnostic

`benchmarks/candidate_experiment.py` compares three settings against the live
checkpoint and index, including the new source-exclusion policy:

| Setting | FAISS nprobe | Tiles per query view | Maximum aligned candidates | Correct movie #1 |
|---|---:|---:|---:|---:|
| Current search | 64 | 512 | 40 | 12/15 |
| Deeper alignment | 64 | 512 | 200 | 13/15 |
| Wider retrieval and deeper alignment | 256 | 2048 | 200 | 12/15 |

The fixed selection contains the three previously failed documented
development screenshots whose movie entries remain active, plus twelve
previous successes chosen with seed 20261103. There are no test-partition
queries. All twelve controls retain correct movie identification with both
experimental settings. This is a failure-enriched diagnostic, not a general
accuracy estimate, and correct movie identification does not imply exact
frame identity. No production ranking settings were changed.

Deeper alignment recovers the My Night at Maud's skyline. Visual inspection
confirms that the new result depicts the query composition. The matching movie
starts at candidate rank 195, beyond the current 40-frame alignment budget.
Refinement returns `frame_000244.jpg`, which has 36 SIFT inliers.

Wider approximate retrieval increases the candidate pool from 4,967 to 14,604
frames and moves the first matching-movie frame to rank 209, just outside its
200-frame alignment cutoff. Therefore increasing the approximate search
budget without coordinating the subsequent budget can lose a recovery. The
specific failure is a cutoff issue, not evidence of a false geometric match.

The other two diagnostic failures remain: the Maud's book-title insert and
the very dark Once Upon a Time in America face. Both still return visually
different material with very few agreeing local features. Their exact indexed
counterparts have not been independently annotated. More candidates alone did
not establish a fix.

Settings run in a fixed order and share feature/page caches, so timings from
this script are not a fair standalone latency comparison. The final ranking
still uses existing SIFT confirmation and temporal refinement; this experiment
does not validate that verifier or measure exact-frame regressions.

Local artifacts:

- `.local/benchmarks/online-v1/candidate-experiment-selection.json`: fixed IDs.
- `.local/benchmarks/online-v1/candidate-experiment/config.json`: settings,
  model checksum, policy checksum, and implementation checksum.
- `.local/benchmarks/online-v1/candidate-experiment/results.json`: full returned
  candidates and movie-level outcomes for all three settings.
- `.local/benchmarks/online-v1/candidate-experiment/comparison-00.jpg`: the three
  difficult queries beside all three methods.
- `.local/benchmarks/online-v1/candidate-experiment/skyline-candidate-trace.json`:
  the independent candidate-rank trace.

```sh
.venv/bin/python -m benchmarks.candidate_experiment
```

The local fixed selection and original query files must already exist.
The completed run verified that query checksums match the original benchmark,
every case belongs to development, no excluded source appears among results,
and exact-frame truth remains unknown rather than inferred from scores.

## Next acceptance tests

1. **Candidate recall:** expand the promising deeper-alignment experiment to
   the complete usable development cohort. Measure recoveries, regressions,
   and independently timed cost. Consider a second pass on uncertain queries
   rather than paying the larger budget on every upload. Also test reference
   bar normalization; this diagnostic has not isolated its causal effect.
2. **Frame selection:** annotate the closest available frame or correct shot
   independently for real web screenshots. Evaluate aligned appearance over
   the subject/query region. Retain the known Black Swan and Butch Cassidy
   template-reranker regressions as negative controls. Reject improvements
   that gain movie accuracy by selecting worse frames within the movie.
3. **Appearance robustness:** create labelled crops and compound edits from
   separate indexed sources, with the provided editing tool. Test crop,
   resolution, color, and compression individually and together. Keep these
   mechanism tests separate from independently sourced web-image accuracy.
4. **Abstention:** include absent footage, unrelated pictures, and ambiguous
   very dark or text-only queries. Measure confident false matches and valid
   matches rejected. An inlier threshold or NCC threshold is not ground truth.
5. **Backbones:** compare a larger pretrained encoder only after the preceding
   metrics exist, using the same queries and reference coverage. Include
   inference cost and reindexing/storage cost, not only training duration.

Tune on development only. Freeze the candidate policy, ranking method, and
thresholds before a final test-partition evaluation. The earlier test results
remain reported; they must not be used to select these new settings.
