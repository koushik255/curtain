# Independent web-image benchmark, 2026-11-03

The controlled 24/24 codec diagnostic does not describe general web retrieval.
This run collected and tested 475 independently published scene images for
95 of the 474 indexed movie labels. The unchanged live search identifies the
source movie for 396/475 images (83.4%); the fixed template reranker also gets
396/475. These are **movie-identification counts, not exact-frame accuracy**.

Source provenance limits the usable screenshot benchmark: 319 images have
documented frame-capture sources, covering 69 movies. The other 156 scene
images have unverified still-camera versus video-frame provenance. They are
retained as a separate discovery cohort, never asserted to be exact-match
positives. Five documented screenshots per movie are available for 61 of the
95 selected labels; this acquisition has not established that coverage for
the other 34. Publicity-style scene stills can depict the movie without having
an identical frame anywhere in its released footage.

## Results

| Source cohort | Images | Existing movie #1 | Template movie #1 | Correct movie somewhere in returned 20 |
|---|---:|---:|---:|---:|
| Documented frame captures, development | 184 | 171 (92.9%) | 171 | 171 |
| Documented frame captures, test | 135 | 120 (88.9%) | 120 | 122 |
| Documented frame captures, all | 319 | 291 (91.2%) | 291 | 293 |
| Unverified scene stills, separate discovery cohort | 156 | 105 (67.3%) | 105 | 112 |
| Entire acquisition, including unverified stills | 475 | 396 (83.4%) | 396 | 405 |

The reranker changes the first frame in 24 cases, including 22 documented
captures, and changes no movie-identification outcome. This does not establish
that its frame changes are correct. Randomized A/B contact-sheet inspection
finds clear improvements and regressions; most neighboring-frame comparisons
remain unresolved. Exact-frame truth stays null. In particular, the Butch
Cassidy sunset query is moved from two large foreground riders to a different
composition with small distant riders, despite passing the 0.95 NCC gate.
The Black Swan dance-studio query also has a visibly poorer body-position
match after reranking. The prototype should not be promoted on the basis of
its controlled codec result.

On documented screenshots, 26 of the 28 wrong-movie results do not contain
even the correct movie in the returned 20. Five wrong-movie first results
exceed the current 15-SIFT-inlier confirmation threshold. The corresponding
counts for the complete, noisier discovery cohort are 70/79 and 17.
The confirmation heuristic is not a calibrated guarantee of scene identity.

CPU median baseline latency is about 1.62 seconds, with an additional
1.12 seconds for the template experiment over the same returned candidates.
These are indicative timings under concurrent local work, not a controlled
latency benchmark.

## Three catalog problems, not encoder-capacity failures

Native indexed frames and local acquisition metadata establish:

| Catalog label | Indexed material | Evidence |
|---|---|---|
| Mommy (2014) | Goodnight Mommy (2014) | Bandaged mother and twin boys; source selection explicitly says Goodnight Mommy. |
| The Return (2003) | The Return of the King (2003) | Gandalf and Frodo; source selection explicitly names the extended Lord of the Rings film. |
| The Hidden Fortress (1958) | Color interviews / supplementary material | Native timeline samples contain interviews, not the black-and-white feature; only 4,909 frames, about 41 minutes at 2 fps. |

These affect 15 of the 319 documented queries. All 15 remain in the raw
results. Excluding these specific post hoc catalog findings would give
291/304 movie identifications, but that is a diagnostic decomposition,
not a replacement headline score or an exhaustive catalog correction.

The sibling ingest repository's `common.find_video` selects the largest video
by file size. Its completeness check compares extraction against the chosen
file's duration, not an independently verified feature runtime. Stored source
selections demonstrate title collisions for Mommy and The Return. The exact
upstream filename-selection history for Hidden Fortress was not recovered;
the current largest-file rule alone does not prove how that bad import happened.
No sibling-repository or production files were edited.

For comparison, [Cannes documents Dolan's square-format Mommy](https://www.festival-cannes.com/en/2014/competition-xavier-dolan-s-maternal-eulogy/),
while [the Goodnight Mommy producer identifies the Austrian feature and its twins](https://www.ulrichseidl.com/en/ulrich-seidl-filmproduktion/films-in-distribution/goodnight-mommy).

## A genuine candidate-recall failure

The My Night at Maud's city panorama has matching native footage. A separate
known-movie exhaustive diagnostic finds `frame_000244.jpg` with 36 SIFT inliers,
and visual inspection confirms the skyline. But the first candidate from that
movie is only rank 195 before SIFT; the normal pipeline aligns at most 40 and
returns 20. Reranking the returned 20 cannot recover it. This diagnostic uses
the movie label as an oracle and is not counted as an unassisted success.

The source panorama omits the pillarboxing present in the indexed frame.
Reference preprocessing, candidate budget, and geometry-aware verification
deserve testing before assuming the only issue is backbone size. This run
does not isolate a codec-only cause or prove that a larger model would fix it.

## Protocol and limits

- Movie IDs were sorted case-insensitively and positions 0, 5, 10, ... selected,
  including the last partial block. Seed: 20261103.
- A movie-level development/test split of 57/38 was fixed before acquisition.
  These are benchmark partitions, not the encoder's training split. Of the
  475 queries, 400 belong to training movies, 60 to unseen movies, and 15 to
  validation movies according to checkpoint provenance.
- The complete discovered candidate pools and deterministic shuffled order
  were recorded. Replacement followed download failure, duplicate detection,
  or source review, never retrieval failure. Posters, ambiguous posed photos,
  wrong films, title cards, and repeated crops were excluded before retrieval.
  A mislabeled Ali gallery containing Easy Rider frames was caught this way.
- A subsequent conservative source-provenance audit applies the same
  source-based rule to successful and failed images. It creates report strata;
  it does not replace failed queries or modify the frozen cohort.
- There are 475 unique original-byte hashes. The frozen source list includes
  page URLs, image URLs, checksums, dimensions, review decisions, and partitions.
  Downloads, cached pages, rejected originals, and attempt ledgers remain local.
- Queries use a standardized 640-pixel JPEG-90 upload proxy. This approximates
  the UI; browser resizing/encoding can differ, and the UI retains an original
  when it is smaller than the recompressed file.
- The model is `curtain_models/all-films-v1/best.pt`, using the full
  `.local/faiss-all-films-v1` index. Checkpoint and catalog hashes are recorded.
- NCC uses the same returned 20 candidates, its previously chosen 0.95 gate,
  and a frozen implementation hash. No labelled candidate is injected.
- Dedicated galleries account for 305 images. They favor clean full frames.
  The cohort is not a random sample of all internet screenshots, user crops,
  or remasters. Scene stills and absent/mislabeled footage cannot silently
  become positive frame-match examples.

## Reproducible appearance-editing tool

`benchmarks.web_appearance` fits descriptive measurements on the 184 documented
development captures only. Its 156 usable aligned pairs describe approximate
crop coverage and luminance changes relative to retrieved native frames.
These measurements are biased toward successful alignments, and do not
establish exact timestamps or the source codec. Median measured coverage is
about 99%, demonstrating the clean-gallery bias.

The tool exports actual JPEG/WebP round trips plus crops, color/gamma changes,
blur, resizing, black bars, subtitles, and player overlays. Every variant
retains its source checksum, source-space crop box, recipe, and seed. An
optional focus point centers crops on a subject. Stress presets are explicitly
hypotheses, separate from observed measurements. Very small crops can remove
most identifying information; they are not representative full-face queries.

All benchmark movies are refused as augmentation-export sources. The example
uses a known Godfather II frame outside the 95-movie sample. Training defaults,
model weights, the production index, and ranking behavior were not changed.

## Artifacts and reproduction

Local results are in `.local/benchmarks/online-v1/`:

- `review.html`: filterable source/query/before/after comparisons, defaulting
  to documented captures; manual scene review can be exported separately.
- `summary.json`, `results.jsonl`, `run-config.json`: complete scores and traces.
- `source-provenance.json`, `catalog-audit.json`: source and index findings.
- `documented-diagnosis.json`, `development-diagnosis.json`: separate oracle
  diagnostics, never mixed into accuracy.
- `result-sheets-changes/`, `blind-preferences.json`: A/B comparisons and
  conservative visual preferences, not exact-frame labels.
- `appearance-lab/preview.jpg`, `appearance-lab/edits.json`: editable examples
  with preserved source labels and full recipes.

The portable source snapshot is `benchmarks/data/online_queries_v1.json`.
Restoration checks original bytes and refuses changed remote assets rather
than silently substituting them. Some public URLs may later disappear.

```sh
.venv/bin/python -m benchmarks.online_benchmark restore \
  --out .local/benchmarks/online-v1-replay --workers 8
CURTAIN_SOURCE_EXCLUSIONS=benchmarks/data/no_exclusions.json .venv/bin/python -m benchmarks.online_benchmark run \
  --out .local/benchmarks/online-v1-replay
.venv/bin/python -m benchmarks.online_benchmark report \
  --out .local/benchmarks/online-v1-replay
.venv/bin/python -m benchmarks.online_review report
.venv/bin/python -m benchmarks.web_appearance profile
.venv/bin/python -m benchmarks.web_appearance variants \
  --source .local/benchmarks/godfather-controls-v1/originals/nofilm-chair \
  --movie The_Godfather_Part_II_1974 --focus-x .45 --focus-y .48
.venv/bin/python -m unittest benchmarks.test_online_benchmark
```

Priority: repair verified catalog mismatches and validate feature identity at
ingest; improve candidate recall and reference normalization; replace raw
inlier-count/small-patch ranking with broader aligned appearance evidence and
proper negative calibration. Then compare larger pretrained backbones on the
same fixed sources, including inference and reindexing costs as well as training.

Subsequent operational change: the three confirmed bad source spans are now
excluded by `rcmindex/excluded_sources.json` before candidate rescoring and
are omitted from the searchable catalog. Raw files, weights, and stored frame
IDs remain intact. The benchmark results above precede this exclusion; use
the explicit empty policy shown above to reproduce that baseline. Exclusion
does not remove the old examples from already-trained weights or reclaim the
stored FAISS vectors. A future clean rebuild can do that separately.
The running service was restarted and verified at 471 searchable movies and
6,978,930 searchable frames, excluding 48,001 stored frames. Nine unit tests
and a full-engine search/routing smoke test pass; the live catalog omits all
three entries. Stored approximate-search slots can still be occupied by their
vectors before filtering, so a future physical index rebuild could improve
efficiency. This change has not been assigned a new full-benchmark accuracy.
