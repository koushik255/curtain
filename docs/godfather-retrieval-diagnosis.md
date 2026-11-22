# Godfather II crop, codec and exact-frame diagnosis

The failure can occur with a frame that is definitely indexed. This audit
separates failure to retrieve it from failure to select it after retrieval.

The identified live configuration is `curtain_models/all-films-v1/best.pt`
with `.local/faiss-all-films-v1`, confirmed from the service drop-in and by
reproducing a live query's embedding score. `tight-v1` is available but is not
the live model. Diagnostic reports record checkpoint and catalog hashes.
Production model, index, ranking code and service configuration were not changed.

## Samples and labels

`benchmarks/data/godfather_web_v1.json` records six independently sourced web
images of seated characters, selected before retrieval, with source URLs and
crop centers. Five find the correct movie in all seven initial variants.
The very dark Empire image fails; its exact indexed counterpart has not been
established, so it was excluded from the controlled positive set.

For five successful web examples, the returned indexed frames were copied as
new controlled sources with explicit target IDs. Their exact identities are
known because the test images are generated from those stored frames. These
five frames cover only a few scenes and do not constitute independent samples
of the film or web. The controlled frame tests establish mechanisms; they do
not establish exact-frame accuracy on arbitrary web images.

Sources and reports live under `.local/benchmarks/godfather-web-v1/`,
`godfather-controls-v1/`, and `godfather-codecs-v1/`. The controls directory
contains `failure-examples.jpg`, showing each query, its known indexed source,
and the incorrect search result.

## Two demonstrated failure mechanisms

### Frame selection after successful retrieval

A crop containing 40% of each side (16% of the area), with changed colour/
shadows and a 160-pixel width, can retrieve the exact frame but select another.
For one Michael-on-the-couch example, `frame_003642.jpg` is a candidate at rank
7, but the site returns `frame_003681.jpg`, 19.5 seconds later at two frames
per second. The exact target is also in the final 20 candidates available to
an experimental reranker.

The site's `align.rerank` and `TrainedSearchEngine.refine` rank confirmed frames
by their SIFT inlier counts. Many matches can come from a static background,
and compression changes which points survive. More inliers therefore need
not mean a closer match to the exact source frame.

This also occurs without a crop or added colour change. After H.264 and VP9
encoding of the labelled `frame_003642.jpg`, the target has the best exact
embedding score within the film and among fetched candidates. SIFT instead
selects `frame_003638.jpg`. The AV1 example also ranks the exact target first
within the movie but selects that neighbouring frame after alignment.
Increasing encoder capacity does not directly repair this downstream rule.

### Small-region representation and retrieval

At 15% of each side (2.25% of the frame area), all five controlled queries fail
to retrieve their targets. These are deliberately severe crops, sometimes
only a mouth, torso or hand; they should not be presented as ordinary full-face
screenshots. The targets score poorly even in an exhaustive scan of all
24,253 Godfather II frames. This is not explained solely by FAISS approximation.

The index's finest tiles cover 25% of each side. A much smaller query is
resized to the same 192-pixel input, changing feature scale and context.
Re-encoding the known targets at a finer 8x8 grid increases similarity sharply
with the same encoder:

| Labelled frame | Current stored target score | Re-encoded 8x8 shared-trunk score |
|---|---:|---:|
| BAMF couch | 0.462 | 0.655 |
| No Film School chair | 0.456 | 0.768 |
| Hatena couch | 0.480 | 0.739 |
| Pinterest couch | 0.470 | 0.777 |
| Capsule sofa | 0.471 | 0.721 |

All five finer-grid target scores exceed the scores of their currently
returned wrong frames, including when those wrong frames are also re-encoded
at the finer grid. This is a pairwise representation diagnostic, not a rebuilt
global-index accuracy result. Finer grids introduce many more possible false
matches, so these numbers do not prove that all five queries would rank first
in a complete finer index.

Independent per-tile encoding also improves some targets over the shared-trunk
approximation, confirming that crop scale and reference extraction both merit
testing. Simply changing query resolution (80, 112, 144, 192 or 256 pixels)
does not recover any of these five as the top movie result.

## Actual codec tests and a ranking prototype

Three labelled frames were encoded with H.264, HEVC, VP9 and AV1, both as full
frames and followed by crop/colour/low-resolution changes. Each encoder receives
nine consecutive indexed frames at two fps; these are real encoder round-trips
but not native 24-fps footage or independent mastered releases. Fixed settings
and exact source IDs are stored in `godfather-codecs-v1/manifest.json`.
CRFs across codecs are not equivalent visual quality or bitrate.

An offline prototype searches scale and position using normalized grayscale
template correlation, then reorders only the candidates already returned.
It never injects a labelled target or changes confidence flags.

| Targeted controlled subset | Existing exact #1 | Template experiment exact #1 |
|---|---:|---:|
| Five frames, seven initial transformations | 35/35 | 35/35 |
| Additional grade, low-resolution and extreme-crop cases | 7/15 | 10/15 |
| Actual codecs, full frame | 9/12 | 12/12 |
| Actual codecs plus crop/grade/low resolution | 4/12 | 12/12 |

The remaining five extreme-crop failures have no correct frame among the
returned candidates. A reranker cannot recover an unavailable candidate.
These are small, correlated, synthetic diagnostics. The prototype assumes
scale and translation; it is not a verified production fix for rotation,
perspective, subtitles, remastering or absent-movie queries. Its correlation
score is not a calibrated probability.

A broader sanity check used the frozen top-five candidate lists from the
110-query validation run of 2026-10-26. With an experimental correlation gate
of 0.95 (keep the original result below it), exact retrieval changes from
96/110 to 99/110, recovering two tight crops and one heavily compressed query,
with no correct-to-incorrect changes in this set. This is candidate reranking
on a saved run, not a fresh full-pipeline deployment benchmark. The gate is
not a match-confidence threshold. This check does not establish behaviour on
new web releases, absent movies or the 350-query stacked suite; runtime has
not been benchmarked.

## Model comparisons

On the same 15 harder controlled queries, the complete `tight-v1` pipeline
gets 8/15 exact, compared with 7/15 for the live `all-films-v1`. All five
extreme crops still fail.

A pretrained SSCD model was compared on exactly the same film-only candidate
set as the live encoder: 12,127 cached one-fps frames plus the two labelled
target frames absent from that cache. Cache correspondence was checked against
fresh embeddings at three positions. The live encoder uses tiles and its
normal query readings; SSCD uses one whole-frame vector per reference/query.
The live encoder ranks the exact target first for 6/15, SSCD for 5/15. SSCD
helps some graded crops and worsens several low-resolution cases. This is not
a controlled model-size ablation or a global retrieval benchmark, and does not
rule out fine-tuning a larger pretrained backbone for the task.

## What to improve

1. Evaluate a better final frame comparison that uses aligned pixel/appearance
   evidence alongside local feature matches, rather than maximizing SIFT inlier
   count. Preserve the existing pipeline as a comparator, include difficult
   independent web examples, and measure latency and false matches before
   production use.
2. Test multiple reference scales on a bounded index with realistic distractors.
   An 8x8 grid produces 225 overlapping tiles plus a global vector, versus the
   current 49 plus one: about 4.5x the vectors if applied everywhere. Test
   selective scales or a coarse-to-fine retrieval design before a full rebuild.
3. Train region matching at the actual indexed reference scales and with the
   actual shared-trunk pooling path. Include varied crop aspect ratios and
   compound codec, grade and resolution changes. Continue using nearby-frame
   negatives, and explicitly evaluate confusion within a shot.
4. Compare pretrained ResNet-18/50 using that improved recipe. More capacity
   remains an experiment; the demonstrated downstream ranking failure and
   crop-scale mismatch need their own measurements.

## Reproduce the principal diagnostics

```sh
.venv/bin/python -m benchmarks.web_screenshots fetch \
  --manifest benchmarks/data/godfather_web_v1.json \
  --out .local/benchmarks/godfather-web-v1

.venv/bin/python -m benchmarks.crop_diagnosis

.venv/bin/python -m benchmarks.crop_diagnosis \
  --manifest .local/benchmarks/godfather-controls-v1/manifest.json \
  --out .local/benchmarks/godfather-controls-v1 \
  --variants crop15 crop40-release crop40-release-lowres

.venv/bin/python -m benchmarks.crop_diagnosis \
  --manifest .local/benchmarks/godfather-codecs-v1/manifest.json \
  --out .local/benchmarks/godfather-codecs-v1 --variants web

.venv/bin/python -m benchmarks.template_rerank \
  .local/benchmarks/godfather-codecs-v1/diagnosis-all-films-v1-20261103-145535.json \
  --out .local/benchmarks/godfather-codecs-v1/template-rerank.json
```

The labelled controls and encoded clips must already have been generated for
the last three commands; their local manifests describe their provenance.
