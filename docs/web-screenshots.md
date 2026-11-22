# Independent web screenshots

The acceptance case is a screenshot found on the web, from another copy of a
film, retrieving its matching indexed frame. This needs evaluation independent
of synthetic queries generated from our indexed frames.

`benchmarks/data/web_screenshots_v1.json` records 20 image-search selections,
their source pages, image URLs and expected movies. Eighteen downloaded:
16 images across six indexed films and two *Eyes Wide Shut* controls. Two
hosts failed; those cases remain in the manifest and report. These are a small
convenience sample, not a representative accuracy estimate. These six films
have also been used repeatedly in existing benchmarks, so this is not an
untouched test split.

## Run

With the site running locally:

```sh
.venv/bin/python -m benchmarks.web_screenshots fetch
.venv/bin/python -m benchmarks.web_screenshots run
.venv/bin/python -m benchmarks.web_screenshots sheet \
  --results .local/benchmarks/web-screenshots-v1/results-YYYYMMDD-HHMMSS.json
```

The runner saves immutable original downloads, SHA-256 hashes, per-query API
responses, the live catalog/health counts, a Markdown report and an optional
query/result contact sheet under `.local/benchmarks/web-screenshots-v1/`.
`--base-url` selects a different running deployment. The health endpoint does
not expose a checkpoint hash, so this report identifies the live service and
time, not a cryptographically pinned model/index pair.

Four variants are evaluated:

- Original downloaded bytes, sent directly to the search API.
- A browser-like upload: longest side at most 640 pixels, JPEG quality 90.
- JPEG quality 20 at the same dimensions.
- WebP quality 30 at the same dimensions.

The browser approximation uses Pillow; browser canvas resize/JPEG can differ,
and the real page sometimes keeps a smaller original file. The quality settings
are not equivalent bitrate or equivalent perceptual quality. This tests added
image compression, not the source video's H.264/HEVC/AV1/VP9 codec. The separate
`encodings` and `stacked` suites test video encoding recipes.

## Results, 2026-11-03

The live service reported 474 movies, 7,026,931 frames and 351,346,550 vectors.
The saved report is `results-20261103-042417.json` (and `.md`/`.jpg`).

| Variant | Right movie, rank 1 | Right movie, top 5 | Correct movie with SIFT confirmation |
|---|---:|---:|---:|
| Original | 14/16 | 14/16 | 14/16 |
| Browser-like | 14/16 | 14/16 | 14/16 |
| JPEG 20 | 14/16 | 14/16 | 14/16 |
| WebP 30 | 14/16 | 14/16 | 14/16 |

These are the same 16 images in each row, not 64 independent examples.
Exact frames have not been independently labelled. A geometric match can
confirm the scene while preferring another frame with the same background.
Do not report these counts as exact-frame accuracy.

The same two images failed in all four variants: `gladiator-comingsoon` and
`handmaiden-review`. Both were flagged weak. An additional diagnostic searched
every frame of their known movies with each checkpoint's matching stored
embeddings (`all-films-v1` and `tight-v1`), bypassing global FAISS candidate
selection. Their top candidates resemble the relevant scenes, but none of the
top 12 per model/query reached 15 SIFT inliers. The diagnosis and candidate
sheets are saved as `film-only-diagnosis.json` and `diagnose-*.jpg`.
The images may be publicity stills or alternate viewpoints; this has not been
verified. Their exact source frames have not been established as present in
the indexed release. This diagnostic does not establish absence from the film.

Frame selection is less stable than movie recognition: among the 14 successful
original queries, browser-like conversion changed the top frame for three,
JPEG 20 for five, and WebP 30 for two. The largest shift was nine stored frames
(4.5 seconds at two frames per second). An unchanged frame is not necessarily
the correct frame either. This needs manual frame labels and temporal ranking
evaluation, rather than using movie accuracy as a substitute.

*Eyes Wide Shut* is absent from the live catalog and both saved 474-movie
catalogs. Its eight variant searches returned unrelated films. Seven were
flagged weak; the JPEG 20 version of `eyes-bluray` received 17 SIFT inliers and
was incorrectly marked confirmed. It cannot have a true match in this index.
Two negative images are insufficient to tune a replacement confidence rule.

## Implications

1. Preserve independent web queries for evaluation. Add manual labels for
   nearest stored frame, same scene, publicity/alternate shot and absent movie;
   do not train on these evaluation images.
2. Improve frame ranking separately from movie retrieval. Compare nearby
   candidates using alignment residuals/appearance as well as SIFT inlier
   count; static backgrounds can agree across multiple frames. Existing
   nearby-frame contrastive negatives should remain part of the comparison.
3. Evaluate rejection of absent movies and unavailable frames on a larger
   negative set before changing the production confidence rule.
4. Compare pretrained ResNet-18, pretrained ResNet-50 and a copy-detection
   baseline on the same labelled web queries, using each model's own reference
   embeddings. Measure candidate recall, frame ranking and rejection separately.
   This run compares compression variants on the live model; it does not test
   a larger backbone.
