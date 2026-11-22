# Kurosawa visual similarity experiment

A local experiment using the Kurosawa films and other movies already available
in Curtain. The aim is to find useful frame comparisons for close viewing in a
film course project. Retrieved similarity is **not evidence of influence**.

Run with the existing Curtain environment:

```sh
../.venv/bin/python kurosawa_experiment.py
```

Open `results/pilot-v1/gallery.html` in a browser. Its images are local files in
the adjacent `frames/` directory, so keep that folder with the page. The gallery
works offline. No server or model runs when viewing it.

The gallery includes a diverse shortlist, each reference's top five later-film
matches, the three strongest pairs per later film from distinct source films,
and an earlier-film comparison group. Use **Best pairs per later film** for a
specific target title, including titles outside the references' top five.
Filter either film, mark useful
or superficial comparisons, and add notes. **Export review notes before closing
or refreshing**; marks live in memory for the current page session.

`contact-sheet.jpg` gives a quick view of twelve automatically ranked pairs.
`results.json` contains the complete frame mapping and rankings. `inventory.json`
records the films and sampling counts; `summary.json` records the run settings,
model checkpoint hash, and final counts. `global.npy` and `spatial.npy` follow
the row order in `results.json`'s `records` list.

## Collection and sampling

The source set is eleven Kurosawa titles in the existing catalog:

- Rashomon (1950)
- Ikiru (1952)
- Seven Samurai (1954)
- Throne of Blood (1957)
- Yojimbo (1961)
- Sanjuro (1962)
- High and Low (1963)
- Red Beard (1965)
- Kagemusha (1980)
- Ran (1985)
- Dreams (1990)

The additional catalog entry, `The_Hidden_Fortress_1958`, is **excluded from both
sets**. Its 4,909 frames span about 41 minutes at the extraction rate; inspected
frames contain interviews and film excerpts, consistent with a documentary
extra rather than the feature. Four samples per source title were inspected;
the eleven included titles appeared consistent with their labels. The audit sheet
is `results/source-audit.jpg`. This is a sample-based identity check, not a
complete verification of every stored frame or of other catalog titles.

The default pilot selects up to 96 shot midpoints per Kurosawa film and up to
16 per other film, evenly distributed across shot order. It uses the existing
shot maps for 473 included films (eleven source titles and 462 targets), omits midpoints in the opening/closing 2% of each
film, and rejects nearly blank frames. This is sparse sampling, **not an
exhaustive search of the seven million stored frames**. Shot durations vary,
so the samples are not uniformly spaced in time. A single midpoint can miss
important staging changes within a shot. Some logos, credits, or subtitles
can survive this filtering.

Frame-clock times use Curtain's filename-number / 2 convention and are marked
approximate. They are not independently checked against particular releases.
Every result preserves the original filename and path for verification.

## Features and retrieval

The pilot uses the locally cached DINOv2 ViT-S/14 checkpoint. It downloads
nothing and trains nothing. Frames are converted to RGB, resized with aspect
ratio preserved, and black-padded to 224 × 224. ImageNet normalization matches
the pretrained encoder's expected inputs.

Each frame has a normalized global CLS-token vector and four independently
normalized quadrant averages of the 16 × 16 patch-token grid. The spatial
vector concatenates the quadrant features with a factor of 1/2, so its norm
is one. The ranking is an experimental heuristic:

```text
score = 0.5 × global cosine + 0.5 × mean corresponding-quadrant cosine
```

This offers a coarse spatial signal alongside subject similarity. It is not
a trained or validated cinematography metric. Padding, frame aspect ratios,
monochrome images, similar subjects, and costumes can all affect the ranking.
The model does not analyze movement, cutting, narrative, or sound.

For each reference, retrieval returns at most five distinct films per time
group. Later-film matches require a strictly later release year. Earlier-film
matches require a strictly earlier year; same-year comparisons are omitted.
Release order alone establishes neither direct nor indirect influence.

The shortlist greedily retains high scores while using each reference/target
frame once, no more than three pairs per target film, and no more than twelve
per Kurosawa film. These caps provide variety, not human curation. No pair is
automatically labeled as historically documented influence.

## Reproducing and expanding

Per-film embeddings are cached in `results/embedding-cache/`, keyed by the
feature/preprocessing version, selected filenames, and frame file metadata.
Interrupted runs resume from completed films. The expected cached model files
are named explicitly in the script, and its checkpoint hash is checked before
using caches. If replacing the checkpoint, update `CHECKPOINT_SHA256` and
`VERSION`. If modifying feature extraction, change `VERSION` to invalidate
prior embeddings.

A denser run can use its own output directory:

```sh
../.venv/bin/python kurosawa_experiment.py \
  --source-shots 192 --target-shots 64 --out results/denser-v1
```

Changing sample counts changes the cache keys for affected movies. Results
stay inside this lab; the script only reads the existing Curtain frames,
catalog, shot maps, and cached model. It does not modify the live site.

The run checks vector norms/finiteness, frame paths, distinct-film retrieval,
score arithmetic, and release chronology. Those checks verify the experiment's
mechanics; usefulness still needs human review.

The next decision should be based on reviewing matches: first increase
sampling coverage if good compositions are being missed, then compare feature
choices or use reviewed pairs to tune ranking. More films expand the search
space but do not themselves improve the similarity model.

Model reference: [DINOv2 official repository](https://github.com/facebookresearch/dinov2).

Director credits were cross-checked against the
[Eye Filmmuseum Kurosawa programme](https://assets.eyefilm.nl/downloads/blocks/PBR2025-Eye-Filmmuseum-puts-Akira-Kurosawas-films-back-on-the-big-screen-EN.pdf)
and [JFDB's director entry](https://jfdb.jp/en/person/1372).
