# Kurosawa experiment — progress and handoff

Updated: 2026-11-03 UTC.

## User's goal and agreed scope

The user is preparing a film course final about Akira Kurosawa's influence on
other movies. They want to experiment with the movies already available in
Curtain, discover promising visual comparisons, and improve the approach based
on what those comparisons show. They have not asked to acquire more films,
train a classifier, publish anything, or modify the live Curtain site.

The immediate deliverable should be useful, reviewable frame pairs for a film
analysis project. Visual similarity is a research lead, not proof of influence.
Documented influence requires separate historical sources and close viewing.

## Current state: the pilot has completed

The final pilot ran successfully and produced a local gallery and result data:

| Item | Final value |
| --- | --- |
| Original Curtain catalog | 474 titles |
| Included Kurosawa reference films | 11 |
| Other comparison films | 462 |
| Excluded mislabeled title | 1 |
| Embedded Kurosawa reference frames | 1,050 |
| Embedded comparison frames | 7,168 |
| Total embedded frames | 8,218 |
| Selected frames before blank filtering | 8,446 |
| Frames rejected by near-blank filter | 228 |
| Automatically selected diverse shortlist | 100 pairs |
| Later films with per-film matches | 424 |
| Final result timestamp | 2026-11-03T16:04:42.923287+00:00 |

**This is sparse sampling, not an exhaustive search of all stored frames.**
The default selects up to 96 distinct shot midpoints per reference film and
16 per other film. The source/comparison sets exclude other Kurosawa films
from each other, so matches do not accidentally compare Kurosawa with himself.

The final counts come from `results/pilot-v1/summary.json`. Earlier messages
mentioned ten, then nine references; those were provisional. A systematic
filmography check found two additional available titles, *Dreams* and *Red
Beard*. **Use the final eleven-title set below.**

## Source audit and important data issue

The included films are *Rashomon*, *Ikiru*, *Seven Samurai*, *Throne of Blood*,
*Yojimbo*, *Sanjuro*, *High and Low*, *Red Beard*, *Kagemusha*, *Ran*, and
*Dreams*. Four frames spread across each indexed source were visually checked
and appeared consistent with the labeled features. This is a sample-based
identity check, not complete validation of every frame or film edition.

`The_Hidden_Fortress_1958` is excluded from **both** sets. Its folder contains
4,909 frames, about 41 minutes at the extraction rate. Inspected frames contain
color interviews and film excerpts, consistent with a documentary extra
rather than the feature. Do not silently reintroduce it as Kurosawa feature
footage or infer that we have the full film. Its precise documentary identity
has not been established.

Evidence is in `results/source-audit.jpg`, `results/source-audit.json`, and
`results/additional-source-audit.jpg`. The comparison titles have not all
received equivalent identity checks; verify any candidate used in the final.

## Files and environment

Working directory: `/home/koushik/curtain/curtain_lab`.

Existing Curtain checkout/data: `/home/koushik/curtain`.

| Path relative to the working directory | Purpose |
| --- | --- |
| `kurosawa_experiment.py` | Offline sampling, embeddings, retrieval, validation, and report generation |
| `README.md` | Method, limitations, commands, and expansion instructions |
| `check_gallery.mjs` | Headless Chromium checks for the actual report |
| `results/pilot-v1/gallery.html` | Browsable local report, about 5 MB |
| `results/pilot-v1/frames/` | Local gallery images; keep with the HTML |
| `results/pilot-v1/contact-sheet.jpg` | First twelve automatically ranked shortlist pairs |
| `results/pilot-v1/results.json` | Records, per-reference rankings, shortlist, and per-film matches |
| `results/pilot-v1/summary.json` | Final counts, method, model hash, and provenance |
| `results/pilot-v1/inventory.json` | Catalog coverage, exclusions, and sampling |
| `results/pilot-v1/global.npy` / `spatial.npy` | Feature arrays aligned with `results.json` records |
| `results/embedding-cache/` | Completed per-film caches; retain to avoid unnecessary inference |
| `NEXT_AGENT_PROMPT.md` | Copyable continuation prompt |

Use the existing Python environment, `../.venv/bin/python`. PyTorch, NumPy,
Pillow, and the small DINOv2 model/repository are already present. No GPU was
available through PyTorch during this session. The initial CPU run took about
13 minutes; the corrected run reused caches and completed in about 33 seconds.
That cached time is not a fresh-indexing benchmark.

All new code and outputs for this task are inside `curtain_lab`. Existing
Curtain code, frames, indexes, and the live site were not modified. The parent
checkout already has unrelated modified/untracked files. Preserve those; do
not reset, clean, or commit them as part of this experiment. Generated results
and Python caches are ignored by the lab's `.gitignore` but remain on disk.

## Method already implemented

- Use Curtain's existing shot maps and catalog, preserving original filenames.
- Skip shot midpoints in the first/last 2% of indexed material; reject nearly
  blank images. This does not reliably remove all credits, subtitles, or logos.
- Embed RGB frames with cached DINOv2 ViT-S/14, preserving aspect ratio with
  black padding to 224 × 224.
- Rank by an experimental blend: 50% global CLS-token cosine and 50% mean
  cosine across four corresponding patch quadrants. No model was trained.
- Return up to five distinct target films per reference in each of two groups:
  strictly later release years and strictly earlier release years. Same-year
  matches are omitted. Earlier films are a comparison group, not a statistical
  validation or automatic claim about who influenced whom.
- Also retain the three strongest pairs per later film from different Kurosawa
  source films. This supports looking up a target outside the general top five.
- Build a 100-pair shortlist with diversity caps. It is automatically ranked,
  not human curated or historically validated.

Frame times are approximate filename-number / 2 clocks, not independently
verified timestamps for a particular release. Scores are ranking similarities,
not probabilities of influence. Color, aspect ratio, matching subjects, and
costumes can influence them. Motion, editing, narrative, and sound are not
modeled. More films expand coverage; they do not automatically improve ranking.

## Verification completed and still pending

Completed on the final run:

- Python compilation and JavaScript syntax checks.
- Runtime checks of vector norms/finiteness, frame paths, distinct-film
  retrieval, score arithmetic, and chronology.
- Sample-based identity inspection of the eleven reference titles and the
  excluded documentary folder.

A headless-browser check passed against a **small fixture** for initial
rendering, source filtering, review notes, earlier-film chronology, and image
loading. The check script was subsequently expanded to cover target-film
filtering and downloading review notes; those additions are not yet confirmed.

**The real gallery has not received a confirmed browser-check result, and the
actual ranked contact sheet has not received a confirmed visual review.**
An attempt to view the contact sheet and run the real browser check was
interrupted by the user before tool results returned. At handoff verification,
`results/pilot-v1/gallery-preview.png` did not exist. Do not claim the real
gallery passed, or that the strongest matches are useful, without checking.
An interrupted browser-check process may need checking before another launch.

Chromium's startup failed inside the execution sandbox with
`sandbox_host_linux.cc ... shutdown: Operation not permitted`. Running the
fixture check outside the sandbox succeeded. Follow the environment's approval
rules if the real browser check needs the same escalation; do not circumvent
the sandbox. The available binary is:

```text
/home/koushik/.cache/ms-playwright/chromium_headless_shell-1234/chrome-headless-shell-linux64/chrome-headless-shell
```

## Recommended next steps, in order

1. **Inspect existing outputs first; do not rebuild the index.** Read the
   README and summary, view `contact-sheet.jpg`, and inspect more shortlist
   pairs with their original frames. Assess whether similarities concern
   composition/blocking/depth or only subjects/costumes/monochrome appearance.
2. **Finish the real gallery check.** From the lab run `node check_gallery.mjs`
   with the appropriate execution permissions. It checks rendering, source and
   target filters, review marks, review-note download, image loading, chronology,
   and saves `results/pilot-v1/gallery-preview.png`. Inspect that screenshot.
3. **Review a manageable set of matches.** Record specific observations and
   useful/superficial/unsure judgments in a new Markdown or JSON artifact.
   Gallery notes are session-only: export before refreshing or closing.
   Use “Best pairs per later film” for focused exploration such as *Star Wars*.
4. **Check the surrounding shots and historical evidence.** Verify identities
   and approximate times before using results in the course final. Browse for
   reliable director interviews or film scholarship for any influence claim.
   Keep documented influence separate from resemblance or speculation.
5. **Choose one improvement based on those observations.** If relevant moments
   are missing, expand sampling in selected films or use multiple frames per
   shot. If ranking is superficial, compare global-only, spatial-only, or
   grayscale variants on the same reviewed cases. Do not assume training is
   needed before establishing useful labels and a baseline.

An optional denser run is documented in README. Use a separate result folder
to preserve the pilot. Broadening sampling is a follow-up decision, not work
that has already been completed. Do not acquire additional movies or deploy
changes without a request.

Some leading automatic pairs are *High and Low* → *Prisoners*, *Ikiru* →
*All About Lily Chou-Chou*, and *Sanjuro* → *The Sword of Doom*. These are
**unreviewed retrieval results**, not confirmed influences or recommendations
for inclusion in the project.

## Copyable continuation prompt

See `NEXT_AGENT_PROMPT.md`. Start by reading this handoff and the existing
outputs, then finish visual review and validation before proposing new model
work.
