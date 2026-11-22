# Pilot review — Kurosawa visual similarity (pilot-v1)

Reviewed 2026-11-03. Covers the contact sheet, shortlist pairs #0–59 (60 of
100), and the "best pairs per later film" for 12 titles with a possible
Kurosawa connection. Judgments are from single frames, by eye. No surrounding
shots have been checked yet. Times are approximate (filename number / 2).

## Verdict

**The pilot finds lighting and framing, not staging.** The ranking rewards:
(1) dark, low-key, near-monochrome images; (2) a single figure in medium
close-up, roughly centred or left-weighted; (3) matching subjects such as
cars, crowds, newspapers or cavalry. In the 60 shortlist pairs reviewed, about
**8 are useful staging or depth leads**, about 10 are subject matches that
could support a point with more context, and **most (~40) are superficial**:
any two dark medium close-ups score 0.77–0.81.

**Coverage is the main limit, more than the scorer.** Only 16 frames per
comparison film were sampled, so the same comparison frame keeps coming back.
*Star Wars* ~00:34:58 is the best match for three different Kurosawa films,
and so is *For a Few Dollars More* ~01:33:27. The search could not find the
relevant moments because they were never sampled. That decides the next
experiment (below).

## Browser check (real gallery): passed

`node check_gallery.mjs` against `results/pilot-v1/gallery.html`: 12 cards
render; 12 source and 463 target options (11 + 462 plus "All"); source filter,
target filter (*Best pairs per later film*), review marks, and review-note
export (1 pair) all work; earlier-film view chronology is correct; visible
images load; 0 JS exceptions. Screenshot: `results/pilot-v1/gallery-preview.png`
(inspected; layout and images look right). Chromium ran without needing the
sandbox escalation this time.

## Useful leads (staging, depth, or blocking)

| # | Kurosawa | Other film | What is shared |
|---|---|---|---|
| 0 | *High and Low* ~01:33:10 `High_and_Low_1963/frame_011179.jpg` | *Prisoners* (2013) ~01:23:34 `Prisoners_2013/frame_010029.jpg` | Two detectives filmed from the back seat, heads framing the windscreen, road centred. Real composition match, but a common setup for car scenes. |
| 9 | *Sanjuro* ~01:15:48 `Sanjuro_1962/frame_009095.jpg` | *The Thing* (1982) ~00:30:25 `The_Thing_1982/frame_003650.jpg` | A corridor going deep into the frame, with figures at several depths. Real depth match; the setting is unrelated. |
| 19 | *High and Low* ~01:10:24 `High_and_Low_1963/frame_008447.jpg` | *The Secret in Their Eyes* (2009) ~00:17:39 | A figure in the foreground at the frame edge, with a group arranged in the room behind. Layered depth staging. |
| 36 | *Ikiru* ~01:55:22 `Ikiru_1952/frame_013844.jpg` | *12 Angry Men* (1957) ~00:02:08 | Men in suits seated in a tight cluster, staged frontally. Same blocking idea (the group as a unit). |
| 37 | *Yojimbo* ~00:25:00 `Yojimbo_1961/frame_003000.jpg` | *Harakiri* (1962) ~00:20:28 | Rows of retainers staged frontally in a shoji room. Contemporaries, not influence. |
| 50 | *High and Low* ~01:16:16 `High_and_Low_1963/frame_009152.jpg` | *Oppenheimer* (2023) ~00:13:27 | A packed room seen from the back, rows of heads leading to a board or front wall. |
| 51 | *High and Low* ~00:02:54 `High_and_Low_1963/frame_000348.jpg` | *Thank You for Smoking* (2005) ~00:06:24 | Men spread across a wide frame on low sofas around a coffee table. **Best blocking match in the pilot.** |
| — | *Yojimbo* ~00:10:20 `Yojimbo_1961/frame_001240.jpg` | *Once Upon a Time in the West* (1968) ~00:03:20 `Once_Upon_a_Time_in_the_West_1968/frame_000400.jpg` | A dark wooden interior opening onto a bright doorway at the back. Deep framing. Leone's link to *Yojimbo* is documentable; see the caveat below. |
| — | *High and Low* ~00:49:28 `High_and_Low_1963/frame_005936.jpg` | *Heat* (1995) ~01:42:25 | Men in a modern room, a large window behind, figures spread in depth. Looser match. |

## Subject or setting matches (may help with context; not composition)

*Sanjuro* ~00:56:06 → *The Sword of Doom* ~00:43:12 (tatami room, seated figures);
*Yojimbo* ~00:36:20 → *Sword of Doom* ~01:40:40 (a jidaigeki street);
*Ikiru* ~00:54:07 → *All About Lily Chou-Chou* ~02:13:42 (a dark crowd);
*Kagemusha* ~00:39:58 / *Ran* ~02:22:56 → *Return of the King* (cavalry charges);
*Ran* ~01:08:02 → *Unforgiven* ~00:09:22 (a horseman against the sky);
*High and Low* ~01:04:02 → *Look Back* ~00:35:10 (newspaper close-ups);
*Ikiru* ~01:32:07 → *Sweet Smell of Success* ~00:31:59 (men in suits in a crowded room).

## Superficial matches (typical failure)

*Sanjuro* ~00:28:15 → *Memento* ~01:38:48; *High and Low* ~00:04:00 →
*Inception* ~01:27:41; *Red Beard* ~02:27:49 → *The Prestige* ~01:04:56; most
of #20–35. In each, one face sits in a dark frame. *Sanjuro* ~00:28:15
(`frame_003390`) alone is a top match for *Memento*, *For a Few Dollars
More*, *Once Upon a Time in the West* and *Apocalypse Now*. That frame acts as
a "hub" that resembles everything. *Yojimbo* ~01:28:02 → *Cowboy Bebop*
(two textures of dark floorboards) shows the same problem.

## Data issues found

- **`l4-50k-lbfive/The_Return_2003` is a duplicate of *The Lord of the Rings:
  The Return of the King*.** The folders have the same frame count (31,594),
  and `frame_010957.webp` is byte-identical in both. It is not Zvyagintsev's
  *The Return*. Do not cite it as that film. The Curtain data was not changed.
- *The Hidden Fortress* is still excluded (documentary extra). This is the one
  Kurosawa film with the best-documented link to *Star Wars*, and it is missing.

## Implications for the final

- Treat pilot scores as a **search aid**. A score of ≥0.77 does not by itself
  mean the two shots share a composition.
- Choose pairs where you can name a shared **staging decision** (depth,
  grouping, the frame within a frame, the axis of a corridor or road), and
  confirm it holds across the shot, not just at the sampled midpoint.
- Keep documented influence separate: of the leads above, only the Leone pair
  has a well-known historical link to Kurosawa (*A Fistful of Dollars* /
  *Yojimbo*). Even there, the link is to a different Leone film than the one
  retrieved, and needs sources before it is used as a claim.

## Next experiment (chosen from the above)

The pilot showed **coverage** failing first, so the smallest useful step is to
sample densely with the **same scorer**. `focused_experiment.py` embeds every
shot midpoint of the eleven Kurosawa films and of eight focus films:

- possible documented links: *For a Few Dollars More*, *Once Upon a Time in the
  West*, *Star Wars*, *Unforgiven*, *Kill Bill Vol. 1*, *Apocalypse Now*
- jidaigeki contemporaries as controls: *Harakiri*, *The Sword of Doom*

It writes 12 diverse pairs per focus film and a contact sheet to
`results/focused-dense-v1/`. The pilot folder is untouched. If dense results
are still dominated by dark close-ups, the next step is a scorer change
(grayscale input, spatial-only, or a hub penalty), compared on the same
reviewed pairs.

## Update — direction change: let a trained model find the similarities

The user's goal is a model that finds the comparisons itself; historical
sources become optional support for the strongest examples.

### Dense baseline (`results/focused-dense-v1/`, pilot scorer, every shot)

Dense sampling alone fixes the repeated-frame problem and surfaces real
staging. Examples: *High and Low* police briefing ~01:38:55 / ~01:51:24 →
*Star Wars* rebel briefing ~01:40:05 / ~01:39:40; *Sanjuro* ~01:33:11 →
*Once Upon a Time in the West* ~00:50:24 (a group standing on open ground);
*Yojimbo* ~01:15:48 → *West* ~01:13:06 (two at a table, deep cluttered
interior). Still ~5 of 12 per film are bearded/face close-ups.

### Trained composition head (`composition_model.py`, `results/composition-v1/`)

A small transformer head (~0.6M params) trained on CPU in 3.5 min on frozen
DINOv2 4×4 patch grids. It is self-supervised: each frame vs. two
appearance-scrambled views (grayscale or recoloured, exposure/gamma, blur,
noise, ≤12% reframe), InfoNCE, with ~10% of films held out. A hub penalty is
applied at search. No human labels were used.

- Training was **too easy** (99% held-out top-1 at epoch 0). It still changed
  the ranking a lot, but a harder task should teach more.
- **Better:** no face-close-up dominance. Matches are now about layout and
  geometry: *Kagemusha* ~01:39:46 → *Star Wars* ~01:40:44 (assembly from
  behind); *Ran* ~02:09:18 → *West* ~01:11:02 (riders in dust);
  *Kagemusha* ~02:50:18 → *West* ~00:22:30 (horizon silhouettes);
  *Yojimbo* ~00:48:24 → *West* ~02:13:58 (a figure framed in a doorway);
  *High and Low* ~00:55:15 train → *Star Wars* ~01:56:36 X-wing (horizontal
  motion streak).
- **New failure:** abstract or low-structure frames (fog, dappled light,
  dream colours, explosions) and bare "rectangle in the centre" matches. About
  4 of 12 per film.

### Next (v2 of the model)

1. Harder positives: other frames from the **same shot** (camera and actors
   move, the staging persists), in addition to scrambles.
2. Drop low-structure frames (edge-density filter) from search.
3. Human-in-the-loop: gallery marks (useful/superficial) → fine-tune the head
   with a ranking loss, and report before/after on held-out marks.

## Sequence baseline — 2026-11-04

The user clarified that the target is shared directing technique: camera
angles, staging, and how framing develops over time, even across different
subjects. Added and ran `sequence_experiment.py` on 1,520 four-frame clips
(80 shots per film; 11 Kurosawa films and eight comparison films). Each clip
spans 1.5 seconds at the existing 2fps and stays within detected shot boundaries.

The baseline uses the trained v1 midpoint composition vector (65%) and ordered
coarse optical flow (35%). Motion is measured in a 4×4 spatial grid for each
of three chronological intervals. These are heuristic weights, not learned
temporal features. Motion below 0.002 image fractions/second contributes no
motion similarity. 1,363 clips exceed that threshold.

Outputs: `results/sequence-v1/index.html` and 48 animated pair GIFs; underlying
clips, scores and comparison statistics are saved alongside. Relative to
composition alone on the same sampled clips, 45/48 selected pairs change.
This is an ablation of ranking, not proof that retrieval improves. A static
preview was inspected for rendering; motion matches have not been human-rated.

Synthetic tests verify directional sensitivity (reversing a translated
sequence yields cosine below -0.99), static-frame rejection, boundary safety,
and rejection of nonconsecutive frame clocks. Both unittest checks pass.

This does not yet learn camera angles, distinguish camera from actor movement,
or train a temporal network. Fast action may alias at 2fps. Next is qualitative
review of the animated comparisons, followed by a temporal training experiment
with this baseline retained for comparison. Older v2 work remains unfinished.

## Latest update — 2026-11-04: trained temporal model completed

User authorized teaching the model sequence/directing similarity. Added
`temporal_model.py` and `test_temporal_model.py`. Run
`../.venv/bin/python temporal_model.py all`; cached extraction can resume.
Individual steps: `extract`, `train`, `search`.

- Extracted frozen composition features for all 6,080 frames in the existing
  1,520 clips. Originals and earlier results were preserved.
- Visual-only temporal training learned reversal sensitivity but failed flow
  prediction (near/worse than zero prediction). Logs saved separately in
  `results/temporal-v1/first-attempt-train-log.json` and
  `visual-only-train-log.json`; visual-only checkpoint also archived.
- Final model is a **trained joint visual-plus-flow temporal head**, 336,704
  parameters. Inputs: four frame embeddings, ordered feature differences, and
  three intervals of coarse optical flow. Transformer learns ordered clip
  embeddings with contrastive, motion-denoising, visual-change and reversal
  losses. Retrieval uses learned embeddings, not fixed composition/flow scores.
  It still requires measured flow at inference; do not claim visual-only motion
  prediction or learned angle labels.
- Trained 80 epochs on 1,280 clips. Validation: 240 clips from Star Wars,
  Once Upon a Time in the West, Harakiri. These films select checkpoints, so
  are not independent tests. Holdout only applies to temporal-head training;
  the older frozen composition head previously saw catalogue films.
- Validation clean motion reconstruction MSE 0.677; perturbed-input MSE 0.693
  vs zero prediction 1.153 (about 40% lower). This tests motion encoding and
  denoising, since flow is an input. Initial feature-noise clip retrieval was
  already 100%, so that metric is uninformative. Reversal sensitivity improves.
- `results/temporal-v1/index.html`: 48 animated trained matches, each with a
  collapsible baseline comparison. `head.pt`, `features.npz`, `validation.json`,
  `train-log.json`, `report.json`, `README.md`, `preview.jpg` alongside.
- Only 2/48 selected pairs overlap the baseline: ranking change, not quality
  improvement. Static preview shows some wide-shot/close-up mismatches.
  No human motion/technique ratings yet. Do not promote over baseline yet.
- Five tests pass: flow direction/static rejection, safe clip boundaries,
  model order/flow sensitivity, finite gradient update, film partition.
  `../.venv/bin/python -m unittest test_temporal_model test_sequence_experiment -v`
- **Still not achieved:** validated camera-angle/auteur understanding. Next
  evaluate shared filmmaking techniques across different scenes; improve
  composition retention and training signal, using both baseline and trained
  gallery. User labels remain optional, not a prerequisite for autonomous work.

