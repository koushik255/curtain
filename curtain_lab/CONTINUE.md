# CONTINUE — Kurosawa composition model (paused 2026-11-03)

Read this first. More detail is in `REVIEW.md`. The older pilot handoff is
`HANDOFF.md`, which is partly superseded: the goal changed, see below.

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

## Previous update — 2026-11-04: sequence experiment completed

The user clarified the goal: similar **directing choices** (camera angle,
framing, blocking, camera/actor movement), across different subject matter,
rather than simply similar-looking images. They authorized treating consecutive
0.5-second screenshots as clips. This is now the active direction; the older
v2 same-shot-positive plan below remains unfinished, not the latest result.

- Added `sequence_experiment.py`. Run with
  `../.venv/bin/python sequence_experiment.py` (resumable motion cache).
- Ran 1,520 clips: 80 sampled shots each from 11 Kurosawa + 8 focus films.
  Each clip is four consecutive screenshots, spanning 1.5 seconds, wholly
  within a detected shot. Verified filename clock increments are consecutive.
- Retrieval combines the existing trained v1 midpoint composition embedding
  (65%) with ordered 4×4-grid optical-flow descriptors over three intervals
  (35%). These weights and the motion threshold are heuristics. This is a
  **sequence retrieval baseline, not a newly trained temporal model**.
- Outputs: `results/sequence-v1/index.html`, 48 side-by-side animated GIFs,
  `clips.json`, `search.json`, `report.json`, `preview.jpg`.
- 1,363 clips exceeded the heuristic motion threshold. Compared with composition
  alone on this same sampled subset (no hub penalty in either branch), only
  3/48 selected pairs remained identical. This measures ranking change,
  **not quality improvement**. No human motion-quality review yet.
- Two tests pass: synthetic translation reverses motion direction; static
  frames give near-zero motion; clips respect shot boundaries and reject gaps.
  `../.venv/bin/python -m unittest test_sequence_experiment -v`
- Limitations: optical flow at 2fps is approximate and can fail on fast movement;
  it does not separate actor motion from camera motion. Detected shot boundaries
  can be wrong. The composition head still only sees the midpoint. Camera angles
  and auteur style are not explicitly learned or validated.
- Next: review GIFs for similar filmmaking technique, compare with composition
  alone, then use these findings to design/train an ordered temporal head.
  Keep the baseline so improvement can be measured. Do not infer authorship or
  influence from retrieval similarity.

## Goal (changed this session)

The user wants **a small trained model that finds composition/staging
similarities by itself**. It should not depend on us hand-curating pairs or
hunting historical sources; history is optional support for the best examples.
Train on CPU (there is no GPU on this machine; the torch build is CUDA, but no
driver is present). No paid compute. Don't acquire films or touch the live site.

## Done

| Step | Output | Result |
|---|---|---|
| Pilot review + real gallery browser check | `REVIEW.md`, `results/pilot-v1/gallery-preview.png` | Check passed. Pilot matches are mostly dark face close-ups; 16 frames/film was too sparse. |
| Dense baseline, pilot scorer | `focused_experiment.py` → `results/focused-dense-v1/` | Every shot of 11 Kurosawa films vs 8 focus films. Finds real staging (*High and Low* briefing → *Star Wars* briefing), still ~5/12 face close-ups. |
| **Model v1**, self-supervised head | `composition_model.py` (`extract`, `train`, `search`) → `results/composition-v1/` | `head.pt`, `sheet-*.jpg`, `search.json`, `train-log.json`. Face bias gone; matches are layout/geometry. New failure: abstract/low-structure frames (fog, dappled light, explosions), ~4/12. Training too easy (99% top-1 at epoch 0). |

Model v1 details: frozen DINOv2 ViT-S/14 → 4×4 grid of patch features →
small transformer head → 128-d vector. InfoNCE between a frame and two
appearance-scrambled views (grayscale/recolour, gamma, blur, noise, ≤12%
reframe). 30 epochs, ~3.5 min CPU. At search time a hub penalty is subtracted.

## In progress when paused: model v2 (user approved items 1 + 2)

1. **Harder positives from the same shot.** `composition_model.py shots`
   extracts two extra frames per training shot (near start and end; view 3 is
   scrambled, view 4 is original colours) into `results/composition-v1/train-shot-grid.npy`.
   **Paused at 21/473 films.** Rerun the same command and it resumes from the
   per-film cache in `results/composition-cache/*.npy` (~15 min total on CPU):

   ```sh
   cd /home/koushik/curtain/curtain_lab
   ../.venv/bin/python composition_model.py shots
   ```

2. **Junk-frame filter for search.** `structure_check.py` (written, **not yet
   run**) compares two measures on frames already judged junk or good from the
   v1 sheets: (A) 4×4 cell uniformity of DINOv2 features, and (B) coarse-edge
   energy after blurring away fine texture. Run it, choose whichever separates
   the two groups (or both), and apply it as a filter in `search()`.

   ```sh
   ../.venv/bin/python structure_check.py
   ```

## Still to write for v2

- `train()`: load `train-shot-grid.npy` and concatenate it to the training
  views (5 views). Sample positive pairs so that ~70% involve a same-shot view
  (3 or 4) against a mid-frame view (0–2). Log validation top-1 for the hard
  pair (view 0 vs view 4) separately, to show the task is no longer trivial.
  Save as `head-v2.pt`; don't overwrite v1.
- `search()`: add the structure filter and write output to a separate folder
  (e.g. `results/composition-v2/`) so v1 and v2 sheets can be compared.
- Compare v1 against v2 on the same eight focus films (Star Wars, Once Upon a
  Time in the West, etc.), and add the findings to `REVIEW.md`.

## Later (item 3, needs the user)

Fine-tune on the user's gallery marks (Useful/Superficial → exported
`kurosawa-review.json`): ~30–50 marks, ranking loss, before/after on held-out marks.

## Known data issues

- `The_Hidden_Fortress_1958` is a documentary extra; it is excluded.
- `l4-50k-lbfive/The_Return_2003` is a byte-identical copy of *LOTR: Return of
  the King* (31,594 frames each). It is not Zvyagintsev's film. Data was not modified.

## Don't

Rebuild finished caches, modify `results/pilot-v1/`, touch the parent repo's
unrelated changes, or commit without asking.
