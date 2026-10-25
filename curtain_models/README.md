# Models

The trained models Curtain uses, kept here rather than in `.local/` (frames,
indexes and old experiments). They are not in git (`*.pt` and `*.onnx` are
ignored): every retrain would add tens of MB to the history for good.

- `all-films-v1/`: the live model (`best.pt`, its config, split and metrics);
  colour-ft-v1 fine-tuned on Modal (curtain_ml/train_modal.py) on 407 films
  with neighbour negatives and hard augmentations. The site loads it via
  `CURTAIN_CHECKPOINT`; `all-films-v1-onnx/` built `.local/faiss-all-films-v1`.
- `tight-v1/`: all-films-v1 fine-tuned on all 468 training films (1,500
  frames each) with smaller, thinner tile crops (`--tight`); in-film rank
  345 -> 369 of 470. `tight-v1-onnx/` builds `.local/faiss-tight-v1`.
- `colour-ft-v1/`: the model before, combined-rocm-50k-v1 fine-tuned with
  colour-grade augmentation on the RTX 4060. Its index was deleted on
  2026-10-25; `colour-ft-v1-onnx/` can re-index it.

The old `.local/trained_models/colour-ft-v1` and
`.local/rcmindex-model/colour-ft-v1` paths are symlinks to these.
