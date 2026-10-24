# Models

The trained models Curtain uses, kept here rather than in `.local/` (frames,
indexes and old experiments). They are not in git (`*.pt` and `*.onnx` are
ignored): every retrain would add tens of MB to the history for good.

- `colour-ft-v1/`: the live model (`best.pt`, its config, split and logs);
  combined-rocm-50k-v1 fine-tuned with colour-grade augmentation on the
  RTX 4060. The site loads it via `CURTAIN_CHECKPOINT`.
- `colour-ft-v1-onnx/`: the same model exported for rcmindex (`trunk.onnx`,
  `head.*`), used to build `.local/faiss-colour-ft-v1`.

The old `.local/trained_models/colour-ft-v1` and
`.local/rcmindex-model/colour-ft-v1` paths are symlinks to these.
