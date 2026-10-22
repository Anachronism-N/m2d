# Wan-Dancer controlled experiments

Research status: **23 baseline/condition media outputs passed full decode and stage-specific checks. A 24-step Local LoRA SFT pilot and a 4-update online NFT oracle pilot completed, including checkpoint checks. Shared heldout raw flow-MSE favors SFT, not NFT; this is not evidence of visual quality or music synchronization improvement. Seven extra interventions and nine matched checkpoint videos also passed full decode (39 media outputs total); independent human quality review remains pending.** See [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md) for the gated protocol. No GPU-quality claim follows from the unit tests.

## Components

- `patches/`: fixed-commit minimal engineering patch (explicit local model paths, rank-0 Local writes, CLI CFG forwarding). Native music processing, sampling, frame layout, crops and trim remain unchanged.
- `tools/apply_official_patch.py`: verify original SHA256 and unchanged pipeline before applying; refuses unknown contents and a second application.
- `tools/run_native.py`: one-node/eight-rank stage launcher, default dry-run; external manifest and private output directory. Generated media remain unassessed until separate validation.
- `evaluation/diagnostics.py` and `evaluation/run_diagnostics.py`: uncalibrated seconds-based event/pose diagnostics, invalid-sample accounting and paired seed-null aggregation. See [evaluation/README.md](evaluation/README.md). `beat_diagnostics.py` is a small equation-level fixture diagnostic, not the full protocol.
- `training/`: bounded Local LoRA training, online NFT **GT-latent oracle engineering** pilot, shared nonzero-sigma flow-MSE comparison and matched heldout video generation. The pilots use GT anchors and unequal update budgets; they are not a deployable music-reward experiment.
- `evaluation/verified_results.json`: inspected anonymous aggregate measurements; detailed receipts, videos and pose traces remain private.
- `review/`: loopback review service with explicit media allowlist, Range streaming, private append-only ratings and stage-aware ffprobe checks. Metadata validation is not human quality approval.

Run CPU tests with Python 3.10+, NumPy and Torch available; the frontend check also needs Node:

```sh
python -m unittest discover -s research/wandancer/tests -v
WANDANCER_OFFICIAL_SOURCE="$PINNED_OFFICIAL_CHECKOUT" \
    python research/wandancer/tools/test_official_patch.py -v
python -m unittest discover -s research/wandancer/training -p 'test_*.py' -v
node --check research/wandancer/review/app.js
```

Pinned official checkout must match the commit and original hashes in `patches/metadata.json`. Official inference and upstream training use separate environments because both install a package named `diffsynth` with different interfaces.

## Private inputs and outputs

No data, videos, reference images, identity labels, weight files, credentials or internal service addresses are committed here. Manifests, cache and run receipts are external. Run/stage identifiers are unique and prior outputs are not overwritten. All detailed pose traces and per-case records remain private. Only inspected aggregate findings are eligible for publication.

Public method references: [Wan-Dancer](https://github.com/Wan-Video/Wan-Dancer), [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), [Flow-GRPO](https://github.com/yifan123/flow_grpo), [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT), [DiffusionART](https://github.com/vanzll/DiffusionART), [VideoAlign](https://github.com/KwaiVGI/VideoAlign). Public Wan examples do not establish music-dance alignment or direct Wan-Dancer compatibility.
