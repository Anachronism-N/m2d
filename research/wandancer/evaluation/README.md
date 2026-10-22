# Music–motion diagnostics (uncalibrated, CPU-only)

This is an **interpretable diagnostic tool, not an RL trainer, validated reward,
FACT reproduction, or automatic success judge**. Only Python stdlib and NumPy are
required. It does not load Torch, run NLF, use a GPU, modify media, or infer beat
labels. Synthetic tests establish equations and guards, **not real-video reward
reliability**. Real pose/beat calibration has been attempted and failed provisional gates; the current heuristic must not be optimized as an RL reward. Independent generated RGB quality review remains pending. See [EXPERIMENT_PROGRESS.md](EXPERIMENT_PROGRESS.md).

## Definitions and raw components

All timestamps and sigma use **seconds on the same absolute clip timeline**.
Explicit `window_seconds: [start, end]` clips both event sets (inclusive endpoints);
excluded counts are reported. Use actual decoded timestamps, trim and audio
warp/offset from the run manifest, not Global container timing or assumed zero offset.

Let M be motion events, A audio beats, and sigma = 0.05 seconds by default:

- **motion_to_music** = mean over m in M of exp(-min over a in A |m-a|² / (2 sigma²)).
- **music_to_motion** = mean over a in A of exp(-min over m in M |a-m|² / (2 sigma²)).

Each direction saves every nearest distance (seconds), Gaussian similarity, source
count, mean, and coverage within 2 sigma. There is **no weighted total reward**.
An empty source or target yields mean zero; absent neighbors use JSON null
distance and zero similarity. Empty motion/music is invalid. Matching is
many-to-one, not unique beat assignment.

Validity gates remain separate from raw components:

| Gate | Default / behavior |
|---|---|
| Motion confidence | valid_fraction >= 0.8, explicitly supplied/derived |
| Activity | positive min_activity and activity_units required; score >= threshold |
| Music coverage | >= 0.5 within 2 sigma |
| Motion coverage | >= 0.5 within 2 sigma |
| Motion event density | <= 4 events/second |
| Event spacing | >= 0.12 seconds |
| Pose direction reversals | fraction <= 0.45, fixed heuristic |

Thresholds are provisional protocol settings, not validated thresholds. Missing
quality is invalid. **Valid is not success**: a synchronized periodic template can
score one. Low motion-interval CV flags a regular-template candidate, not proof
of poor dance. Static and severe jitter are tested. No ground-truth choreography
or uniquely correct motion is assumed.

Phase and tempo are distinct descriptions:

- Median audio interval / median motion interval approximates tempo ratio; with
  >=3 events on both sides, ratios near 0.5/1/2 label half/same/double candidates.
  Half/double ambiguity is not automatically a failure. This is not dynamic tempo tracking.
- Best shift is applied to **music** (positive = later), searching +-0.25 seconds
  in 0.01-second steps with the same events at every shift. Its optimized score
  never replaces the original score. Sparse/edge events and changing tempo make
  this descriptive, not a validated phase estimator or evidence of audio use.

## Optional pose event heuristic

NPZ must contain `joints` (T,J,3), `valid` **boolean** (T,) or (T,J), and
`timestamps_seconds` finite strictly increasing (T,), T>=3. Convert extractor
confidence upstream using an explicitly recorded threshold; never silently
cast floating confidence to boolean. Object arrays are forbidden (allow_pickle=False).

No NLF-specific adapter is claimed. Verify actual units, joint order, camera/root
behavior and timestamps before exporting a real NPZ. Freeze `pose_joint_indices`
if needed; the default all-joints set may include unstable fitted joints.

The fixed heuristic takes absolute selected-joint velocity, median joint speed,
a centered 0.10-second box smooth, then local speed minima with prominence
>=0.15 times max(p90-p10, min_activity), and a 0.20-second greedy prominence-based
separation. No interpolation crosses invalid frames. Activity is median smoothed
speed; its required threshold is **coordinate units per second**. It saves
parameters, joints, speed traces, prominence, validity/reversal fractions and
method name. This is a pause/change heuristic, **not the full standard FACT method**.

Adjacent unsmoothed velocity direction reversals catch severe frame-alternating
jitter, not all high-frequency noise. Camera motion, fit smoothing, irregular
sampling, joint choice, occlusion and tracking failures affect the score. There
is no multi-person association. NLF/SMPL fitting success is not RGB correctness.
Calibrate actual video NLF before ranking or optimizing with these thresholds.

## External private manifest and CLI

From the publication checkout, set both variables to external files. Output
parent must already exist:

```sh
python research/wandancer/evaluation/run_diagnostics.py \
    --manifest "$PRIVATE_MANIFEST" --output "$NEW_PRIVATE_REPORT"
```

Output is exclusively created with mode 0600; existing files (including originals)
are never overwritten. Output inside the publication checkout is rejected.
Per-sample metadata/traces are **private**, not safe public aggregates. Inputs
are read-only, paths not hardcoded. Exit 0 means processing completed, **not**
every record valid. Extraction/input failures remain in the denominator; structural
manifest/output errors exit 2. Empty cases do not invent pending generation results.

Minimal **synthetic-only** example (activity is not calibrated for real coordinates):

```json
{
  "schema_version": 1,
  "alignment_parameters": {"sigma_seconds": 0.05},
  "beat_shifts_seconds": [0.2, -0.2],
  "cases": [{
    "sample_id": "synthetic-target", "case_id": "synthetic-case", "seed": 42,
    "conditioning_group_id": "synthetic-frozen-contract",
    "window_seconds": [0, 3],
    "audio_beats_seconds": [0.5, 1.0, 1.5, 2.0, 2.5],
    "motion_events_seconds": [0.5, 1.0, 1.5, 2.0, 2.5],
    "motion_quality": {
      "valid_fraction": 1.0, "activity_score": 1.0, "min_activity": 0.1,
      "activity_units": "synthetic_units/second"
    }
  }],
  "paired_null_comparisons": []
}
```

For pose, replace motion_events_seconds/motion_quality with `pose_npz` (absolute
or relative to manifest) and explicit `pose_parameters` including `min_activity`
and `coordinate_units`; optional `pose_joint_indices`. Pose parameters may be
top-level or per-case. Audio beats remain external; no fabricated beat annotations.

## Counterfactuals and paired-seed null aggregation

Same-dance beat shifts hold motion fixed, shift beats **without circular wrap**,
and compare baseline/shifted scores on shared temporal support. Both full reports
and signed baseline-minus-shift raw components are saved, including invalid cases.
This tests alignment sensitivity only. To test generation audio use, actually
rerun generation under interventions and compare real paired conditions/seeds.

Optional manifest `paired_null_comparisons` entries:

```json
{"pair_id":"pair-1", "target_sample_id":"target-1", "null_sample_id":"null-1"}
```

Members must exist, be distinct, match non-null case_id/seed/conditioning_group_id
and evaluation parameters/windows, and have valid diagnostics. Rejections/counts
remain visible. The external conditioning_group_id must attest frozen reference,
prompt, Global/mask, fps, offsets and all non-intervened settings. Equal strings
are **not proof**; preserve actual run identity, intervention provenance and full
private configuration. This code cannot verify generation provenance.

Target-minus-null deltas are reported in both directions separately. Repeated
pairs within a case/seed average first, then seeds within case, then equal case
weighting. Frames/seeds/candidate pairs are **not independent cases**. No p-values,
ranking claims or success labels. Fewer than 5 case clusters gives **no interval**.
Otherwise a seeded case-cluster percentile bootstrap (1000 resamples, seed 0)
is exploratory, assumes independent cases and is not confirmatory. Correlated
songs/identities need independent grouping upstream; five clusters is not itself
adequate evidence. The fixed 3-case diagnostic set cannot support population/SOTA claims.

## Independent calibration/review gate

1. Freeze beat/pose extraction and units/thresholds on actual NLF video outputs
   with manual timing checks; retain failed/low-confidence fits.
2. Include real static, jitter, regular-template, shifted audio, occlusion/tracking
   and malformed-body negatives; compare paired seeds/nulls, not selected best samples.
3. Independently blind humans to scores/conditions for RGB anatomy/multi-limb,
   identity, naturalness and observable action/beat judgments; report disagreement,
   failures, diversity and audio counterfactual responses.
4. **VideoReward has no audio input**; it cannot replace synchronization scoring.
   Automatic scoring or NLF fit cannot substitute for human RGB review. No hand/finger guarantee.

## Standalone synthetic tests

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
    -s research/wandancer/tests -p test_diagnostics.py -v
```

Tests cover Gaussian seconds, both directions, phase/half/double, empty/sparse/
low-confidence/static/high-density/severe-jitter negatives, pose invalid gaps,
same-dance shifts, pair validity, repeated-seed/case balancing, small-sample
uncertainty and private/no-overwrite CLI/NPZ behavior. They do **not** test real
Wan-Dancer quality or a real NLF/video/reward pipeline.

## NLF observation adapter

`extract_pose.py` uses explicit private video manifests and a new private output namespace. The current NLF TorchScript detector contains hard-coded `cuda:0` operations: all three CPU attempts failed and were retained. Use `--device cuda` on an authorized available GPU; `map_location=cpu` does not establish CPU portability. No weights or H3 scripts are changed. Actual decoder timestamps, detection validity and largest-box selection are recorded; this is not identity tracking. GPU extraction completed for the original 16 generated videos with failures and largest-box selection reported separately; fitted-body observations do not establish RGB quality. Real reward calibration failed provisional gates. Additional intervention/checkpoint observations are evaluated separately; see the verified progress report.
