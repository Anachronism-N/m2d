# Verified experiment progress

This is an interim engineering report, not a final quality or RL result.

## Verified

- Official weights: pinned HF revision `85ce88dd8d025459dcf0fe93982d6da8b9002957`; 23 files, 85,667,867,184 bytes verified against official hashes. Global/Local safetensors each contain 1,424 tensor keys; header-only inspection did not load tensor contents.
- Both experiment nodes: Torch 2.6.0+cu124 and locally compiled FlashAttention 2.6.3; BF16 attention forward/backward and official pipeline import passed. Model loading/inference remain separate acceptance checks.
- Official KPop controlled Global: 48 steps, CFG5, shift5, seed0, portrait canvas720×1280. Process returned0; generated container decoded as38 frames at8fps,720×1280,4.75s,noaudio. This container represents sparse planning for the19.295782s music, not4.75s dense dance.
- Exact native audio feature function produced579×35 features for the official music (dynamic input_fps7.5); each of six5.0625s diagnostic tracks produced152×35 (input_fps30). Features were finite. Sampling-rate logic was not silently changed.
- Independent CPU verification:18 official-patch/launcher tests,43 diagnostic/review tests,JS syntax check. Synthetic tests are not real reward validation.
- Private review HTTP manifest200 and media Range206 tested with actual reference/GT media. GT independently probed as81frames,16fps,5.0625s. No raw media is published.

## Running/pending

- Native Local24-step generation from the actual Global plan is RUNNING; final output, temporal stitching, body/identity quality and music responsiveness are not approved.
- Six paired short-case Global runs are queued/running with native48steps and matched conditions; subsequent Local24steps use each matching plan.
- Real NLF-body observations and reward calibration are in progress. Fitted skeleton validity cannot establish RGB anatomical correctness.
- SFT/RL comparisons have not run. No music-control improvement, reward gain, final quality ranking or AR speed claim is established.

## Repairs and limitations

- Minimal hash-checked official patch: explicit local models/tokenizer; rank0-only Local media writes; Local CLI CFG forwarding. Native architecture, audio feature logic, time mapping, output crop and trim preserved.
- Flash build initially lacked CUDA library include paths; exact installed CUDA runtime headers repaired the build. A wheel-wait detector quoting issue was repaired; the built wheel subsequently passed independent hash and kernel checks. Failed attempts are retained privately.
- Initial target repository did not grant write access. Publication moved to the user-specified `Anachronism-N/m2d`; first commit `fb88c32` was pushed and matched the remote.
- Native Local continuation duplicate attempt was rejected by existing-run protection; the already-running correct Local was preserved rather than overwritten/restarted.
- Only exact inspected resource-holder processes were released. Unrelated work and dirty source trees were not modified.

Complete conclusions require all scheduled runs, media checks, independent quality/music/command assessments and gated training comparisons.
