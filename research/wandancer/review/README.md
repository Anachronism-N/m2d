# Private review adapter

Loopback-only standard-library service. Supply external private case/run manifests, explicit allowed media roots and an external ratings file. No media scanning, upload or import-time probing is performed. Use a separately installed `ffprobe`; missing/failed probe is **blocked**, never passed.

```sh
python research/wandancer/review/server.py \
  --manifest "$PRIVATE_CASE_MANIFEST" \
  --allowed-root "$PRIVATE_MEDIA_ROOT" \
  --reviews "$PRIVATE_RATINGS_JSONL" \
  --ffprobe "$FFPROBE_EXECUTABLE" --port 8767
```

Case manifest is `schema_version:1` and `cases:[...]`. Each case has `case_id`, optional absolute `reference`, `gt`, `original_audio`, `swap_audio`, and `runs`. Each run has `run_id`, `model`, `version`, `condition`, `seed`, `status`, and Global/Local stage objects. Stages use `status`, optional `path`, `working_canvas`, `crop`, `trim` and `expected`. Outputs remain private and ratings use reviewer aliases rather than personal identities.

Global author baseline: 38 frames, 8fps container, no audio. Container time is **not music/plan time**. Local requires 30fps and audio; crop and trim contracts must be recorded explicitly. Default author tail trim is 0.2 seconds, but final encoded time is measured rather than guessed. Do not apply old 81-frame review assertions to generated Wan-Dancer videos.

Only manifest-listed media inside allowed canonical roots may be read; traversal and escaping symlinks are rejected. GET/HEAD support single byte ranges. Writes require a per-session token, loopback Host/Origin, bounded JSON and known complete media target; validation status accompanies every rating. A passed media contract does not establish natural dance, identity, anatomical correctness or music responsiveness.

The new adapter follows the project's existing master-audio/synchronized-video workflow. It does not import or alter the old review or H3, and requires no third-party frontend assets. Reverse-proxy access must preserve the loopback upstream Host; no unvalidated external Host wildcard is enabled.

CPU fixtures test path/range/ratings guards. Actual service startup and HTTP checks are separate acceptance events; browser playback, human scores and generated-media acceptance remain pending until observed.
