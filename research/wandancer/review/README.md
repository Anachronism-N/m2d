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

Global author baseline: 38 frames, 8fps container, no audio. This is an **anchor container, not continuous video**; container time is **not music/plan time**. Local requires 30fps and audio; crop and trim contracts must be recorded explicitly. Default author tail trim is 0.2 seconds, but final encoded time is measured rather than guessed. Do not apply old 81-frame review assertions to generated Wan-Dancer videos.

Only manifest-listed media inside allowed canonical roots may be read; traversal and escaping symlinks are rejected. GET/HEAD support single byte ranges. Writes require a per-session token, loopback Host/Origin, bounded JSON and known complete media target; validation status accompanies every rating. A passed media contract does not establish natural dance, identity, anatomical correctness or music responsiveness.

The new adapter follows the project's existing master-audio/synchronized-video workflow. It does not import or alter the old review or H3, and requires no third-party frontend assets. For an authenticated internal reverse proxy that preserves its external Host/Origin, add `--trusted-origin "$EXACT_HTTPS_PROXY_ORIGIN"`. Only that exact Host and Origin are accepted; no wildcard is enabled, and the loopback bind, per-session write token and media allowlist remain enforced.

CPU fixtures test path/range/ratings guards; they are distinct from service, media and human-review acceptance. The existing private service was reloaded and checked twice after adding three heldout Base/SFT/NFT cases and seven separate intervention-audio cases; manifest HTTP200 and all16 added Local Range206 reads also passed: 14 cases / 32 runs, with all 16 new Local media contracts and seven reused Global anchor contracts passing. The nine heldout Global stages remain blocked because this is a Local-only oracle-GT-anchor pilot, not generated Global output. Heldout Local uses the row-specific 4.966667-second audio with zero head/tail trim and a 149/30-second duration contract; intervention Local uses a 0.2-second tail trim.

This is **open review, not blind review**: the interface exposes case/run/model/condition labels and GT. Actual input audio is bound independently for each intervention; private prompt/provenance is not exported by the API. Both reload checks observed **zero ratings**, with the private ratings file unchanged and absent. No independent human assessment or quality approval is claimed. Browser playback, human scores and dance/music-responsiveness acceptance remain unverified; passing media contracts and decode checks do not establish them.

A separate private blind-review namespace is now prepared with10 neutral cases and16 variants. Model/condition names, GT and Global containers are hidden; the randomized alias mapping stays private until ratings are frozen. Actual manifest HTTP200,16 Local contracts and16 Range206 reads passed. Ratings are still0: this establishes a blinded view, not independent reviewer identity or completed human assessment. No media or alias mapping is published.
