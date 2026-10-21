"""CPU-only evaluation of an external, private JSON manifest."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import numpy as np

if __package__:
    from .diagnostics import (AlignmentParameters, PoseParameters, aggregate_paired_null,
                              compare_beat_shifts, diagnose_events, extract_pose_events)
else:
    from diagnostics import (AlignmentParameters, PoseParameters, aggregate_paired_null,
                             compare_beat_shifts, diagnose_events, extract_pose_events)


def evaluate_manifest(manifest, manifest_directory):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("cases"), list):
        raise ValueError("manifest requires schema_version: 1 and cases list")
    parameters = AlignmentParameters(**manifest.get("alignment_parameters", {}))
    pose_config = manifest.get("pose_parameters")
    records = []
    seen = set()
    for case in manifest["cases"]:
        sample_id = case.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in seen:
            raise ValueError("each case requires a unique nonempty sample_id string")
        seen.add(sample_id)
        record = {key: case.get(key) for key in ("sample_id", "case_id", "seed", "conditioning_group_id")}
        try:
            if ("pose_npz" in case) == ("motion_events_seconds" in case):
                raise ValueError("supply exactly one of pose_npz or motion_events_seconds")
            if "pose_npz" in case:
                config = case.get("pose_parameters", pose_config)
                if config is None:
                    raise ValueError("pose input requires explicit pose_parameters and activity units/threshold")
                source = Path(case["pose_npz"])
                if not source.is_absolute():
                    source = Path(manifest_directory) / source
                with np.load(source, allow_pickle=False) as arrays:
                    extraction = extract_pose_events(arrays["joints"], arrays["valid"],
                                                     arrays["timestamps_seconds"], PoseParameters(**config),
                                                     case.get("pose_joint_indices"))
                events = extraction["motion_events_seconds"]
                quality = extraction["motion_quality"]
                record["pose_extraction"] = extraction["extraction"]
            else:
                events = case["motion_events_seconds"]
                quality = case.get("motion_quality")
            record["diagnostics"] = diagnose_events(case["audio_beats_seconds"], events,
                                                     case["window_seconds"], quality, parameters)
            record["beat_shift_comparisons"] = compare_beat_shifts(
                case["audio_beats_seconds"], events, case["window_seconds"],
                case.get("beat_shifts_seconds", manifest.get("beat_shifts_seconds", [])),
                quality, parameters)
        except (ValueError, TypeError, KeyError, OSError) as error:
            # A failed pose extraction remains a denominator entry, never a dropped sample.
            record["diagnostics"] = {"valid": False, "invalid_reasons": ["input_or_extraction_failure"]}
            record["failure"] = {"type": type(error).__name__, "message": str(error)}
        records.append(record)
    aggregate = aggregate_paired_null(records, manifest.get("paired_null_comparisons", []),
                                      **manifest.get("bootstrap_parameters", {}))
    return {"schema_version": 1, "status": "uncalibrated_diagnostics_only",
            "alignment_parameters": asdict(parameters), "records": records,
            "n_records": len(records),
            "n_valid_diagnostics": sum(r["diagnostics"]["valid"] for r in records),
            "paired_null_aggregate": aggregate,
            "limitations": ["synthetic_tests_do_not_validate_real_video_reward",
                            "pose_heuristic_not_calibrated_on_real_NLF",
                            "VideoReward_has_no_audio_input_and_cannot_score_music_sync",
                            "RGB_multi_limb_identity_and_naturalness_require_independent_human_review",
                            "beat_shift_tests_alignment_sensitivity_not_model_music_use",
                            "no_weighted_total_reward_or_success_label"]}


def write_private_report(report, output_path, publication_root=None):
    path = Path(output_path).resolve()
    root = Path(publication_root).resolve() if publication_root is not None else Path(__file__).resolve().parents[3]
    if path == root or root in path.parents:
        raise ValueError("evaluation output must be outside the publication checkout")
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    # Existing files, including original manifests/media, are never overwritten.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="external JSON manifest; never edited")
    parser.add_argument("--output", type=Path, required=True, help="new private JSON path outside checkout")
    args = parser.parse_args(argv)
    try:
        with args.manifest.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        report = evaluate_manifest(manifest, args.manifest.resolve().parent)
        write_private_report(report, args.output)
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, f"diagnostics failed: {error}\n")
    print(f"diagnosed {report['n_records']} records; {report['n_valid_diagnostics']} valid gates; "
          "uncalibrated, no reward/success claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
