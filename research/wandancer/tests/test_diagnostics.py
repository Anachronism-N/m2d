"""Synthetic CPU tests verify equations/guards, NOT real-video reward reliability."""
import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
from diagnostics import (AlignmentParameters, PoseParameters, aggregate_paired_null,
                         compare_beat_shifts, diagnose_events, extract_pose_events)
from run_diagnostics import evaluate_manifest, write_private_report


QUALITY = {"activity_score": 1.0, "min_activity": 0.1,
           "activity_units": "synthetic_units/second", "valid_fraction": 1.0}
BEATS = np.arange(0.5, 4.6, 0.5)
WINDOW = [0, 5]


def report(events=BEATS, quality=None, music=BEATS):
    return diagnose_events(music, events, WINDOW, QUALITY if quality is None else quality)


def record(sample, case="case", seed=42, score=1.0):
    result = report()
    for direction in ("motion_to_music", "music_to_motion"):
        result["raw_components"][direction]["gaussian_nn_mean"] = score
    return {"sample_id": sample, "case_id": case, "seed": seed,
            "conditioning_group_id": "frozen_conditions", "diagnostics": result}


class AlignmentTests(unittest.TestCase):
    def test_perfect_but_not_success_or_weighted_reward(self):
        result = report()
        self.assertTrue(result["valid"])
        for direction in ("motion_to_music", "music_to_motion"):
            self.assertEqual(result["raw_components"][direction]["gaussian_nn_mean"], 1)
            self.assertEqual(result["raw_components"][direction]["coverage"], 1)
        self.assertIn("regular_motion_template_candidate_not_success", result["interpretation_flags"])
        self.assertNotIn("reward", result)
        self.assertNotIn("success", result)

    def test_gaussian_seconds_not_squared_sigma_or_frames(self):
        result = diagnose_events([1.0], [1.05], [0, 2], QUALITY)
        expected = np.exp(-0.5)
        self.assertAlmostEqual(result["raw_components"]["motion_to_music"]["gaussian_nn_mean"], expected)
        self.assertEqual(result["parameters"]["sigma_seconds"], 0.05)

    def test_shifted_events_low_alignment_and_phase(self):
        result = report(BEATS + 0.20)
        self.assertFalse(result["valid"])
        self.assertIn("low_music_coverage", result["invalid_reasons"])
        self.assertLess(result["raw_components"]["music_to_motion"]["gaussian_nn_mean"], 0.01)
        self.assertAlmostEqual(result["temporal_diagnostics"]["best_shift_applied_to_music_seconds"], 0.20)
        self.assertEqual(result["temporal_diagnostics"]["tempo_label"], "same_tempo_candidate")

    def test_half_double_and_two_directions(self):
        half = report(BEATS[::2])
        self.assertEqual(half["raw_components"]["motion_to_music"]["gaussian_nn_mean"], 1)
        self.assertLess(half["raw_components"]["music_to_motion"]["gaussian_nn_mean"], 0.7)
        self.assertEqual(half["temporal_diagnostics"]["tempo_label"], "half_time_candidate")
        double = report(np.arange(0.5, 4.6, 0.25))
        self.assertEqual(double["raw_components"]["music_to_motion"]["gaussian_nn_mean"], 1)
        self.assertLess(double["raw_components"]["motion_to_music"]["gaussian_nn_mean"], 0.7)
        self.assertEqual(double["temporal_diagnostics"]["tempo_label"], "double_time_candidate")

    def test_empty_motion_gets_no_reward_in_either_direction(self):
        result = report([])
        self.assertFalse(result["valid"])
        self.assertIn("empty_motion", result["invalid_reasons"])
        for direction in ("motion_to_music", "music_to_motion"):
            self.assertEqual(result["raw_components"][direction]["gaussian_nn_mean"], 0)
        self.assertTrue(all(x is None for x in result["raw_components"]["music_to_motion"]["nearest_distances_seconds"]))
        json.dumps(result, allow_nan=False)

    def test_empty_music_invalid(self):
        self.assertIn("empty_music", report(music=[])["invalid_reasons"])

    def test_low_confidence_static_and_missing_quality(self):
        low = dict(QUALITY, valid_fraction=0.1)
        static = dict(QUALITY, activity_score=0.0)
        self.assertIn("low_motion_confidence", report(quality=low)["invalid_reasons"])
        self.assertIn("inactive_motion", report(quality=static)["invalid_reasons"])
        unknown = report(quality={})
        self.assertIn("missing_motion_confidence", unknown["invalid_reasons"])
        self.assertIn("missing_activity_gate", unknown["invalid_reasons"])

    def test_low_coverage_sparse_perfect_hits_not_success(self):
        result = report(BEATS[:2])
        self.assertEqual(result["raw_components"]["motion_to_music"]["gaussian_nn_mean"], 1)
        self.assertIn("low_music_coverage", result["invalid_reasons"])

    def test_low_motion_coverage_even_when_all_music_covered(self):
        result = report([0.5, 2.0, 3.5], music=[0.5])
        self.assertEqual(result["raw_components"]["music_to_motion"]["coverage"], 1)
        self.assertIn("low_motion_coverage", result["invalid_reasons"])

    def test_high_density_and_close_extrema_invalid(self):
        result = report(np.arange(0.1, 4.9, 0.025))
        self.assertIn("high_motion_event_density", result["invalid_reasons"])
        self.assertIn("closely_spaced_motion_events", result["invalid_reasons"])

    def test_input_boundaries(self):
        for events in ([1, 1], [2, 1], [float("nan")], [[1]]):
            with self.assertRaises(ValueError):
                report(events)
        with self.assertRaises(ValueError):
            AlignmentParameters(sigma_seconds=0)
        with self.assertRaises(ValueError):
            AlignmentParameters(min_music_coverage=1.1)
        with self.assertRaises(ValueError):
            diagnose_events(BEATS, BEATS, [1, 1], QUALITY)
        with self.assertRaises(ValueError):
            report(quality=dict(QUALITY, valid_fraction=2))

    def test_fixed_window_exclusions_recorded(self):
        result = diagnose_events([0.5, 1.5, 3.0], [0.5, 1.5, 3.0], [0, 2], QUALITY)
        self.assertEqual(result["excluded_audio_events"], 1)
        self.assertEqual(result["excluded_motion_events"], 1)

    def test_same_dance_shift_common_support_and_no_wrap(self):
        comparisons = compare_beat_shifts(BEATS, BEATS, WINDOW, [0.2, -0.2, 10], QUALITY)
        self.assertEqual(comparisons[0]["window_seconds"], [0.2, 5.0])
        self.assertEqual(comparisons[1]["window_seconds"], [0.0, 4.8])
        self.assertGreater(comparisons[0]["raw_baseline_minus_shifted"]["music_to_motion"], 0.9)
        self.assertFalse(comparisons[0]["valid"])
        self.assertEqual(comparisons[2]["invalid_reasons"], ["no_common_temporal_support"])
        with self.assertRaises(ValueError):
            compare_beat_shifts(BEATS, BEATS, WINDOW, [0], QUALITY)


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(0, 5, 1 / 100)
        self.valid = np.ones((len(self.times), 2), dtype=bool)
        self.params = PoseParameters(min_activity=0.1, coordinate_units="synthetic_units")

    def pose(self, x):
        joints = np.zeros((len(x), 2, 3))
        joints[:, :, 0] = x[:, None]
        return joints

    def test_static_and_monotone_do_not_manufacture_events(self):
        for x in (np.zeros(len(self.times)), self.times):
            extraction = extract_pose_events(self.pose(x), self.valid, self.times, self.params)
            self.assertEqual(extraction["motion_events_seconds"], [])
        static = extract_pose_events(self.pose(np.zeros(len(self.times))), self.valid, self.times, self.params)
        self.assertEqual(static["motion_quality"]["activity_score"], 0)
        self.assertIn("inactive_motion", diagnose_events(BEATS, [], WINDOW, static["motion_quality"])["invalid_reasons"])

    def test_synthetic_periodic_pose_records_extractor_not_FACT(self):
        extraction = extract_pose_events(self.pose(np.sin(2 * np.pi * self.times)),
                                         self.valid, self.times, self.params, [0])
        self.assertGreater(len(extraction["motion_events_seconds"]), 5)
        self.assertGreaterEqual(min(np.diff(extraction["motion_events_seconds"])), 0.20)
        self.assertIn("not_FACT", extraction["extraction"]["method"])
        self.assertEqual(extraction["extraction"]["joint_indices"], [0])
        self.assertEqual(extraction["extraction"]["calibration_status"], "not_calibrated_on_real_video_NLF")

    def test_high_frequency_direction_jitter_invalid(self):
        extraction = extract_pose_events(self.pose((-1.0) ** np.arange(len(self.times))),
                                         self.valid, self.times, self.params)
        self.assertTrue(extraction["motion_quality"]["high_frequency_motion"])
        result = diagnose_events(BEATS, extraction["motion_events_seconds"], WINDOW, extraction["motion_quality"])
        self.assertIn("high_frequency_motion", result["invalid_reasons"])

    def test_invalid_gaps_no_interpolation_and_low_coverage(self):
        valid = self.valid.copy()
        valid[100:400] = False
        extraction = extract_pose_events(self.pose(np.sin(2 * np.pi * self.times)), valid,
                                         self.times, self.params)
        self.assertLess(extraction["motion_quality"]["valid_fraction"], 0.8)
        self.assertTrue(any(value is None for value in extraction["extraction"]["smoothed_speed"]))
        self.assertFalse(any(1 <= t <= 4 for t in extraction["motion_events_seconds"]))

    def test_boolean_valid_and_nonfinite_selected_joint_handling(self):
        pose = self.pose(np.sin(2 * np.pi * self.times))
        with self.assertRaises(ValueError):
            extract_pose_events(pose, self.valid.astype(float), self.times, self.params)
        pose[:200, 0] = np.nan
        extraction = extract_pose_events(pose, self.valid, self.times, self.params, [0])
        self.assertLess(extraction["motion_quality"]["valid_fraction"], 0.8)
        with self.assertRaises(ValueError):
            extract_pose_events(pose, self.valid, self.times, self.params, [0, 0])


class AggregationTests(unittest.TestCase):
    def pair(self, i, case="case", seed=42):
        target, null = record(f"target{i}", case, seed, 0.9), record(f"null{i}", case, seed, 0.7)
        return [target, null], {"pair_id": f"pair{i}", "target_sample_id": target["sample_id"],
                               "null_sample_id": null["sample_id"]}

    def test_small_sample_no_fake_confidence(self):
        records, pair = self.pair(0)
        result = aggregate_paired_null(records, [pair])
        self.assertEqual(result["n_accepted_pairs"], 1)
        self.assertEqual(result["n_case_clusters"], 1)
        component = result["components"]["music_to_motion"]
        self.assertAlmostEqual(component["case_balanced_mean_delta"], 0.2)
        self.assertIsNone(component["case_cluster_bootstrap_95pct_interval"])
        self.assertIsNone(result["success_claim"])

    def test_unmatched_invalid_missing_pairs_remain_in_report(self):
        records, pair = self.pair(0)
        records[1]["seed"] = 99
        unmatched = aggregate_paired_null(records, [pair])
        self.assertEqual(unmatched["n_rejected_pairs"], 1)
        records[1]["seed"] = 42
        records[1]["diagnostics"]["valid"] = False
        self.assertEqual(aggregate_paired_null(records, [pair])["n_rejected_pairs"], 1)
        self.assertEqual(aggregate_paired_null([], [pair])["n_rejected_pairs"], 1)

    def test_unmatched_scoring_contract_rejected(self):
        records, pair = self.pair(0)
        records[1]["diagnostics"]["parameters"]["sigma_seconds"] = 0.10
        self.assertEqual(aggregate_paired_null(records, [pair])["n_rejected_pairs"], 1)
        records, pair = self.pair(0)
        records[1]["diagnostics"]["window_seconds"] = [1, 5]
        self.assertEqual(aggregate_paired_null(records, [pair])["n_rejected_pairs"], 1)

    def test_case_balancing_and_seed_blocks_no_pseudoreplicates(self):
        records, pairs = [], []
        for i in range(10):
            row, pair = self.pair(i, "case_A", seed=i)
            records.extend(row)
            pairs.append(pair)
        row, pair = self.pair(10, "case_B")
        row[0]["diagnostics"]["raw_components"]["music_to_motion"]["gaussian_nn_mean"] = 0.7
        records.extend(row)
        pairs.append(pair)
        result = aggregate_paired_null(records, pairs)
        self.assertEqual(result["n_case_clusters"], 2)
        self.assertEqual(result["n_case_seed_blocks"], 11)
        self.assertAlmostEqual(result["components"]["music_to_motion"]["case_balanced_mean_delta"], 0.1)
        self.assertIsNone(result["components"]["music_to_motion"]["case_cluster_bootstrap_95pct_interval"])

    def test_bootstrap_deterministic_exploratory(self):
        records, pairs = [], []
        for i in range(5):
            row, pair = self.pair(i, case=f"case{i}")
            records.extend(row)
            pairs.append(pair)
        a = aggregate_paired_null(records, pairs, bootstrap_seed=8, bootstrap_samples=100)
        b = aggregate_paired_null(records, pairs, bootstrap_seed=8, bootstrap_samples=100)
        self.assertEqual(a, b)
        self.assertIsNotNone(a["components"]["music_to_motion"]["case_cluster_bootstrap_95pct_interval"])
        self.assertIn("not_confirmatory", a["uncertainty_warning"])


class ManifestTests(unittest.TestCase):
    def manifest(self):
        return {"schema_version": 1, "cases": [
            {"sample_id": "synthetic_1", "case_id": "synthetic_case", "seed": 42,
             "conditioning_group_id": "synthetic_frozen", "window_seconds": WINDOW,
             "audio_beats_seconds": BEATS.tolist(), "motion_events_seconds": BEATS.tolist(),
             "motion_quality": QUALITY}], "beat_shifts_seconds": [0.2]}

    def test_manifest_failure_preserves_denominator(self):
        manifest = self.manifest()
        failed = copy.deepcopy(manifest["cases"][0])
        failed["sample_id"] = "failed_pose"
        failed.pop("motion_events_seconds")
        failed["pose_npz"] = "missing.npz"
        failed["pose_parameters"] = {"min_activity": 0.1, "coordinate_units": "synthetic_units"}
        manifest["cases"].append(failed)
        result = evaluate_manifest(manifest, Path.cwd())
        self.assertEqual(result["n_records"], 2)
        self.assertEqual(result["n_valid_diagnostics"], 1)
        self.assertEqual(result["records"][1]["failure"]["type"], "FileNotFoundError")
        self.assertIn("VideoReward_has_no_audio_input_and_cannot_score_music_sync", result["limitations"])

    def test_private_exclusive_output_and_original_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "manifest.json"
            original.write_text(json.dumps(self.manifest()))
            original_bytes = original.read_bytes()
            output = root / "private-report.json"
            write_private_report({"ok": True}, output, publication_root=root / "checkout")
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            with self.assertRaises(FileExistsError):
                write_private_report({"overwrite": True}, output, publication_root=root / "checkout")
            with self.assertRaises(FileExistsError):
                write_private_report({}, original, publication_root=root / "checkout")
            self.assertEqual(original.read_bytes(), original_bytes)
            with self.assertRaises(ValueError):
                write_private_report({}, root / "checkout" / "report.json", publication_root=root / "checkout")

    def test_cli_event_and_pose_no_torch_exit_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.manifest()
            times = np.arange(0, 5, 0.01)
            joints = np.zeros((len(times), 1, 3))
            joints[:, 0, 0] = np.sin(2 * np.pi * times)
            np.savez(root / "pose.npz", joints=joints, valid=np.ones(len(times), dtype=bool), timestamps_seconds=times)
            pose_case = copy.deepcopy(manifest["cases"][0])
            pose_case.pop("motion_events_seconds")
            pose_case.update(sample_id="synthetic_pose", pose_npz="pose.npz",
                             pose_parameters={"min_activity": 0.1, "coordinate_units": "synthetic_units"})
            manifest["cases"].append(pose_case)
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            original = path.read_bytes()
            output = root / "report.json"
            process = subprocess.run([sys.executable, str(ROOT / "evaluation" / "run_diagnostics.py"),
                                      "--manifest", str(path), "--output", str(output)],
                                     capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(path.read_bytes(), original)
            result = json.loads(output.read_text())
            self.assertEqual(result["n_records"], 2)
            self.assertIn("pose_extraction", result["records"][1])
            self.assertNotIn("torch", sys.modules)
            repeated = subprocess.run([sys.executable, str(ROOT / "evaluation" / "run_diagnostics.py"),
                                       "--manifest", str(path), "--output", str(output)], capture_output=True, text=True,
                                      env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
            self.assertEqual(repeated.returncode, 2)


if __name__ == "__main__":
    unittest.main()
