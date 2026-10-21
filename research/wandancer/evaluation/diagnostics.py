"""Uncalibrated motion/music diagnostics, not a training reward or quality judge."""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class AlignmentParameters:
    sigma_seconds: float = 0.05
    min_valid_fraction: float = 0.8
    min_music_coverage: float = 0.5
    min_motion_coverage: float = 0.5
    coverage_radius_sigma: float = 2.0
    max_motion_event_hz: float = 4.0
    min_motion_separation_seconds: float = 0.12
    phase_search_seconds: float = 0.25
    phase_step_seconds: float = 0.01

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if any(value > 1 for value in (self.min_valid_fraction, self.min_music_coverage, self.min_motion_coverage)):
            raise ValueError("fraction thresholds must not exceed one")
        if self.phase_step_seconds > self.phase_search_seconds:
            raise ValueError("phase step exceeds search radius")


@dataclass(frozen=True)
class PoseParameters:
    # Threshold is coordinate-units/second and must be calibrated per extractor.
    min_activity: float
    coordinate_units: str
    smoothing_seconds: float = 0.10
    prominence_fraction: float = 0.15
    min_separation_seconds: float = 0.20
    max_direction_reversal_fraction: float = 0.45

    def __post_init__(self):
        if not self.coordinate_units:
            raise ValueError("coordinate_units is required")
        for name in ("min_activity", "smoothing_seconds", "prominence_fraction",
                     "min_separation_seconds", "max_direction_reversal_fraction"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.prominence_fraction > 1 or self.max_direction_reversal_fraction > 1:
            raise ValueError("fraction thresholds must not exceed one")


def _events(values, name):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite one-dimensional seconds array")
    if np.any(np.diff(result) <= 0):
        raise ValueError(f"{name} must be strictly increasing (no duplicate events)")
    return result


def _window(window):
    result = np.asarray(window, dtype=float)
    if result.shape != (2,) or not np.all(np.isfinite(result)) or result[1] <= result[0]:
        raise ValueError("window_seconds must contain finite start < end")
    return float(result[0]), float(result[1])


def _clip(events, window):
    return events[(events >= window[0]) & (events <= window[1])]


def nearest_distances(source, target):
    """Absolute distance in seconds; absent targets yield infinity, not a reward."""
    if not len(source):
        return np.array([], dtype=float)
    if not len(target):
        return np.full(len(source), np.inf)
    indices = np.searchsorted(target, source)
    left = target[np.maximum(indices - 1, 0)]
    right = target[np.minimum(indices, len(target) - 1)]
    return np.minimum(np.abs(source - left), np.abs(source - right))


def _direction(source, target, sigma, radius):
    distances = nearest_distances(source, target)
    similarity = np.exp(-0.5 * (distances / sigma) ** 2)
    return {
        "source_count": len(source),
        "gaussian_nn_mean": float(similarity.mean()) if len(similarity) else 0.0,
        "coverage": float(np.mean(distances <= radius)) if len(distances) else 0.0,
        "nearest_distances_seconds": [float(x) if np.isfinite(x) else None for x in distances],
        "nearest_similarities": similarity.tolist(),
    }


def _tempo_phase(music, motion, params):
    result = {"tempo_ratio_motion_to_music": None, "tempo_label": "insufficient_events",
              "motion_interval_cv": None, "best_shift_applied_to_music_seconds": None,
              "best_shift_music_to_motion_similarity": None,
              "phase_search_is_descriptive_not_a_reward": True}
    if len(music) >= 3 and len(motion) >= 3:
        music_intervals, motion_intervals = np.diff(music), np.diff(motion)
        ratio = float(np.median(music_intervals) / np.median(motion_intervals))
        labels = [(0.5, "half_time_candidate"), (1.0, "same_tempo_candidate"),
                  (2.0, "double_time_candidate")]
        candidate = min(labels, key=lambda item: abs(np.log2(ratio / item[0])))
        result["tempo_ratio_motion_to_music"] = ratio
        result["tempo_label"] = candidate[1] if abs(np.log2(ratio / candidate[0])) <= 0.15 else "other_tempo"
        result["motion_interval_cv"] = float(np.std(motion_intervals) / np.mean(motion_intervals))
    if len(music) and len(motion):
        count = int(np.floor(params.phase_search_seconds / params.phase_step_seconds))
        shifts = np.arange(-count, count + 1) * params.phase_step_seconds
        # Avoid edge-event cherry-picking: the exact same events are used at every shift.
        scores = [float(np.exp(-0.5 * (nearest_distances(music + s, motion) /
                                      params.sigma_seconds) ** 2).mean()) for s in shifts]
        best = min(range(len(shifts)), key=lambda i: (-scores[i], abs(shifts[i])))
        result["best_shift_applied_to_music_seconds"] = float(shifts[best])
        result["best_shift_music_to_motion_similarity"] = scores[best]
    return result


def diagnose_events(audio_beats_seconds, motion_events_seconds, window_seconds,
                    motion_quality=None, parameters=None):
    """Report both NN directions and independent validity gates; never combine scores."""
    params = parameters or AlignmentParameters()
    window = _window(window_seconds)
    all_music = _events(audio_beats_seconds, "audio_beats_seconds")
    all_motion = _events(motion_events_seconds, "motion_events_seconds")
    music, motion = _clip(all_music, window), _clip(all_motion, window)
    quality = dict(motion_quality or {})
    reasons = []
    if not len(music):
        reasons.append("empty_music")
    if not len(motion):
        reasons.append("empty_motion")
    valid_fraction = quality.get("valid_fraction")
    if valid_fraction is None:
        reasons.append("missing_motion_confidence")
    elif not np.isfinite(valid_fraction) or not 0 <= valid_fraction <= 1:
        raise ValueError("valid_fraction must be finite in [0,1]")
    elif valid_fraction < params.min_valid_fraction:
        reasons.append("low_motion_confidence")
    activity = quality.get("activity_score")
    minimum = quality.get("min_activity")
    if activity is None or minimum is None or not quality.get("activity_units"):
        reasons.append("missing_activity_gate")
    elif (not np.isfinite(activity) or activity < 0 or not np.isfinite(minimum) or minimum <= 0):
        raise ValueError("activity must be nonnegative; min_activity must be positive, both finite")
    elif activity < minimum:
        reasons.append("inactive_motion")
    if quality.get("high_frequency_motion", False):
        reasons.append("high_frequency_motion")
    density = float(len(motion) / (window[1] - window[0]))
    if density > params.max_motion_event_hz:
        reasons.append("high_motion_event_density")
    if len(motion) > 1 and np.min(np.diff(motion)) < params.min_motion_separation_seconds:
        reasons.append("closely_spaced_motion_events")
    radius = params.coverage_radius_sigma * params.sigma_seconds
    motion_to_music = _direction(motion, music, params.sigma_seconds, radius)
    music_to_motion = _direction(music, motion, params.sigma_seconds, radius)
    if len(music) and music_to_motion["coverage"] < params.min_music_coverage:
        reasons.append("low_music_coverage")
    if len(motion) and motion_to_music["coverage"] < params.min_motion_coverage:
        reasons.append("low_motion_coverage")
    temporal = _tempo_phase(music, motion, params)
    flags = ["uncalibrated_diagnostic_not_reward", "rgb_and_identity_not_evaluated"]
    if temporal["motion_interval_cv"] is not None and temporal["motion_interval_cv"] < 0.05:
        flags.append("regular_motion_template_candidate_not_success")
    return {
        "schema_version": 1, "valid": not reasons, "invalid_reasons": reasons,
        "interpretation_flags": flags, "parameters": asdict(params),
        "window_seconds": list(window), "audio_beats_seconds": music.tolist(),
        "motion_events_seconds": motion.tolist(),
        "excluded_audio_events": len(all_music) - len(music),
        "excluded_motion_events": len(all_motion) - len(motion),
        "raw_components": {"motion_to_music": motion_to_music,
                           "music_to_motion": music_to_motion,
                           "motion_event_density_hz": density},
        "motion_quality": quality, "temporal_diagnostics": temporal,
    }


def extract_pose_events(joints, valid, timestamps_seconds, parameters, joint_indices=None):
    """Fixed smoothed-speed minima heuristic; NOT a reproduced FACT implementation.

    No interpolation across invalid frames. Selected joints must all be valid for
    a frame to count. Absolute selected-joint speed preserves translation; it does
    not establish kinematic correctness or music-conditioned intent.
    """
    pose = np.asarray(joints, dtype=float)
    times = _events(timestamps_seconds, "timestamps_seconds")
    mask = np.asarray(valid)
    if pose.ndim != 3 or pose.shape[-1] != 3 or pose.shape[0] != len(times) or len(times) < 3:
        raise ValueError("joints must be (T,J,3), with T>=3 timestamps and J>=1")
    if pose.shape[1] == 0 or mask.dtype != np.bool_:
        raise ValueError("valid must be a boolean array; convert confidence with an explicit threshold")
    if mask.shape == (len(times),):
        mask = np.broadcast_to(mask[:, None], pose.shape[:2])
    if mask.shape != pose.shape[:2]:
        raise ValueError("valid must have shape (T,) or (T,J)")
    indices = list(range(pose.shape[1])) if joint_indices is None else list(joint_indices)
    if not indices or any(not isinstance(i, (int, np.integer)) or i < 0 or i >= pose.shape[1] for i in indices):
        raise ValueError("joint_indices must contain in-range integers")
    if len(set(indices)) != len(indices):
        raise ValueError("joint_indices must not repeat joints")
    pose = pose[:, indices]
    frame_valid = np.all(mask[:, indices] & np.all(np.isfinite(pose), axis=2), axis=1)
    interval_valid = frame_valid[1:] & frame_valid[:-1]
    velocity = np.diff(pose, axis=0) / np.diff(times)[:, None, None]
    speeds = np.median(np.linalg.norm(velocity, axis=2), axis=1)
    speeds[~interval_valid] = np.nan
    speed_times = (times[:-1] + times[1:]) / 2
    smooth = np.full(len(speeds), np.nan)
    half_width = parameters.smoothing_seconds / 2
    for i, t in enumerate(speed_times):
        left = int(np.searchsorted(speed_times, t - half_width, side="left"))
        right = int(np.searchsorted(speed_times, t + half_width, side="right"))
        if np.all(interval_valid[left:right]):
            smooth[i] = float(np.mean(speeds[left:right]))
    finite = smooth[np.isfinite(smooth)]
    activity = float(np.median(finite)) if len(finite) else 0.0
    reversal_valid = interval_valid[1:] & interval_valid[:-1]
    dot = np.sum(velocity[1:] * velocity[:-1], axis=2)
    reversal = np.mean(dot < 0, axis=1)
    reversal_fraction = float(np.mean(reversal[reversal_valid])) if np.any(reversal_valid) else 0.0
    spread = float(np.percentile(finite, 90) - np.percentile(finite, 10)) if len(finite) else 0.0
    prominence = parameters.prominence_fraction * max(spread, parameters.min_activity)
    candidates = []
    for i in range(1, len(smooth) - 1):
        if not np.all(np.isfinite(smooth[i - 1:i + 2])):
            continue
        if not (smooth[i] < smooth[i - 1] and smooth[i] <= smooth[i + 1]):
            continue
        left = np.searchsorted(speed_times, speed_times[i] - parameters.min_separation_seconds)
        right = np.searchsorted(speed_times, speed_times[i] + parameters.min_separation_seconds, side="right")
        neighborhood = smooth[left:right]
        if not np.all(np.isfinite(neighborhood)):
            continue
        strength = min(np.max(smooth[left:i + 1]), np.max(smooth[i:right])) - smooth[i]
        if strength >= prominence:
            candidates.append((float(strength), float(speed_times[i])))
    selected = []
    if activity >= parameters.min_activity:
        for _, t in sorted(candidates, reverse=True):
            if all(abs(t - other) >= parameters.min_separation_seconds for other in selected):
                selected.append(t)
    return {
        "motion_events_seconds": sorted(selected),
        "motion_quality": {"activity_score": activity, "min_activity": parameters.min_activity,
                           "activity_units": parameters.coordinate_units + "/second",
                           "valid_fraction": float(frame_valid.mean()),
                           "direction_reversal_fraction": reversal_fraction,
                           "high_frequency_motion": reversal_fraction > parameters.max_direction_reversal_fraction},
        "extraction": {"method": "smoothed_selected_joint_speed_local_minima_v1_not_FACT",
                       "calibration_status": "not_calibrated_on_real_video_NLF",
                       "parameters": asdict(parameters), "joint_indices": indices,
                       "speed_timestamps_seconds": speed_times.tolist(),
                       "smoothed_speed": [float(x) if np.isfinite(x) else None for x in smooth],
                       "absolute_prominence_threshold": prominence,
                       "no_interpolation_over_invalid_frames": True},
    }


def compare_beat_shifts(audio_beats_seconds, motion_events_seconds, window_seconds,
                        shifts_seconds, motion_quality, parameters=None):
    """Same motion, shifted beats; compare on shared support without wrapping."""
    window = _window(window_seconds)
    music = _events(audio_beats_seconds, "audio_beats_seconds")
    motion = _events(motion_events_seconds, "motion_events_seconds")
    comparisons = []
    for shift in shifts_seconds:
        if not np.isfinite(shift) or shift == 0:
            raise ValueError("counterfactual shifts must be finite and nonzero")
        support = (window[0] + max(0, shift), window[1] + min(0, shift))
        if support[1] <= support[0]:
            comparisons.append({"shift_seconds": float(shift), "valid": False,
                                "invalid_reasons": ["no_common_temporal_support"]})
            continue
        baseline = diagnose_events(music, motion, support, motion_quality, parameters)
        shifted = diagnose_events(music + shift, motion, support, motion_quality, parameters)
        valid = baseline["valid"] and shifted["valid"]
        comparisons.append({"shift_seconds": float(shift), "window_seconds": list(support),
                            "valid": valid, "baseline": baseline, "shifted": shifted,
                            "raw_baseline_minus_shifted": {
                                direction: baseline["raw_components"][direction]["gaussian_nn_mean"] -
                                shifted["raw_components"][direction]["gaussian_nn_mean"]
                                for direction in ("motion_to_music", "music_to_motion")},
                            "interpretation": "descriptive alignment sensitivity, not proof of generation audio use"})
    return comparisons


def aggregate_paired_null(records, pairs, bootstrap_seed=0, bootstrap_samples=1000):
    """Pair within frozen conditions/seed, then aggregate seeds within case clusters."""
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be a positive integer")
    by_id = {}
    for record in records:
        sample_id = record.get("sample_id")
        if not sample_id or sample_id in by_id:
            raise ValueError("records must have unique nonempty sample_id")
        by_id[sample_id] = record
    accepted, rejected, used = [], [], set()
    for pair in pairs:
        pair_id = pair.get("pair_id")
        if not pair_id or pair_id in used:
            raise ValueError("pairs must have unique nonempty pair_id")
        used.add(pair_id)
        target = by_id.get(pair.get("target_sample_id"))
        null = by_id.get(pair.get("null_sample_id"))
        reason = None
        if target is None or null is None:
            reason = "missing_pair_member"
        elif target["sample_id"] == null["sample_id"]:
            reason = "identical_pair_member"
        elif any(target.get(key) is None or target.get(key) != null.get(key)
                 for key in ("case_id", "seed", "conditioning_group_id")):
            reason = "unmatched_case_seed_or_frozen_conditions"
        elif not target.get("diagnostics", {}).get("valid") or not null.get("diagnostics", {}).get("valid"):
            reason = "invalid_diagnostics_member"
        elif any(target["diagnostics"].get(key) != null["diagnostics"].get(key)
                 for key in ("parameters", "window_seconds")):
            reason = "unmatched_evaluation_parameters_or_window"
        if reason:
            rejected.append({"pair_id": pair_id, "reason": reason})
            continue
        accepted.append({"pair_id": pair_id, "case_id": target["case_id"], "seed": target["seed"],
                         "delta_target_minus_null": {
                             direction: target["diagnostics"]["raw_components"][direction]["gaussian_nn_mean"] -
                             null["diagnostics"]["raw_components"][direction]["gaussian_nn_mean"]
                             for direction in ("motion_to_music", "music_to_motion")}})
    # Repeated nulls/candidates for one seed are not independent replicates.
    cases = {}
    for pair in accepted:
        cases.setdefault(pair["case_id"], {}).setdefault(pair["seed"], []).append(pair)
    components = {}
    rng = np.random.default_rng(bootstrap_seed)
    for direction in ("motion_to_music", "music_to_motion"):
        case_means = [float(np.mean([np.mean([p["delta_target_minus_null"][direction] for p in seed_pairs])
                                    for seed_pairs in seeds.values()])) for seeds in cases.values()]
        interval = None
        if len(case_means) >= 5:
            resamples = rng.choice(case_means, size=(bootstrap_samples, len(case_means)), replace=True).mean(axis=1)
            interval = [float(x) for x in np.percentile(resamples, [2.5, 97.5])]
        components[direction] = {"case_balanced_mean_delta": float(np.mean(case_means)) if case_means else None,
                                 "case_cluster_bootstrap_95pct_interval": interval}
    return {"accepted_pairs": accepted, "rejected_pairs": rejected,
            "n_accepted_pairs": len(accepted), "n_rejected_pairs": len(rejected),
            "n_case_clusters": len(cases),
            "n_case_seed_blocks": sum(len(seeds) for seeds in cases.values()),
            "components": components, "bootstrap_seed": bootstrap_seed,
            "bootstrap_samples": bootstrap_samples,
            "uncertainty_warning": "fewer_than_5_case_clusters_no_interval" if len(cases) < 5 else
                "exploratory_case_cluster_interval_assumes_independent_cases_not_confirmatory",
            "success_claim": None}
