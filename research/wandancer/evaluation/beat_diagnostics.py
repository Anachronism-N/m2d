"""Equation-level event diagnostics; diagnostics.py is the full protocol entrypoint."""
import argparse
import json
from pathlib import Path

import numpy as np


def event_times(values, name):
    times = np.asarray(values, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError(name + ' must be a finite nonnegative 1D array of seconds')
    if len(times) > 1 and (np.diff(times) <= 0).any():
        raise ValueError(name + ' must be strictly increasing')
    return times


def nearest_score(source, target, sigma_seconds):
    if not len(source) or not len(target):
        return None
    idx = np.searchsorted(target, source)
    left = target[np.maximum(idx - 1, 0)]
    right = target[np.minimum(idx, len(target) - 1)]
    distance = np.minimum(np.abs(source - left), np.abs(source - right))
    return float(np.exp(-0.5 * (distance / sigma_seconds) ** 2).mean())


def diagnostics(motion_beats, music_beats, duration_seconds, *, sigma_seconds=0.05,
                valid_ratio=1.0, activity=None, min_activity=None,
                max_motion_events_per_second=8.0, min_valid_ratio=0.7):
    motion = event_times(motion_beats, 'motion_beats')
    music = event_times(music_beats, 'music_beats')
    if not np.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError('duration_seconds must be positive')
    if not np.isfinite(sigma_seconds) or sigma_seconds <= 0:
        raise ValueError('sigma_seconds must be positive')
    if not 0 <= valid_ratio <= 1 or not 0 <= min_valid_ratio <= 1:
        raise ValueError('valid ratios must be in [0,1]')
    if not np.isfinite(max_motion_events_per_second) or max_motion_events_per_second <= 0:
        raise ValueError('max_motion_events_per_second must be positive')
    if any(len(x) and x[-1] > duration_seconds for x in (motion, music)):
        raise ValueError('events exceed the declared clip duration')
    if activity is not None and (not np.isfinite(activity) or activity < 0):
        raise ValueError('activity must be finite and nonnegative')
    if min_activity is not None and (not np.isfinite(min_activity) or min_activity < 0):
        raise ValueError('min_activity must be finite and nonnegative')
    reasons = []
    if not len(motion):
        reasons.append('no_motion_events')
    if not len(music):
        reasons.append('no_music_events')
    if valid_ratio < min_valid_ratio:
        reasons.append('insufficient_pose_coverage')
    density = len(motion) / duration_seconds
    if density > max_motion_events_per_second:
        reasons.append('excessive_motion_event_density')
    if min_activity is not None:
        if activity is None:
            reasons.append('activity_not_measured')
        elif activity < min_activity:
            reasons.append('insufficient_activity')
    forward = nearest_score(motion, music, sigma_seconds)
    reverse = nearest_score(music, motion, sigma_seconds)
    return {
        'time_unit': 'seconds', 'sigma_seconds': sigma_seconds,
        'motion_to_music_raw': forward, 'music_to_motion_raw': reverse,
        'motion_event_count': len(motion), 'music_event_count': len(music),
        'motion_events_per_second': density, 'valid_ratio': valid_ratio,
        'activity': activity, 'eligible': not reasons, 'invalid_reasons': reasons,
        'thresholds': {'min_valid_ratio': min_valid_ratio,
                       'max_motion_events_per_second': max_motion_events_per_second,
                       'min_activity': min_activity},
        'validated_rgb_quality': False, 'validated_dance_reward': False,
        'caveats': ['event definitions and thresholds require real-dance calibration',
                    'periodic templates can score highly without good choreography',
                    'pose fitting does not prove RGB anatomical correctness'],
    }


def shifted_comparison(motion_beats, music_beats, duration_seconds, shift_seconds, **kwargs):
    if not np.isfinite(shift_seconds):
        raise ValueError('shift_seconds must be finite')
    music = event_times(music_beats, 'music_beats')
    shifted = music + shift_seconds
    # Compare only matching original events whose shifted times remain in the clip.
    keep = (shifted >= 0) & (shifted <= duration_seconds)
    original = diagnostics(motion_beats, music[keep], duration_seconds, **kwargs)
    altered = diagnostics(motion_beats, shifted[keep], duration_seconds, **kwargs)
    a, b = original['motion_to_music_raw'], altered['motion_to_music_raw']
    return {'shift_seconds': shift_seconds, 'matched_music_event_count': int(keep.sum()),
            'original': original, 'shifted': altered,
            'motion_to_music_delta': a - b if a is not None and b is not None else None,
            'causal_music_use_proven': False}


def paired_summary(values, null_values):
    a, b = np.asarray(values, dtype=float), np.asarray(null_values, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError('paired values must be nonempty equal-length 1D arrays')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('paired values must be finite')
    delta = a - b
    return {'n_pairs': len(a), 'mean_delta': float(delta.mean()),
            'median_delta': float(np.median(delta)),
            'sample_std_delta': float(delta.std(ddof=1)) if len(delta) > 1 else None,
            'individual_deltas': delta.tolist(),
            'population_claim_supported': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    allowed = {'motion_beats', 'music_beats', 'duration_seconds', 'sigma_seconds',
               'valid_ratio', 'activity', 'min_activity',
               'max_motion_events_per_second', 'min_valid_ratio'}
    unknown = set(data) - allowed
    if unknown:
        parser.error('unknown input fields: ' + ', '.join(sorted(unknown)))
    result = diagnostics(**data)
    # No overwrite of source, caches, or previous experiment results.
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write('\n')


if __name__ == '__main__':
    main()
