"""Uncalibrated SMPL arm observations on existing poses; no quality/reward verdict."""
import numpy as np

SMPL_PARENTS = (4294967295, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21)


def arm_observations(joints, valid, timestamps, parents, min_duration_seconds=0.2):
    if tuple(parents) != SMPL_PARENTS:
        raise ValueError('Verified SMPL24 topology required; laterality still needs RGB validation')
    p = np.asarray(joints, dtype=np.float64)
    v = np.asarray(valid)
    t = np.asarray(timestamps, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (24, 3) or len(p) < 3:
        raise ValueError('At least three SMPL24 xyz frames required')
    if v.shape != (len(p),) or v.dtype.kind != 'b':
        raise ValueError('Explicit boolean frame validity required')
    if t.shape != (len(p),) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Finite strictly increasing decoded timestamps required')
    if not np.isfinite(min_duration_seconds) or min_duration_seconds <= 0:
        raise ValueError('Positive event duration required')
    torso = p[:, 12] - p[:, 0]
    length = np.linalg.norm(torso, axis=1)
    good = v & np.isfinite(p).all(axis=(1, 2)) & (length > 1e-8)
    up = np.divide(torso, length[:, None], out=np.zeros_like(torso), where=length[:, None] > 1e-8)
    report = {'method': 'torso_relative_wrist_height_and_upper_arm_elevation',
              'coordinate_units': 'dimensionless torso ratio and degrees',
              'thresholds_calibrated': False, 'laterality_RGB_verified': False,
              'instruction_success_approved': False, 'valid_fraction': float(good.mean()), 'arms': {}}
    for side, shoulder, elbow, wrist in [('SMPL_left', 16, 18, 20), ('SMPL_right', 17, 19, 21)]:
        arm = p[:, elbow] - p[:, shoulder]
        arm_length = np.linalg.norm(arm, axis=1)
        mask = good & (arm_length > 1e-8)
        direction = np.divide(arm, arm_length[:, None], out=np.zeros_like(arm), where=arm_length[:, None] > 1e-8)
        elevation = np.degrees(np.arccos(np.clip(np.sum(direction * -up, axis=1), -1, 1)))
        height = np.divide(np.sum((p[:, wrist] - p[:, shoulder]) * up, axis=1), length,
                           out=np.zeros(len(p)), where=length > 1e-8)
        variants = []
        for cutoff in (0.0, 0.1, 0.2):
            active = mask & (height >= cutoff) & (elevation >= 60)
            episodes = []
            start = None
            for i, high in enumerate(active):
                if high and start is None:
                    start = i
                if start is not None and (not high or i == len(active) - 1):
                    end = i if high else i - 1
                    if t[end] - t[start] >= min_duration_seconds:
                        episodes.append({'start_seconds': float(t[start]), 'end_seconds': float(t[end]),
                                         'starts_at_clip_boundary': start == 0,
                                         'onset_observed_from_valid_low': start > 0 and bool(mask[start - 1])})
                    start = None
            variants.append({'wrist_height_torso_ratio_threshold': cutoff, 'upper_arm_elevation_threshold_degrees': 60,
                             'min_duration_seconds': min_duration_seconds, 'episodes': episodes,
                             'active_valid_frame_fraction': float(active.sum() / mask.sum()) if mask.any() else None})
        report['arms'][side] = {'valid_fraction': float(mask.mean()),
                               'median_upper_arm_elevation_degrees': float(np.median(elevation[mask])) if mask.any() else None,
                               'p95_wrist_height_torso_ratio': float(np.percentile(height[mask], 95)) if mask.any() else None,
                               'sensitivity': variants}
    return report
