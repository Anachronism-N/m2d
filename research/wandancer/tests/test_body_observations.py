import importlib.util
from pathlib import Path
import unittest

import numpy as np

path = Path(__file__).resolve().parents[1] / 'evaluation/body_observations.py'
spec = importlib.util.spec_from_file_location('body_observations', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BodyObservationTests(unittest.TestCase):
    def setUp(self):
        self.pose = np.zeros((31, 24, 3))
        self.times = np.arange(31) / 30
        self.valid = np.ones(31, dtype=bool)
        self.pose[:, 12, 1] = 1
        for shoulder, elbow, wrist, x in [(16, 18, 20, -0.3), (17, 19, 21, 0.3)]:
            self.pose[:, shoulder] = [x, 0.8, 0]
            self.pose[:, elbow] = [x, 0.5, 0]
            self.pose[:, wrist] = [x, 0.2, 0]

    def measure(self, pose=None, valid=None, times=None, parents=None):
        return module.arm_observations(self.pose if pose is None else pose,
                                       self.valid if valid is None else valid,
                                       self.times if times is None else times,
                                       module.SMPL_PARENTS if parents is None else parents)

    def raise_left(self, start=0):
        self.pose[start:, 18] = [-0.6, 0.8, 0]
        self.pose[start:, 20] = [-0.7, 1.2, 0]

    def test_down_arms_do_not_invent_events(self):
        report = self.measure()
        for arm in report['arms'].values():
            self.assertEqual(arm['sensitivity'][0]['episodes'], [])
        self.assertFalse(report['instruction_success_approved'])

    def test_observed_onset_and_clip_boundary_are_different(self):
        self.raise_left(10)
        event = self.measure()['arms']['SMPL_left']['sensitivity'][1]['episodes'][0]
        self.assertAlmostEqual(event['start_seconds'], 10 / 30)
        self.assertTrue(event['onset_observed_from_valid_low'])
        self.raise_left(0)
        event = self.measure()['arms']['SMPL_left']['sensitivity'][1]['episodes'][0]
        self.assertTrue(event['starts_at_clip_boundary'])
        self.assertFalse(event['onset_observed_from_valid_low'])

    def test_rigid_transform_and_scale_invariance(self):
        self.raise_left(10)
        before = self.measure()
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        after = self.measure(self.pose @ rotation.T * 2 + [3, -7, 4])
        for side in before['arms']:
            self.assertAlmostEqual(before['arms'][side]['p95_wrist_height_torso_ratio'],
                                   after['arms'][side]['p95_wrist_height_torso_ratio'])
            self.assertEqual(before['arms'][side]['sensitivity'], after['arms'][side]['sensitivity'])

    def test_invalid_gap_not_bridged(self):
        self.raise_left(0)
        self.valid[15] = False
        self.pose[15] = np.nan
        events = self.measure()['arms']['SMPL_left']['sensitivity'][0]['episodes']
        self.assertEqual(len(events), 2)
        self.assertLess(events[0]['end_seconds'], events[1]['start_seconds'])
        self.assertFalse(events[1]['onset_observed_from_valid_low'])

    def test_input_guards_and_unverified_topology(self):
        with self.assertRaises(ValueError):
            self.measure(valid=np.ones(31))
        with self.assertRaises(ValueError):
            self.measure(times=np.zeros(31))
        with self.assertRaises(ValueError):
            self.measure(parents=tuple(range(24)))
        with self.assertRaises(ValueError):
            self.measure(pose=self.pose[:, :22])

    def test_degenerate_torso_and_no_valid_frames_are_reported(self):
        report = self.measure(pose=np.zeros_like(self.pose))
        self.assertEqual(report['valid_fraction'], 0)
        for arm in report['arms'].values():
            self.assertIsNone(arm['p95_wrist_height_torso_ratio'])
        self.assertFalse(self.measure(valid=np.zeros(31, dtype=bool))['instruction_success_approved'])


if __name__ == '__main__':
    unittest.main()
