import importlib.util
from pathlib import Path
import unittest

import numpy as np

p = Path(__file__).resolve().parents[1] / 'evaluation' / 'beat_diagnostics.py'
spec = importlib.util.spec_from_file_location('beat_diagnostics', p)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class BeatDiagnosticsTests(unittest.TestCase):
    def test_exact_alignment(self):
        result = module.diagnostics([0.5, 1, 1.5], [0.5, 1, 1.5], 2)
        self.assertEqual(result['motion_to_music_raw'], 1)
        self.assertEqual(result['music_to_motion_raw'], 1)
        self.assertTrue(result['eligible'])
        self.assertFalse(result['validated_rgb_quality'])

    def test_shift_reduces_alignment(self):
        result = module.shifted_comparison([0.5, 1, 1.5], [0.5, 1, 1.5], 2, 0.2)
        self.assertGreater(result['motion_to_music_delta'], 0.9)
        self.assertFalse(result['causal_music_use_proven'])

    def test_empty_is_not_rewarded(self):
        result = module.diagnostics([], [0.5, 1], 2)
        self.assertIsNone(result['motion_to_music_raw'])
        self.assertFalse(result['eligible'])

    def test_pose_failure_and_static_are_ineligible(self):
        result = module.diagnostics([0.5, 1], [0.5, 1], 2,
                                    valid_ratio=0.4, activity=0, min_activity=0.01)
        self.assertEqual(result['motion_to_music_raw'], 1)
        self.assertFalse(result['eligible'])
        self.assertIn('insufficient_pose_coverage', result['invalid_reasons'])
        self.assertIn('insufficient_activity', result['invalid_reasons'])

    def test_dense_events_flagged(self):
        events = np.arange(0, 2, 0.05)
        result = module.diagnostics(events, [0.5, 1, 1.5], 2)
        self.assertFalse(result['eligible'])
        self.assertIn('excessive_motion_event_density', result['invalid_reasons'])

    def test_unmeasured_activity_not_assumed_valid(self):
        result = module.diagnostics([0.5], [0.5], 1, min_activity=0.01)
        self.assertFalse(result['eligible'])

    def test_single_direction_can_hide_missing_beats(self):
        result = module.diagnostics([0.5], [0.5, 1, 1.5], 2)
        self.assertEqual(result['motion_to_music_raw'], 1)
        self.assertLess(result['music_to_motion_raw'], 0.5)

    def test_invalid_input(self):
        for events in ([1, 0.5], [0.5, 0.5], [np.nan], [-0.1], [3]):
            with self.assertRaises(ValueError):
                module.diagnostics(events, [0.5], 2)

    def test_boundary_shift_compares_matched_events(self):
        result = module.shifted_comparison([0.1, 1], [0.1, 1, 1.9], 2, 0.2)
        self.assertEqual(result['matched_music_event_count'], 2)
        self.assertEqual(result['original']['music_event_count'], 2)
        self.assertEqual(result['shifted']['music_event_count'], 2)

    def test_paired_summary_does_not_claim_population_effect(self):
        result = module.paired_summary([0.8, 0.6], [0.7, 0.7])
        self.assertAlmostEqual(result['mean_delta'], 0)
        self.assertEqual(result['n_pairs'], 2)
        self.assertFalse(result['population_claim_supported'])


if __name__ == '__main__':
    unittest.main()
