import unittest
from scripts.build_direction_dataset import night_exclusion_reason


class NightProvenanceTests(unittest.TestCase):
    def test_blank_manifest_type_does_not_hide_attacks(self):
        meta = {'sample_type': '', 'source_image': 'D:/light_attack/source.jpg'}
        self.assertEqual(night_exclusion_reason('ordinary.jpg', meta), 'augmentation_provenance')
        self.assertEqual(night_exclusion_reason('001_existing_source_attack.jpg', {'sample_type': ''}),
                         'augmentation_provenance')

    def test_original_tracked_capture_and_missing_manifest(self):
        self.assertIsNone(night_exclusion_reason('20260615_12.jpg', {'sample_type': '', 'group_id': 'ZJ1'}))
        self.assertEqual(night_exclusion_reason('ordinary.jpg', {}), 'untracked')


if __name__ == '__main__':
    unittest.main()
