"""Checks for temporal direction and clip boundary/timestamp safety."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from sequence_experiment import clip_records, motion


class SequenceChecks(unittest.TestCase):
    def test_flow_direction_and_static(self):
        image = np.random.default_rng(0).integers(0, 256, (108, 192), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for k in range(4):
                path = Path(directory)/f'{k}.png'
                Image.fromarray(np.roll(image, 2*k, axis=1)).save(path)
                paths.append(str(path))
            forward, energy = motion({'paths': paths})
            reverse, _ = motion({'paths': paths[::-1]})
            _, static = motion({'paths': [paths[0]]*4})
        self.assertGreater(energy, .002)
        self.assertLess(static, 1e-5)
        cosine = forward@reverse/(np.linalg.norm(forward)*np.linalg.norm(reverse))
        self.assertLess(cosine, -.9)

    def test_clip_boundaries_and_sampling_gaps(self):
        catalog = {'movies': [{'movie':'film', 'first':0}],
                   'names':[f'collection/film/frame_{n}.jpg' for n in range(12)]}
        records = [{'movie':'film', 'shot_start_row':lo, 'shot_end_row':hi, 'row':row}
                   for lo, hi, row in [(0,3,1), (3,7,4), (7,12,10)]]
        with patch('sequence_experiment.pilot.frame_path', side_effect=lambda root, name:Path(name)):
            clips = clip_records(Path('/tmp'), catalog, records, 80)
            self.assertEqual(len(clips), 2)
            self.assertEqual(clips[0]['clip_seconds'], [1.5,2.,2.5,3.])
            self.assertEqual(clips[1]['clip_seconds'], [4.,4.5,5.,5.5])
            catalog['names'][5] = 'collection/film/frame_99.jpg'
            with self.assertRaisesRegex(ValueError, 'Nonconsecutive'):
                clip_records(Path('/tmp'), catalog, records, 80)


if __name__ == '__main__':
    unittest.main()
