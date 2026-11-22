"""Checks for benchmark validity and preservation of augmentation labels."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from benchmarks.online_benchmark import SAMPLE, eligible_queries, freeze, read, restore, save
from benchmarks.web_appearance import Edit, render, variants


class BenchmarkValidity(unittest.TestCase):
    def test_sample_is_fixed_and_movie_partitions_do_not_overlap(self):
        sample=read(SAMPLE)
        self.assertEqual(sample['catalog_movies'],474)
        self.assertEqual(len(sample['movies']),95)
        movies=[r['movie'] for r in sample['movies']]
        self.assertEqual(len(set(movies)),95)
        self.assertEqual(movies,sorted(movies,key=str.casefold))
        dev={r['movie'] for r in sample['movies'] if r['partition']=='development'}
        test={r['movie'] for r in sample['movies'] if r['partition']=='test'}
        self.assertEqual((len(dev),len(test)),(57,38))
        self.assertFalse(dev & test)

    def test_replacement_cannot_reuse_review_of_old_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            save(out/'manifest.json',{'queries':[{'id':'case','metadata':{'sha256':'new'}}]})
            save(out/'reviews.json',{'case':{'sha256':'old','status':'eligible'}})
            self.assertEqual(eligible_queries(out),[])
            with self.assertRaisesRegex(ValueError,'require review'):
                freeze(SimpleNamespace(out=out))

    def test_restore_rejects_remote_image_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)/'run';snapshot=Path(folder)/'snapshot.json'
            save(snapshot,{'frozen':{'queries':[{'id':'query','image_url':'https://example.test/scene.jpg',
                                                'metadata':{'sha256':'expected'}}]}})
            with patch('benchmarks.online_benchmark.fetch',return_value=b'changed image'):
                with self.assertRaisesRegex(RuntimeError,'could not be restored'):
                    restore(SimpleNamespace(out=out,queries=snapshot,workers=1))
            self.assertFalse((out/'frozen.json').exists())
            self.assertIn('changed',read(out/'restore-ledger.json')[0]['error'])

    def test_known_frame_label_and_crop_pixels_survive_edit(self):
        pixels=np.arange(10*20*3,dtype=np.uint8).reshape(10,20,3)
        result,label=render(Image.fromarray(pixels),Edit(width=.5,height=.6,x=1,y=0,format='PNG'))
        self.assertEqual(label['source_crop_xyxy'],(10,0,20,6))
        self.assertEqual(label['retained_area'],.3)
        np.testing.assert_array_equal(np.asarray(result),pixels[:6,10:20])

    def test_lossy_recipe_combines_edits_and_preserves_label(self):
        rng=np.random.default_rng(0)
        source=Image.fromarray(rng.integers(0,256,(180,320,3),dtype=np.uint8))
        result,label=render(source,Edit(width=.4,height=.4,long_side=80,quality=15,
                                        gamma=1.3,subtitle='Example',player_controls=True))
        self.assertEqual(label['source_crop_xyxy'],(96,54,224,126))
        self.assertEqual(result.size,(80,45))
        self.assertAlmostEqual(label['retained_area'],.16)
        self.assertGreater(np.std(np.asarray(result)),5)
        self.assertEqual(len(label['encoding_sha256']),64)

    def test_augmentation_export_refuses_benchmark_movie(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            save(out/'frozen.json',{'queries':[{'movie':'Held_Out_2000'}]})
            with self.assertRaisesRegex(ValueError,'outside all benchmark movies'):
                variants(SimpleNamespace(out=out,movie='held_out_2000'))


if __name__=='__main__':unittest.main()
