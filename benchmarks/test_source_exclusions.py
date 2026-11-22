import unittest
from unittest.mock import patch

import numpy as np

from rcmindex import search


class SourceExclusions(unittest.TestCase):
    def setUp(self):
        self.engine = {'firsts':np.array([0,3,6]), 'ends':np.array([3,6,9]),
                       'active_movies':np.array([True,False,True]), 'stride':50}

    def test_span_boundaries_and_neighbours(self):
        np.testing.assert_array_equal(search.active_frames(self.engine,np.arange(9)),[0,1,2,6,7,8])
        np.testing.assert_array_equal(search.neighbours(self.engine,np.array([2,3,6]),reach=2),[0,1,2,6,7,8])

    def test_bad_candidates_removed_before_rescoring(self):
        class Index:
            def search(self,queries,count):
                return np.zeros((1,4)),np.array([[150,200,100,300]])
        self.engine['index']=Index()
        def vectors(engine,frames):
            self.assertFalse(any(3 <= f < 6 for f in frames))
            return frames
        def scores(frames,queries):
            return np.array([frames*.01],dtype=np.float32)
        with patch.object(search,'frame_vectors',side_effect=vectors),patch.object(search,'scores',side_effect=scores):
            ranked=search.search(self.engine,[np.ones((1,128),dtype=np.float32)],top=20)
        self.assertTrue(ranked)
        self.assertFalse(any(3 <= f < 6 for f,_ in ranked))

    def test_all_candidates_blocked(self):
        class Index:
            def search(self,queries,count):
                return np.zeros((1,2)),np.array([[150,200]])
        self.engine['index']=Index()
        with patch.object(search,'frame_vectors',side_effect=AssertionError('Should not read blocked vectors')):
            self.assertEqual(search.search(self.engine,np.ones((1,128),dtype=np.float32)),[])


if __name__=='__main__':unittest.main()
