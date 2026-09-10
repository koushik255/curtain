import unittest
import numpy as np
from curtain_ml.retrieval import normalize_rows


class NormalizationTests(unittest.TestCase):
    def test_unit_rows_and_no_mutation(self):
        data = np.array([[3, 4], [0, 2]], dtype=np.float16)
        saved = data.copy()
        result = normalize_rows(data)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1, atol=1e-6)
        np.testing.assert_array_equal(data, saved)
        self.assertEqual(result.dtype, np.float32)

    def test_reject_invalid(self):
        for data in ([[0, 0]], [[float('nan'), 1]], [1, 2]):
            with self.assertRaises(ValueError):
                normalize_rows(np.array(data))

    def test_cosine_not_vector_length(self):
        data = normalize_rows(np.array([[1.001, 0], [1, 0]]))
        np.testing.assert_allclose(data @ np.array([1., 0.]), [1, 1])


if __name__ == '__main__':
    unittest.main()
