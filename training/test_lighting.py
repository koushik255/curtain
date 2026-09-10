import random
import unittest
import numpy as np
import torch
from PIL import Image

from training.lighting import LightingConfig, RandomLighting
from training.train import CurtainEncoder, contrastive_loss, training_transform


class LightingTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.random.default_rng(12).integers(0, 256, (48, 64, 3), dtype=np.uint8))

    def test_clean_copy(self):
        transform = RandomLighting(LightingConfig(clean_probability=1, mild_probability=0))
        result = transform(self.image)
        self.assertIsNot(result, self.image)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(self.image))

    def test_seed_and_input_preserved(self):
        transform = RandomLighting(LightingConfig(clean_probability=0, mild_probability=0))
        before = np.asarray(self.image).copy()
        torch.manual_seed(10)
        first = transform(self.image)
        torch.manual_seed(10)
        second = transform(self.image)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        np.testing.assert_array_equal(before, np.asarray(self.image))
        self.assertFalse(np.array_equal(before, np.asarray(first)))
        self.assertEqual(first.size, self.image.size)
        self.assertEqual(first.mode, 'RGB')

    def test_invalid_config(self):
        for kwargs in ({'clean_probability': 0.8, 'mild_probability': 0.8},
                       {'gamma': (0, 1)}, {'temperature_strength': 0.5}):
            with self.assertRaises(ValueError):
                LightingConfig(**kwargs)
        with self.assertRaises(ValueError):
            training_transform(32, 'unknown')

    def test_legacy_default_unchanged(self):
        random.seed(3); torch.manual_seed(3)
        first = training_transform(32)(self.image)
        random.seed(3); torch.manual_seed(3)
        second = training_transform(32, 'legacy')(self.image)
        torch.testing.assert_close(first, second)

    def test_training_step(self):
        torch.set_num_threads(2)
        transform = training_transform(32, 'lighting-v1')
        first = torch.stack([transform(self.image) for _ in range(2)])
        second = torch.stack([transform(self.image) for _ in range(2)])
        self.assertEqual(first.shape, (2, 3, 32, 32))
        self.assertTrue(torch.isfinite(first).all())
        self.assertFalse(torch.equal(first, second))
        model = CurtainEncoder(128)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        before = model.backbone.fc.weight.detach().clone()
        loss = contrastive_loss(model(first), model(second), 0.07)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.backbone.fc.weight))


if __name__ == '__main__':
    unittest.main()
