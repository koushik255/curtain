import unittest
import torch
from PIL import Image
from curtain_ml.augmentations import RandomFrameCrop
from curtain_ml.training import PositivePairDataset, training_transform, CurtainEncoder, contrastive_loss


class FramingTests(unittest.TestCase):
    def test_bounds(self):
        transform = RandomFrameCrop(probability=1)
        torch.manual_seed(42)
        for _ in range(100):
            x,y,right,bottom = transform.box((1245,700))
            self.assertTrue(0 <= x < right <= 1245 and 0 <= y < bottom <= 700)
            self.assertGreaterEqual(right-x, round(1245*.75))
            self.assertGreaterEqual(bottom-y, 525)

    def test_identity_and_seed(self):
        self.assertEqual(RandomFrameCrop(0).box((100,60)), (0,0,100,60))
        torch.manual_seed(2); a=RandomFrameCrop(1).box((100,60))
        torch.manual_seed(2); b=RandomFrameCrop(1).box((100,60))
        self.assertEqual(a,b)

    def test_anchor_and_step(self):
        torch.set_num_threads(2)
        dataset=PositivePairDataset([],32,'crop-v1')
        self.assertIsInstance(dataset.transform.transforms[0],RandomFrameCrop)
        self.assertFalse(any(isinstance(t,RandomFrameCrop) for t in dataset.anchor_transform.transforms))
        image=Image.new('RGB',(160,90),(80,120,180))
        a=torch.stack([dataset.anchor_transform(image) for _ in range(2)])
        b=torch.stack([dataset.transform(image) for _ in range(2)])
        model=CurtainEncoder(128)
        loss=contrastive_loss(model(a),model(b),.07)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


if __name__=='__main__': unittest.main()
