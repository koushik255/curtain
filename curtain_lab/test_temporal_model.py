import unittest
import torch
from temporal_model import TemporalHead, split, reverse_flow


class TemporalChecks(unittest.TestCase):
    def test_order_and_learning(self):
        torch.manual_seed(3)
        model=TemporalHead().eval()
        x=torch.randn(6,4,128)
        z=model(x)
        self.assertEqual(z.shape,(6,128))
        self.assertTrue(torch.allclose(z.norm(dim=1),torch.ones(6),atol=1e-5))
        self.assertGreater((z-model(x.flip(1))).abs().max().item(),.01)
        before=model.input.weight.detach().clone()
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        loss=(model.flow(z)-torch.randn(6,96)).square().mean()
        loss.backward();opt.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before,model.input.weight))

    def test_motion_input_and_reversal(self):
        torch.manual_seed(4)
        model=TemporalHead().eval()
        x=torch.randn(3,4,128); flow=torch.randn(3,96)
        self.assertTrue(torch.equal(reverse_flow(reverse_flow(flow)),flow))
        self.assertGreater((model(x,flow)-model(x,-flow)).abs().max().item(),.01)

    def test_film_split(self):
        clips=[{'movie':name} for name in ['a/Star_Wars_1977','a/Ran_1985','a/Star_Wars_1977','b/Harakiri_1962']]
        tr,va=split(clips)
        self.assertEqual(tr.tolist(),[1]);self.assertEqual(va.tolist(),[0,2,3])


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
