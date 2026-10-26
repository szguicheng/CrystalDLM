import unittest
from unittest.mock import patch
import model


class ObjectivePair(unittest.TestCase):
    def test_generation_and_reading_both_contribute_with_matched_weight(self):
        import torch
        class Reading:
            def loss(self, rows, tok, mode, generator):
                self.rows, self.mode = rows, mode
                return torch.tensor(2.,requires_grad=True), {}
        instance=Reading()
        examples=[{'task':'G'},{'task':'Q'}]
        with patch.object(model.PILOT,'objective',return_value=(torch.tensor(6.,requires_grad=True),
                              {'clean_ar':4.,'denoise':2.})):
            loss,stats=model.objective_tasks(instance,None,examples,None,'flare')
        self.assertEqual(float(loss.detach()),4.)
        self.assertEqual(stats,dict(clean_ar=3.,denoise=1.,reading_ar=2.))
        self.assertEqual(instance.rows,[examples[1]])
        self.assertEqual(instance.mode,'ar')
        loss.backward()


if __name__=='__main__':unittest.main()
