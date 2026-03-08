import unittest
import numpy as np
from optimization.surrogate_models import GaussianProcessModel
from optimization.acquisition_functions import ExpectedImprovement
from optimization.campaign_optimizer import CampaignOptimizer
from core.state import SurfaceState

class TestOptimizer(unittest.TestCase):
    def test_gp_update_predict(self):
        gp = GaussianProcessModel()
        state = SurfaceState(bulk_composition={"Cu": 1.0}, miller_index=(1, 1, 1), termination="Cu")
        # Initial predict
        mu, sigma = gp.predict(state)
        self.assertEqual(mu, 0.0)
        self.assertEqual(sigma, 1.0)
        
        # Update
        data = [{"state": state, "reward": -1.2}]
        gp.update(data)
        mu, sigma = gp.predict(state)
        self.assertAlmostEqual(mu, -1.2, places=1)

    def test_optimizer_recommend(self):
        gp = GaussianProcessModel()
        ei = ExpectedImprovement(best_observed_f=-2.0)
        optimizer = CampaignOptimizer(gp, ei)
        
        state = SurfaceState(bulk_composition={"Cu": 1.0}, miller_index=(1, 1, 1), termination="Cu")
        candidates = [(None, state)]
        
        action, rec_state = optimizer.recommend_next(candidates)
        self.assertEqual(rec_state, state)

if __name__ == '__main__':
    unittest.main()
