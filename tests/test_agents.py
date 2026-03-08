import unittest
from agents.governor_agent import ResearchGovernor
from science.objective_functions import StabilityObjective

class TestAgents(unittest.TestCase):
    def test_governor_initialization(self):
        config = {
            "objective": {"type": "stability"},
            "budget": {"max_evaluations": 10},
            "constraints": {"bulk": {"Cu": 1.0}, "facet": [1, 1, 1]}
        }
        governor = ResearchGovernor(config)
        self.assertEqual(governor.max_evaluations, 10)
        self.assertIsInstance(governor.reward_function, StabilityObjective)

    def test_governor_budget(self):
        config = {"objective": {"type": "stability"}, "budget": {"max_evaluations": 2}}
        governor = ResearchGovernor(config)
        self.assertTrue(governor.has_budget())
        governor.consume_budget()
        governor.consume_budget()
        self.assertFalse(governor.has_budget())

if __name__ == '__main__':
    unittest.main()
