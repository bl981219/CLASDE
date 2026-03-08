import unittest
from science.experiment_graph import KnowledgeGraph, ScienceNode, NodeType, RelationType
from core.state import SurfaceState

class TestGraph(unittest.TestCase):
    def setUp(self):
        self.kg = KnowledgeGraph()

    def test_node_addition(self):
        node = ScienceNode("mat_1", NodeType.MATERIAL, {"composition": {"Cu": 1.0}})
        self.kg.add_node(node)
        self.assertIn("mat_1", self.kg.nodes)
        self.assertEqual(self.kg.nodes["mat_1"].node_type, NodeType.MATERIAL)

    def test_relation_addition(self):
        n1 = ScienceNode("mat_1", NodeType.MATERIAL)
        n2 = ScienceNode("surf_1", NodeType.SURFACE)
        self.kg.add_node(n1)
        self.kg.add_node(n2)
        self.kg.add_relation("mat_1", "surf_1", RelationType.HAS_SURFACE)
        self.assertTrue(self.kg.graph.has_edge("mat_1", "surf_1"))

    def test_record_experiment(self):
        state = SurfaceState(bulk_composition={"Cu": 1.0}, miller_index=(1, 1, 1), termination="Cu")
        self.kg.record_experiment(state, None, {"reward": -1.5}, {"iteration": 1})
        self.assertTrue(len(self.kg.nodes) >= 4) # Material, Surface, Structure, Calculation, Result

if __name__ == '__main__':
    unittest.main()
