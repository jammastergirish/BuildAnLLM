import torch
import torch.nn as nn
import unittest
from finetuning.training.sft_trainer import SFTTrainer
from pretraining.training.visualization import TrainingVisualizationMixin

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        # Add some layer-like naming
        self.layers = nn.ModuleList([nn.Linear(10, 10)])
        self.head = nn.Linear(10, 2)

    def forward(self, x):
        return self.head(self.layers[0](x))

class InteractiveTrainer(TrainingVisualizationMixin):
    """Mock trainer to test mixin directly."""
    def __init__(self, model):
        self.model = model
        self.device = torch.device('cpu')

class TestVisualizationMixin(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel()
        self.trainer = InteractiveTrainer(self.model)

    def test_init_projections(self):
        """Test that projection vectors are initialized and orthogonal."""
        self.trainer._init_random_projections(self.trainer.device)
        
        self.assertTrue(hasattr(self.trainer, 'proj_u'))
        self.assertTrue(hasattr(self.trainer, 'proj_v'))
        
        # Check orthogonality
        dot = torch.dot(self.trainer.proj_u, self.trainer.proj_v)
        self.assertLess(abs(dot.item()), 1e-6)
        
        # Check normalization
        self.assertAlmostEqual(self.trainer.proj_u.norm().item(), 1.0, places=5)
        self.assertAlmostEqual(self.trainer.proj_v.norm().item(), 1.0, places=5)

    def test_trajectory_point(self):
        """Test random projection coordinate calculation."""
        self.trainer._init_random_projections(self.trainer.device)
        point = self.trainer._get_trajectory_point(loss_val=0.5)
        
        self.assertIn('x', point)
        self.assertIn('y', point)
        self.assertIn('loss', point)
        self.assertEqual(point['loss'], 0.5)

    def test_layer_grads(self):
        """Test gradient norm calculation per layer."""
        # Create dummy gradients
        for p in self.model.parameters():
            p.requires_grad = True
            p.grad = torch.ones_like(p)
            
        grads = self.trainer._get_layer_grads()
        
        self.assertTrue(len(grads) > 0)
        self.assertIn('layer', grads[0])
        self.assertIn('norm', grads[0])
        
        # Verify layer names are extracted (SimpleModel has layers, head)
        layer_names = [g['layer'] for g in grads]
        self.assertIn('Layer 0', layer_names)
        self.assertIn('Head', layer_names)

    def test_sft_trainer_inheritance(self):
        """Test that SFTTrainer correctly inherits from mixin."""
        # We don't need to instantiate SFTTrainer fully (which is complex) 
        # to check inheritance and method presence
        self.assertTrue(issubclass(SFTTrainer, TrainingVisualizationMixin))
        self.assertTrue(hasattr(SFTTrainer, '_init_random_projections'))
        self.assertTrue(hasattr(SFTTrainer, '_get_trajectory_point'))
        self.assertTrue(hasattr(SFTTrainer, '_get_layer_grads'))

if __name__ == '__main__':
    unittest.main()
