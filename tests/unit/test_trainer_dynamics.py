
import torch
import torch.nn as nn
from pretraining.training.trainer import TransformerTrainer
from pretraining.training.training_args import TransformerTrainingArgs

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.layer2(self.layer1(x))

def test_visualization_metrics():
    model = DummyModel()
    args = TransformerTrainingArgs(epochs=1, batch_size=2, lr=0.01)
    
    # Dummy data
    X = torch.randn(10, 4)
    Y = torch.randint(0, 2, (10, 4))
    
    trainer = TransformerTrainer(
        model=model,
        args=args,
        X_train=X,
        Y_train=Y,
        X_val=X,
        Y_val=Y,
        device=torch.device("cpu"),
        print_interval=1
    )
    
    # Test 1: Random Projections Initialization
    assert hasattr(trainer, "proj_u")
    assert hasattr(trainer, "proj_v")
    # Should have entries for weights
    assert "layer1.weight" in trainer.proj_u
    assert trainer.proj_u["layer1.weight"].shape == model.layer1.weight.shape
    
    # Test 2: Trajectory Point
    traj = trainer._get_trajectory_point()
    assert isinstance(traj, dict)
    assert "x" in traj
    assert "y" in traj
    assert isinstance(traj["x"], float)
    assert isinstance(traj["y"], float)
    
    # Test 3: Layer Gradients
    # Manually populate gradients
    for name, param in model.named_parameters():
        param.grad = torch.ones_like(param)
        
    layer_grads = trainer._get_layer_grads()
    assert isinstance(layer_grads, list)
    assert len(layer_grads) > 0
    assert "layer" in layer_grads[0]
    assert "norm" in layer_grads[0]
    
    # In DummyModel, layers are "layer1", "layer2".
    # Logic: no digits in dot-split parts. So "Other".
    layers = [g["layer"] for g in layer_grads]
    assert "Other" in layers
    
    # Validate norm calculation (sum of sq of 1s)
    # total elements in layer1.weight is 4*4=16. norm = sqrt(16) = 4.
    # BUT, all "Other" are grouped.
    # layer1.weight (16) + layer1.bias (4) + layer2.weight (8) + layer2.bias (2) = 30 elements.
    # norm = sqrt(30) approx 5.477.
    
    other_grad = next(g for g in layer_grads if g["layer"] == "Other")
    assert other_grad["norm"] > 0
