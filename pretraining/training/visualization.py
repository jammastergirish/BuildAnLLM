import torch

class TrainingVisualizationMixin:
    """Mixin for training visualization metrics (gradients, loss landscape).
    
    Requires the class to have:
    - self.model (nn.Module)
    - self.device (torch.device) - optional if relying on model device, but used for projections
    """

    def _init_random_projections(self, device):
        """Initialize random vectors for 2D loss landscape projection.
        
        We create two random normalized vectors u and v that are orthogonal.
        These are used to project the high-dimensional weight space onto a 2D plane.
        
        Args:
            device: torch.device to store the projections on
        """
        # We use buffers so they are part of state_dict but not parameters
        # For simplicity in this mixin, we'll store them as attributes
        # In a full PyTorch module they should be registered buffers, but here
        # we are mixing into a Trainer class, not a Module.
        
        total_params = sum(p.numel() for p in self.model.parameters())
        self.proj_u = torch.randn(total_params, device=device)
        self.proj_v = torch.randn(total_params, device=device)
        
        # Normalize
        self.proj_u = self.proj_u / (self.proj_u.norm() + 1e-8)
        self.proj_v = self.proj_v / (self.proj_v.norm() + 1e-8)
        
        # Orthogonalize v with respect to u
        self.proj_v = self.proj_v - torch.dot(self.proj_v, self.proj_u) * self.proj_u
        self.proj_v = self.proj_v / (self.proj_v.norm() + 1e-8)

    def _get_trajectory_point(self, loss_val: float = None):
        """Project current weights onto 2D random plane for visualization.
        
        Args:
            loss_val: Optional loss value to attach to the point
            
        Returns:
            dict with x, y, and optional loss
        """
        if not hasattr(self, 'proj_u') or not hasattr(self, 'proj_v'):
            return {"x": 0.0, "y": 0.0, "loss": loss_val}

        # Flatten all parameters
        flat_params = []
        for p in self.model.parameters():
            flat_params.append(p.data.view(-1))
        
        if not flat_params:
             return {"x": 0.0, "y": 0.0, "loss": loss_val}
             
        # Concatenate all params
        # Note: This can be memory intensive for very large models. 
        # For a demo/educational tool with smaller models, this is fine.
        current_w = torch.cat(flat_params)
        
        # Project
        # Ensure current_w is on the same device as projections
        if current_w.device != self.proj_u.device:
            current_w = current_w.to(self.proj_u.device)
            
        x = torch.dot(current_w, self.proj_u).item()
        y = torch.dot(current_w, self.proj_v).item()
        
        result = {"x": x, "y": y}
        if loss_val is not None:
            result["loss"] = loss_val
            
        return result

    def _get_layer_grads(self):
        """Compute gradient norms per layer/block."""
        layer_grads = []
        
        # Helper to get layer name from parameter name
        def get_layer_group(name):
            if "embed" in name or "token" in name or "wte" in name:
                return "Embedding"
            elif "head" in name or "output" in name or "final" in name:
                return "Head"
            elif "norm" in name or "ln" in name:
                return "Norm"
            elif "layers" in name or "blocks" in name or "h." in name:
                # Extract layer index e.g. layers.0.attn, transformer.h.0 -> Layer 0
                search = name.split(".")
                for i, part in enumerate(search):
                    if part in ["layers", "blocks", "h"] and i + 1 < len(search) and search[i+1].isdigit():
                        return f"Layer {search[i+1]}"
            return "Other"

        grads_by_group = {}
        
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                # Only count required grads if we want to filter frozen params?
                # Usually we want to see 0 for frozen params or not show them.
                # Let's show all that have grads (optimizer might set None for frozen)
                
                group = get_layer_group(name)
                grad_norm = p.grad.detach().norm(2).item()
                if group not in grads_by_group:
                    grads_by_group[group] = 0.0
                grads_by_group[group] += grad_norm ** 2
        
        for group, sq_norm in grads_by_group.items():
            layer_grads.append({
                "layer": group,
                "norm": sq_norm ** 0.5
            })
            
        # Sort layers nicely: Embedding -> Layer 0...N -> Norm -> Head
        def sort_key(item):
            name = item["layer"]
            if name == "Embedding": return -2
            if name == "Norm": return 999
            if name == "Head": return 1000
            if name == "Other": return 1001
            if name.startswith("Layer"):
                try:
                    return int(name.split(" ")[1])
                except:
                    return 0
            return 0
            
        return sorted(layer_grads, key=sort_key)
