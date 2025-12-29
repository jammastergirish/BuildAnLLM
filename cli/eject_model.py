
import ast
import argparse
import inspect
from pathlib import Path
import re

# We will read these files
FILES_TO_EJECT = [
    "config.py",
    "pretraining/model/model.py",
    "pretraining/transformer_blocks/transformer_block.py",
    "pretraining/attention/attention.py",
    "pretraining/mlp/mlp.py",
    "pretraining/normalization/layernorm.py",
    "pretraining/normalization/rmsnorm.py",
    "pretraining/embeddings/embed.py",
    "pretraining/positional_embeddings/positional_embedding.py",
    "pretraining/positional_embeddings/rope.py",
    "pretraining/positional_embeddings/alibi.py",
    "pretraining/utils.py",
]

class Ejector(ast.NodeTransformer):
    def __init__(self, config):
        self.config = config
        self.imports_to_keep = set()
        
    def visit_If(self, node):
        """Prune if statements based on config."""
        # Check condition
        try:
            # We try to eval the condition against our config
            # This is a very simple evaluator for common patterns we use
            cond_source = ast.unparse(node.test)
            
            # Handle standard config checks
            if "cfg.positional_encoding" in cond_source:
                if "PositionalEncoding.LEARNED" in cond_source:
                    keep = self.config["positional_encoding"] == "learned"
                elif "PositionalEncoding.ROPE" in cond_source:
                    keep = self.config["positional_encoding"] == "rope"
                elif "PositionalEncoding.ALIBI" in cond_source:
                    keep = self.config["positional_encoding"] == "alibi"
                else:
                    return self.generic_visit(node)
                
                if keep:
                    return self.visit(node.body) # Return body lines directly (unwrapped)
                elif node.orelse:
                    return self.visit(node.orelse)
                else:
                    return None # Remove entirely
            
            # Handle feature flags
            if "use_einops" in cond_source:
                keep = self.config["use_einops"]
                if "not" in cond_source: keep = not keep
                
                if keep:
                     return self.visit(node.body)
                elif node.orelse:
                     return self.visit(node.orelse)
                else:
                     return None
                     
            if "rope is not None" in cond_source:
                keep = self.config["positional_encoding"] == "rope"
                if keep: return self.visit(node.body) # Keep if check but body is visited? No, standard if.
                # Actually for this one, we usually want to KEEP the if check if the feature is on,
                # but removing the check if the feature is OFF (and the body relies on it) 
                # or just set it to False.
                # Simpler approach: If RoPE is OFF, "rope is not None" is False.
                if not keep:
                     return None if not node.orelse else self.visit(node.orelse)
                     
            if "alibi is not None" in cond_source:
                keep = self.config["positional_encoding"] == "alibi"
                if not keep:
                     return None if not node.orelse else self.visit(node.orelse)
            
            # MoE checks
            if "use_moe" in cond_source: 
                 keep = self.config["use_moe"]
                 if keep: return self.visit(node.body)
                 if not keep: 
                      return None if not node.orelse else self.visit(node.orelse)

        except Exception as e:
            pass # Failed to eval, keep node
            
        return self.generic_visit(node)

def eject_model(architecture="gpt", use_einops=True, use_moe=False):
    """
    Reads the source code and concatenates it into a single file,
    pruning unused branches.
    """
    
    config = {
        "positional_encoding": "learned" if architecture == "gpt" else ("rope" if architecture == "llama" else "alibi"),
        "normalization": "layernorm" if architecture in ["gpt", "olmo"] else "rmsnorm",
        "activation": "gelu" if architecture == "gpt" else "swiglu",
        "use_einops": use_einops,
        "use_moe": use_moe
    }
    
    print(f"Ejecting model with config: {config}")
    
    combined_source = "# Auto-generated Ejected Model\n"
    combined_source += "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\nimport math\nfrom typing import Optional, Tuple, Union, List\nfrom torch import Tensor\n\n"
    if use_einops:
        combined_source += "import einops\n"
        
    # We need to read files in dependency order essentially
    # Or just dump them all and hope python resolves names (it won't for classes valid at def time)
    # Order: utils -> embeddedings/pos/norm -> mlp -> attention -> block -> model
    
    # Actually, we can just grab the ASTs, combine them, and clean up imports
    
    # Naive Concatenation Order
    ORDERED_FILES = [
        "pretraining/utils.py",
        "config.py", 
        "pretraining/normalization/layernorm.py",
        "pretraining/normalization/rmsnorm.py",
        "pretraining/embeddings/embed.py",
        "pretraining/positional_embeddings/positional_embedding.py",
        "pretraining/positional_embeddings/rope.py",
        "pretraining/positional_embeddings/alibi.py",
        "pretraining/mlp/mlp.py",
        "pretraining/attention/attention.py",
        "pretraining/transformer_blocks/transformer_block.py",
        "pretraining/model/model.py",
    ]
    
    ejector = Ejector(config)
    
    for fname in ORDERED_FILES:
        path = Path(fname)
        if not path.exists(): 
             # Try absolute path based on wherever we are
             path = Path("/Users/girish/Documents/Mech Interp/Mine") / fname
             
        with open(path, "r") as f:
            source = f.read()
            
        tree = ast.parse(source)
        
        # Remove imports that we are handling manually (internal ones)
        # Keep external ones
        new_body = []
        for node in tree.body:
             if isinstance(node, (ast.Import, ast.ImportFrom)):
                 # Filter internal imports
                 if isinstance(node, ast.ImportFrom) and node.module and ("pretraining" in node.module or "config" in node.module or "ui_components" in node.module):
                     continue
                 if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                     continue # Remove future imports (must be at top)
                 new_body.append(node)
             else:
                 new_body.append(node)
        tree.body = new_body
        
        # Run pruning
        tree = ejector.visit(tree)
        
        # Unparse
        # In python 3.9+ ast.unparse exists
        try:
             segment = ast.unparse(tree)
             combined_source += f"\n# === Source: {fname} ===\n"
             combined_source += segment + "\n"
        except:
             print(f"Failed to unparse {fname}")

    # Final cleanup regexes
    # Remove jaxtyping annotations from string
    # e.g. Float[Tensor, "batch pos"] -> Tensor
    # simple approach: removing type hints entirely might be cleaner for a "minimal" file
    # or just keep them but ensure jaxtyping import is effectively gone/mocked?
    
    with open("ejected_model.py", "w") as f:
        f.write(combined_source)
    print("Created ejected_model.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="gpt", choices=["gpt", "llama", "olmo"])
    parser.add_argument("--no_einops", action="store_true")
    parser.add_argument("--moe", action="store_true")
    args = parser.parse_args()
    
    eject_model(args.arch, not args.no_einops, args.moe)
