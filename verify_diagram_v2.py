
import sys
import os

# Add parent dir to path
sys.path.append(os.getcwd())

from ui_components import generate_graphviz_architecture

config = {
    "n_layers": 2,
    "n_heads": 4,
    "d_model": 64,
    "positional_encoding": "learned",
    "activation": "gelu"
}

try:
    dot_code = generate_graphviz_architecture(config)
    print("Successfully generated DOT code:")
    print(dot_code[:200] + "...") # Print first 200 chars
except Exception as e:
    print(f"Error generating diagram: {e}")
    sys.exit(1)
