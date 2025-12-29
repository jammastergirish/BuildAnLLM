
import pytest
import numpy as np
import plotly.graph_objects as go
from unittest.mock import MagicMock, patch
import sys
import os

# Add root directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ui_components import render_attention_heatmap

@pytest.mark.property
class TestUIProperties:
    """Property-based tests for UI components."""

    @patch('plotly.express.imshow')
    @patch('ui_components.st')
    def test_heatmap_axis_continuity(self, mock_st, mock_imshow):
        """Property: Heatmap axes must map 1:1 with token labels for any sequence length."""
        
        # Test property for random sequence lengths
        for seq_len in [1, 5, 10, 50, 100]:
            # Generate random inputs
            attn_map = np.random.rand(seq_len, seq_len)
            token_labels = [f"T{i}" for i in range(seq_len)]
            
            # Setup mock
            mock_fig = MagicMock()
            mock_imshow.return_value = mock_fig
            
            # Call function
            render_attention_heatmap(attn_map, token_labels, 0, 0)
            
            # Verify Invariant: x and y axis indices must match token count exactly
            args, kwargs = mock_imshow.call_args
            
            # Indices passed to plotly should be [0..seq_len-1]
            expected_indices = list(range(seq_len))
            assert kwargs['x'] == expected_indices, f"Mismatch for seq_len={seq_len}"
            assert kwargs['y'] == expected_indices, f"Mismatch for seq_len={seq_len}"
            
            # Tick values must match indices
            # Check update_xaxes
            x_call_args = mock_fig.update_xaxes.call_args[1]
            assert x_call_args['tickvals'] == expected_indices
            assert len(x_call_args['ticktext']) == seq_len
            assert x_call_args['ticktext'] == token_labels
            
            # Check update_yaxes
            y_call_args = mock_fig.update_yaxes.call_args[1]
            assert y_call_args['tickvals'] == expected_indices
            assert len(y_call_args['ticktext']) == seq_len
            assert y_call_args['ticktext'] == token_labels

    @patch('plotly.express.imshow')
    @patch('ui_components.st')
    def test_heatmap_integrity_with_duplicates(self, mock_st, mock_imshow):
        """Property: Heatmap axes should not collapse even with duplicate tokens."""
        
        # Scenario: All tokens are identical strings
        seq_len = 10
        token_labels = ["dup"] * seq_len # ["dup", "dup", ...]
        attn_map = np.random.rand(seq_len, seq_len)
        
        mock_fig = MagicMock()
        mock_imshow.return_value = mock_fig
        
        render_attention_heatmap(attn_map, token_labels, 0, 0)
        
        # If we used tokens as x/y keys directly, Plotly might collapse them.
        # We verify we are using unique integer indices (0..9)
        _, kwargs = mock_imshow.call_args
        assert kwargs['x'] == list(range(seq_len))
        assert len(kwargs['x']) == seq_len 
        
        # Verify tick labels preserve the duplicates
        x_call_args = mock_fig.update_xaxes.call_args[1]
        assert x_call_args['ticktext'] == token_labels
        assert len(x_call_args['ticktext']) == seq_len
