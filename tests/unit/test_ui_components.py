
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os

# Add root directory to path so we can import ui_components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from ui_components import render_attention_heatmap

@patch('plotly.express.imshow')
@patch('ui_components.st')
def test_render_attention_heatmap(mock_st, mock_imshow):
    """Test that attention heatmap is rendered with correct parameters."""
    
    # Setup mock data
    attn_map = np.random.rand(5, 5)
    token_labels = ["A", "B", "C", "D", "E"]
    layer_idx = 0
    head_idx = 0
    
    # Setup mock figure
    mock_fig = MagicMock()
    mock_imshow.return_value = mock_fig
    
    # Call function
    render_attention_heatmap(attn_map, token_labels, layer_idx, head_idx)
    
    # Verify px.imshow called with correct arguments
    # Key check: x and y should be indices [0, 1, 2, 3, 4], not tokens directly
    args, kwargs = mock_imshow.call_args
    assert kwargs['x'] == [0, 1, 2, 3, 4]
    assert kwargs['y'] == [0, 1, 2, 3, 4]
    assert kwargs['title'] == "Layer 0 Head 0 Attention"
    
    # Verify update_xaxes called with correct ticktext (the actual tokens)
    mock_fig.update_xaxes.assert_called_with(
        tickmode='array',
        tickvals=[0, 1, 2, 3, 4],
        ticktext=token_labels,
        tickangle=0 # Should be 0 for short list
    )
    
    # Verify update_yaxes called with correct ticktext
    mock_fig.update_yaxes.assert_called_with(
        tickmode='array',
        tickvals=[0, 1, 2, 3, 4],
        ticktext=token_labels
    )
    
    # Verify streamlit chart display with correct width
    mock_st.plotly_chart.assert_called_with(mock_fig, width='stretch')

@patch('plotly.express.imshow')
@patch('ui_components.st')
def test_render_attention_heatmap_long_sequence(mock_st, mock_imshow):
    """Test that tick angle is adjusted for long sequences."""
    
    # Setup mock data (> 20 tokens)
    token_labels = [f"T{i}" for i in range(25)]
    attn_map = np.random.rand(25, 25)
    
    mock_fig = MagicMock()
    mock_imshow.return_value = mock_fig
    
    # Call function
    render_attention_heatmap(attn_map, token_labels, 0, 0)
    
    # Verify tickangle is 90
    args, kwargs = mock_fig.update_xaxes.call_args
    assert kwargs['tickangle'] == 90
