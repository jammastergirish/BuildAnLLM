"""Unit tests for training utility functions."""

import pytest
import threading
import time
from collections import deque
from unittest.mock import MagicMock, call, ANY

from training_utils import run_training_thread


@pytest.fixture
def mock_trainer():
    """Create a mock trainer."""
    trainer = MagicMock()
    trainer.max_iters = 10
    trainer.eval_interval = 5
    trainer.print_interval = 2
    trainer.running_loss = 0.5
    
    # Mock args
    trainer.args.epochs = 1
    trainer.args.batch_size = 32
    trainer.args.lr = 1e-4
    trainer.args.weight_decay = 0.01
    trainer.args.save_interval = 5
    
    # Mock methods
    trainer.train_single_step.return_value = {
        "loss": 0.5,
        "running_loss": 0.5
    }
    trainer.estimate_loss.return_value = {
        "train": 0.4,
        "val": 0.6
    }
    
    return trainer


@pytest.fixture
def shared_objects():
    """Create shared objects for threading."""
    shared_loss_data = {
        "iterations": [],
        "train_losses": [],
        "val_losses": []
    }
    shared_logs = deque(maxlen=200)
    training_active_flag = [True]
    lock = threading.Lock()
    progress_data = {}
    
    return (shared_loss_data, shared_logs, training_active_flag, 
            lock, progress_data)


@pytest.mark.unit
class TestRunTrainingThread:
    """Tests for run_training_thread function."""

    def test_run_training_complete(self, mock_trainer, shared_objects):
        """Test complete training run."""
        (shared_loss_data, shared_logs, training_active_flag, 
         lock, progress_data) = shared_objects
        
        run_training_thread(
            mock_trainer, shared_loss_data, shared_logs, 
            training_active_flag, lock, progress_data
        )
        
        
        # Verify training completed
        assert training_active_flag[0] is False
        assert progress_data["progress"] == 1.0
        
        # Check logs (convert deque to list for easier searching)
        logs = list(shared_logs)
        assert any("Completed all 10 iterations" in log for log in logs)
        assert any("Training complete" in log for log in logs)
        
        # Verify calls
        assert mock_trainer.train_single_step.call_count == 10
        # Should evaluate at start (iter 0) and iter 5 and end (iter 9) -> actually logic is iter>0 and %eval == 0
        # Logic in code: if (iter_num > 0 and iter_num % eval_interval == 0) or iter_num == max_iters - 1:
        # iter 5 is %5==0. iter 9 is max_iters-1.
        # Wait, inside loop:
        # for iter_num in range(max_iters): (0 to 9)
        #   step()
        #   if (iter > 0 and iter % eval == 0) or iter == max - 1: evaluate()
        # So iter 5 and iter 9. Total 2 evals.
        assert mock_trainer.estimate_loss.call_count == 2
        
        # Checkpoints: if (iter > 0 and iter % save == 0)
        # iter 5.
        # plus final save at end.
        # logic: trainer.save_checkpoint(iter_num) inside loop
        # _finalize_training calls save_checkpoint(max_iters, is_final=True)
        assert mock_trainer.save_checkpoint.call_count == 2
        mock_trainer.save_checkpoint.assert_has_calls([
            call(5),
            call(10, is_final=True)
        ])

    def test_stop_training_early(self, mock_trainer, shared_objects):
        """Test stopping training locally via flag."""
        (shared_loss_data, shared_logs, training_active_flag, 
         lock, progress_data) = shared_objects
        
        # Stop after 3 iterations
        def side_effect():
            if mock_trainer.train_single_step.call_count >= 3:
                training_active_flag[0] = False
            return {"loss": 0.5, "running_loss": 0.5}
            
        mock_trainer.train_single_step.side_effect = side_effect
        
        run_training_thread(
            mock_trainer, shared_loss_data, shared_logs, 
            training_active_flag, lock, progress_data
        )
        
        # Verify stopped early
        assert mock_trainer.train_single_step.call_count == 3
        # Should log stop message
        assert "Training stopped by user." in shared_logs
        # Should not save final checkpoint if stopped by user
        # Note: logic says if training_active_flag[0]: save_final. 
        # Since we set it to False, it shouldn't save final.
        # But it might save intermediate if we hit save interval (we didn't, 3 < 5).
        assert mock_trainer.save_checkpoint.call_count == 0

    def test_error_handling(self, mock_trainer, shared_objects):
        """Test error handling during training."""
        (shared_loss_data, shared_logs, training_active_flag, 
         lock, progress_data) = shared_objects
        
        # Raise error on 2nd step
        mock_trainer.train_single_step.side_effect = [
            {"loss": 0.5, "running_loss": 0.5},
            RuntimeError("Something went wrong")
        ]
        
        run_training_thread(
            mock_trainer, shared_loss_data, shared_logs, 
            training_active_flag, lock, progress_data
        )
        
        # Verify error handling
        assert training_active_flag[0] is False
        assert "ERROR DETECTED" in shared_logs[-7] # approximate position
        assert "Error during training: Something went wrong" in shared_logs[-5]

    def test_progress_updates(self, mock_trainer, shared_objects):
        """Test progress data updates."""
        (shared_loss_data, shared_logs, training_active_flag, 
         lock, progress_data) = shared_objects
        
        # Enable all_losses tracking
        progress_data["all_losses"] = {
            "iterations": [], 
            "current_losses": [], 
            "running_losses": []
        }
        
        run_training_thread(
            mock_trainer, shared_loss_data, shared_logs, 
            training_active_flag, lock, progress_data
        )
        
        # Check updates happen (every 10 iters or last)
        # 0, 9 (last is 9).
        assert len(progress_data["all_losses"]["iterations"]) >= 2
        assert progress_data["iter"] == 9
        assert progress_data["loss"] == 0.5
        assert progress_data["running_loss"] == 0.5
