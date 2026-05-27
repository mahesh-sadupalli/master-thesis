"""
Continual Learning Strategies for Online Neural Compression

Each strategy modifies the training loop to mitigate catastrophic forgetting.
All strategies follow the same interface:
    - before_window(): Called before training on a new window
    - compute_loss(): Returns the total loss for a training step
    - after_window(): Called after training on a window completes
"""

import torch
import torch.nn as nn
from .replay_buffer import ReplayBuffer


class NaiveStrategy:
    """
    Baseline: naive sequential training with no forgetting mitigation.
    Equivalent to the original online training implementation.
    """

    def __init__(self):
        self.name = "naive"

    def before_window(self, model, window_idx, window_inputs, window_targets, device):
        pass

    def compute_loss(self, model, criterion, outputs, targets, window_inputs, device):
        return criterion(outputs, targets)

    def after_window(self, model, window_idx, window_inputs, window_targets, device):
        pass

    def get_config(self):
        return {"strategy": self.name}


class ExperienceReplayStrategy:
    """
    Experience Replay with reservoir sampling.

    Maintains a buffer of past samples and replays them alongside
    new window data during training.

    Args:
        buffer_size (int): Maximum replay buffer size
        replay_weight (float): Weight for replay loss (alpha)
        replay_batch_size (int): Samples to draw from buffer per step
    """

    def __init__(self, buffer_size=10000, replay_weight=0.5, replay_batch_size=5000):
        self.name = "experience_replay"
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.replay_weight = replay_weight
        self.replay_batch_size = replay_batch_size

    def before_window(self, model, window_idx, window_inputs, window_targets, device):
        pass

    def compute_loss(self, model, criterion, outputs, targets, window_inputs, device):
        current_loss = criterion(outputs, targets)

        if len(self.buffer) == 0:
            return current_loss

        # Sample from replay buffer
        replay_data = self.buffer.sample(self.replay_batch_size, device=device)
        replay_inputs, replay_targets = replay_data
        replay_outputs = model(replay_inputs)
        replay_loss = criterion(replay_outputs, replay_targets)

        return current_loss + self.replay_weight * replay_loss

    def after_window(self, model, window_idx, window_inputs, window_targets, device):
        self.buffer.add_window_batch(window_inputs, window_targets)

    def get_config(self):
        return {
            "strategy": self.name,
            "buffer_size": self.buffer.max_size,
            "replay_weight": self.replay_weight,
            "replay_batch_size": self.replay_batch_size,
        }


