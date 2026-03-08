"""
Replay Buffer with Reservoir Sampling

Maintains a fixed-size buffer of past samples using reservoir sampling
to ensure uniform temporal coverage across all observed windows.
Optionally stores model predictions (logits) for Dark Experience Replay.
"""

import torch
import numpy as np


class ReplayBuffer:
    """
    Fixed-size replay buffer using reservoir sampling.

    Ensures each past sample has equal probability of being retained,
    providing uniform coverage across all temporal windows seen so far.

    Args:
        max_size (int): Maximum number of samples to store
        store_logits (bool): If True, also store model predictions for DER++
    """

    def __init__(self, max_size=10000, store_logits=False):
        self.max_size = max_size
        self.store_logits = store_logits
        self.inputs = None
        self.targets = None
        self.logits = None
        self.count = 0          # Total samples seen (for reservoir sampling)
        self.current_size = 0   # Current buffer occupancy

    def add_window(self, inputs, targets, logits=None):
        """
        Add samples from a temporal window to the buffer using reservoir sampling.

        Args:
            inputs (Tensor): Input coordinates, shape (N, 4)
            targets (Tensor): Target values, shape (N, 4)
            logits (Tensor, optional): Model predictions at time of storage, shape (N, 4)
        """
        inputs_cpu = inputs.detach().cpu()
        targets_cpu = targets.detach().cpu()
        logits_cpu = logits.detach().cpu() if logits is not None else None

        n_samples = inputs_cpu.shape[0]

        # Initialize buffer on first call
        if self.inputs is None:
            n_init = min(n_samples, self.max_size)
            self.inputs = inputs_cpu[:n_init].clone()
            self.targets = targets_cpu[:n_init].clone()
            if self.store_logits and logits_cpu is not None:
                self.logits = logits_cpu[:n_init].clone()
            self.current_size = n_init
            self.count = n_init
            start_idx = n_init
        else:
            start_idx = 0

        # Reservoir sampling for remaining samples
        for i in range(start_idx, n_samples):
            self.count += 1
            if self.current_size < self.max_size:
                # Buffer not full yet — append
                self.inputs = torch.cat([self.inputs, inputs_cpu[i:i+1]], dim=0)
                self.targets = torch.cat([self.targets, targets_cpu[i:i+1]], dim=0)
                if self.store_logits and logits_cpu is not None:
                    self.logits = torch.cat([self.logits, logits_cpu[i:i+1]], dim=0)
                self.current_size += 1
            else:
                # Replace with probability max_size / count
                j = np.random.randint(0, self.count)
                if j < self.max_size:
                    self.inputs[j] = inputs_cpu[i]
                    self.targets[j] = targets_cpu[i]
                    if self.store_logits and logits_cpu is not None:
                        self.logits[j] = logits_cpu[i]

    def add_window_batch(self, inputs, targets, logits=None, n_samples=None):
        """
        Efficiently add a random subset of window samples to the buffer.

        Instead of iterating over all samples (slow for large windows),
        randomly selects n_samples from the window and uses reservoir sampling.

        Args:
            inputs (Tensor): Input coordinates, shape (N, 4)
            targets (Tensor): Target values, shape (N, 4)
            logits (Tensor, optional): Model predictions, shape (N, 4)
            n_samples (int, optional): Number of samples to consider from window.
                Defaults to min(window_size, max_size // 2).
        """
        window_size = inputs.shape[0]
        if n_samples is None:
            n_samples = min(window_size, self.max_size // 2)

        # Random subset from current window
        indices = torch.randperm(window_size)[:n_samples]
        sub_inputs = inputs[indices]
        sub_targets = targets[indices]
        sub_logits = logits[indices] if logits is not None else None

        self.add_window(sub_inputs, sub_targets, sub_logits)

    def sample(self, batch_size, device=None):
        """
        Sample a random batch from the buffer.

        Args:
            batch_size (int): Number of samples to return
            device: PyTorch device to move samples to

        Returns:
            tuple: (inputs, targets) or (inputs, targets, logits) if store_logits
        """
        if self.current_size == 0:
            return None

        actual_batch = min(batch_size, self.current_size)
        indices = torch.randperm(self.current_size)[:actual_batch]

        batch_inputs = self.inputs[indices]
        batch_targets = self.targets[indices]

        if device is not None:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

        if self.store_logits and self.logits is not None:
            batch_logits = self.logits[indices]
            if device is not None:
                batch_logits = batch_logits.to(device)
            return batch_inputs, batch_targets, batch_logits

        return batch_inputs, batch_targets

    def __len__(self):
        return self.current_size

    def memory_usage_kb(self):
        """Return approximate memory usage of the buffer in KB."""
        if self.inputs is None:
            return 0
        floats = self.current_size * (self.inputs.shape[1] + self.targets.shape[1])
        if self.store_logits and self.logits is not None:
            floats += self.current_size * self.logits.shape[1]
        return floats * 4 / 1024  # 4 bytes per float32
