"""PyTorch Dataset for spatio-temporal flow field data with min-max normalization."""

import torch
import numpy as np
from torch.utils.data import Dataset
import pyarrow.csv as pv


class FlowDatasetMinMax(Dataset):
    """
    PyTorch Dataset for spatio-temporal flow field data.
    Applies min-max normalization to map all features to [0, 1] range.

    Args:
        filepath: Path to CSV file containing flow field data
    """
    def __init__(self, filepath):
        print(f"Loading dataset from {filepath}")

        read_options = pv.ReadOptions(
            column_names=['x', 'y', 'z', 't', 'Vx', 'Vy', 'Pressure', 'TKE']
        )
        table = pv.read_csv(filepath, read_options=read_options)
        data = table.to_pandas().values

        self.inputs = data[:, :4].astype(np.float32)
        self.targets = data[:, 4:].astype(np.float32)

        print("Applying min-max normalization to [0, 1] range")

        self.input_min = self.inputs.min(axis=0)
        self.input_max = self.inputs.max(axis=0)
        self.input_range = self.input_max - self.input_min
        self.input_range[self.input_range == 0] = 1.0
        self.inputs = (self.inputs - self.input_min) / self.input_range

        self.target_min = self.targets.min(axis=0)
        self.target_max = self.targets.max(axis=0)
        self.target_range = self.target_max - self.target_min
        self.target_range[self.target_range == 0] = 1.0
        self.targets = (self.targets - self.target_min) / self.target_range

        print(f"Dataset loaded: {len(self)} samples")
        print(f"Input shape: {self.inputs.shape}, Target shape: {self.targets.shape}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.from_numpy(self.inputs[idx]), torch.from_numpy(self.targets[idx])

    def denormalize_input(self, normalized):
        """Convert normalized input coordinates back to original space"""
        if isinstance(normalized, torch.Tensor):
            normalized = normalized.cpu().numpy()
        return normalized * self.input_range + self.input_min

    def denormalize_target(self, normalized):
        """Convert normalized target variables back to original space"""
        if isinstance(normalized, torch.Tensor):
            normalized = normalized.cpu().numpy()
        return normalized * self.target_range + self.target_min
