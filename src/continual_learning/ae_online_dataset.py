"""
Windowed Autoencoder Dataset for Online Training

Reshapes spatio-temporal data into per-spatial-point samples with small
temporal windows, enabling streaming/online training for autoencoders.

Unlike the offline TemporalPointDataset (which concatenates ALL timesteps
into a 1200-dim vector), this dataset provides one window at a time:
each sample is (num_vars * time_seq)-dimensional.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import pyarrow.csv as pv


class WindowedAEDataset(Dataset):
    """
    Dataset that provides per-window slices of spatio-temporal data for AE training.

    Each window contains time_seq consecutive timesteps. For each spatial point,
    the field variables across the window timesteps are flattened into a single
    vector of dimension (time_seq * num_vars).

    Global normalization is computed once across ALL timesteps to ensure
    consistency when replaying past window samples.

    Args:
        filepath (str): Path to CSV file (no header)
        num_windows (int): Number of temporal windows
    """

    def __init__(self, filepath, num_windows=20):
        print("Loading dataset from {}".format(filepath))

        read_options = pv.ReadOptions(
            column_names=['x', 'y', 'z', 't', 'Vx', 'Vy', 'Pressure', 'TKE']
        )
        table = pv.read_csv(filepath, read_options=read_options)
        data = table.to_pandas()

        # Sort by spatial location then time for consistent grouping
        data = data.sort_values(['x', 'y', 'z', 't']).reset_index(drop=True)

        fields = data[['Vx', 'Vy', 'Pressure', 'TKE']].values.astype(np.float32)
        coords = data[['x', 'y', 'z']].values.astype(np.float32)

        # Grid dimensions
        self.num_timesteps = data['t'].nunique()
        self.num_points = len(data) // self.num_timesteps
        self.num_vars = 4
        self.var_names = ['Vx', 'Vy', 'Pressure', 'TKE']
        self.num_windows = num_windows
        self.time_seq = self.num_timesteps // num_windows

        print("Detected {} spatial points x {} timesteps".format(
            self.num_points, self.num_timesteps))
        print("Windows: {}, timesteps per window: {}".format(
            self.num_windows, self.time_seq))

        # Reshape: (num_points * num_timesteps, 4) -> (num_points, num_timesteps, 4)
        self.fields_3d = fields.reshape(self.num_points, self.num_timesteps, self.num_vars)

        # Global normalization across ALL timesteps (critical for replay consistency)
        self.field_min = fields.min(axis=0)  # (4,)
        self.field_max = fields.max(axis=0)  # (4,)
        self.field_range = self.field_max - self.field_min
        self.field_range[self.field_range == 0] = 1.0

        # Normalize all data
        self.fields_3d_norm = (self.fields_3d - self.field_min) / self.field_range

        # Store unique spatial coordinates for visualization
        self.coords = coords.reshape(self.num_points, self.num_timesteps, 3)[:, 0, :]

        # Window input dimension
        self.window_input_dim = self.time_seq * self.num_vars

        # Store unique timestep values
        self.unique_times = np.sort(data['t'].unique())

        print("Dataset ready: {} points, window_input_dim={}".format(
            self.num_points, self.window_input_dim))
        print("Field ranges: {}".format(
            dict(zip(self.var_names,
                     ['{:.4f} to {:.4f}'.format(mn, mx)
                      for mn, mx in zip(self.field_min, self.field_max)]))))

    def get_window_data(self, window_idx):
        """
        Get flattened data for a specific temporal window.

        Args:
            window_idx (int): Window index (0 to num_windows-1)

        Returns:
            torch.FloatTensor: Shape (num_points, time_seq * num_vars)
        """
        start_t = window_idx * self.time_seq
        end_t = start_t + self.time_seq
        if window_idx == self.num_windows - 1:
            end_t = self.num_timesteps

        # Slice: (num_points, time_seq, num_vars)
        window_fields = self.fields_3d_norm[:, start_t:end_t, :]

        # Flatten temporal: (num_points, time_seq * num_vars)
        window_flat = window_fields.reshape(self.num_points, -1)

        return torch.FloatTensor(window_flat)

    def get_all_windows_data(self):
        """
        Get data for all windows concatenated. Used for full-dataset evaluation.

        Returns:
            list of torch.FloatTensor: One tensor per window,
                each shape (num_points, time_seq * num_vars)
        """
        return [self.get_window_data(i) for i in range(self.num_windows)]

    def get_normalization_params(self):
        """Return normalization parameters as a serializable dict."""
        return {
            'field_min': self.field_min.tolist(),
            'field_max': self.field_max.tolist(),
            'field_range': self.field_range.tolist(),
            'num_timesteps': self.num_timesteps,
            'num_points': self.num_points,
            'num_vars': self.num_vars,
            'num_windows': self.num_windows,
            'time_seq': self.time_seq,
            'window_input_dim': self.window_input_dim,
        }

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Use get_window_data(window_idx) for online training"
        )
