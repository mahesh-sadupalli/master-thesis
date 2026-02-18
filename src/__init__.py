"""Source package for neural network-based spatio-temporal flow field compression."""

from .models import RegressionModel, AdvancedRegressionModel
from .dataset import FlowDatasetMinMax
from .metrics import calculate_metrics_torch, calculate_relative_error
from .train_offline import train_offline, save_metrics_to_csv
from .train_online import train_online

__all__ = [
    "RegressionModel",
    "AdvancedRegressionModel",
    "FlowDatasetMinMax",
    "calculate_metrics_torch",
    "calculate_relative_error",
    "train_offline",
    "save_metrics_to_csv",
    "train_online",
]
