"""
Continual Learning Strategies for Online Neural Compression

Implements methods to mitigate catastrophic forgetting in streaming
spatio-temporal data compression. The framework evaluates three
strategies common to all compression architectures (INR, linear
autoencoder, convolutional autoencoder):
    - Naive: baseline sequential training with no replay
    - ExperienceReplay: reservoir-sampled buffer of past windows
      replayed alongside current-window data
"""

from .replay_buffer import ReplayBuffer
from .cl_strategies import (
    NaiveStrategy,
    ExperienceReplayStrategy,
)
