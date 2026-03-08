"""
Continual Learning Strategies for Online Neural Compression

Implements methods to mitigate catastrophic forgetting in streaming
spatio-temporal data compression using Implicit Neural Representations.

Strategies:
    - Naive: Baseline sequential training (no mitigation)
    - ExperienceReplay: Replay buffer with reservoir sampling
    - EWC: Elastic Weight Consolidation
    - LwF: Learning without Forgetting (knowledge distillation)
    - DERpp: Dark Experience Replay++
    - CombinedStrategy: Replay + Distillation + Regularization hybrid
"""

from .replay_buffer import ReplayBuffer
from .cl_strategies import (
    NaiveStrategy,
    ExperienceReplayStrategy,
    EWCStrategy,
    LwFStrategy,
    DERppStrategy,
    CombinedStrategy,
)
