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
import copy
import numpy as np
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


class EWCStrategy:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    After each window, computes the Fisher Information diagonal to estimate
    parameter importance. Penalizes changes to important parameters during
    subsequent windows.

    Args:
        ewc_lambda (float): Strength of the EWC penalty
        fisher_samples (int): Number of samples for Fisher estimation.
            If None, uses all window samples.
    """

    def __init__(self, ewc_lambda=1000.0, fisher_samples=None):
        self.name = "ewc"
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.fisher_dict = {}      # Parameter name -> Fisher diagonal
        self.old_params_dict = {}  # Parameter name -> old parameter values
        self.tasks_seen = 0

    def before_window(self, model, window_idx, window_inputs, window_targets, device):
        pass

    def compute_loss(self, model, criterion, outputs, targets, window_inputs, device):
        current_loss = criterion(outputs, targets)

        if self.tasks_seen == 0:
            return current_loss

        # EWC penalty
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if name in self.fisher_dict:
                ewc_loss += (
                    self.fisher_dict[name] * (param - self.old_params_dict[name]) ** 2
                ).sum()

        return current_loss + (self.ewc_lambda / 2.0) * ewc_loss

    def after_window(self, model, window_idx, window_inputs, window_targets, device):
        # Compute Fisher Information diagonal
        fisher = self._compute_fisher(model, window_inputs, window_targets, device)

        if self.tasks_seen == 0:
            self.fisher_dict = fisher
        else:
            # Running average of Fisher across windows
            for name in fisher:
                self.fisher_dict[name] = (
                    self.fisher_dict[name] * self.tasks_seen + fisher[name]
                ) / (self.tasks_seen + 1)

        # Store current parameters
        self.old_params_dict = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        self.tasks_seen += 1

    def _compute_fisher(self, model, inputs, targets, device):
        """Compute diagonal Fisher Information using gradients of the loss."""
        fisher = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param)

        model.eval()
        criterion = nn.MSELoss()

        # Subsample if needed
        n = inputs.shape[0]
        if self.fisher_samples and self.fisher_samples < n:
            indices = torch.randperm(n)[: self.fisher_samples]
            inputs_sub = inputs[indices]
            targets_sub = targets[indices]
        else:
            inputs_sub = inputs
            targets_sub = targets

        model.zero_grad()
        outputs = model(inputs_sub)
        loss = criterion(outputs, targets_sub)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] = param.grad.detach() ** 2

        model.zero_grad()
        return fisher

    def get_config(self):
        return {
            "strategy": self.name,
            "ewc_lambda": self.ewc_lambda,
            "fisher_samples": self.fisher_samples,
        }


class LwFStrategy:
    """
    Learning without Forgetting (Li & Hoiem, 2018).

    Before training on a new window, records the current model's predictions
    on the new data (teacher outputs). During training, adds a distillation
    loss that preserves the teacher's behavior.

    Args:
        distill_weight (float): Weight for distillation loss (alpha)
    """

    def __init__(self, distill_weight=0.5):
        self.name = "lwf"
        self.distill_weight = distill_weight
        self.teacher_outputs = None
        self.has_teacher = False

    def before_window(self, model, window_idx, window_inputs, window_targets, device):
        if window_idx > 0:
            # Record current model's predictions on new window data (teacher)
            model.eval()
            with torch.no_grad():
                self.teacher_outputs = model(window_inputs).detach()
            self.has_teacher = True
            model.train()

    def compute_loss(self, model, criterion, outputs, targets, window_inputs, device):
        current_loss = criterion(outputs, targets)

        if not self.has_teacher:
            return current_loss

        # Distillation loss: match teacher's predictions on current inputs
        distill_loss = nn.MSELoss()(outputs, self.teacher_outputs)

        return current_loss + self.distill_weight * distill_loss

    def after_window(self, model, window_idx, window_inputs, window_targets, device):
        pass

    def get_config(self):
        return {
            "strategy": self.name,
            "distill_weight": self.distill_weight,
        }


class DERppStrategy:
    """
    Dark Experience Replay++ (Buzzega et al., 2020).

    Combines experience replay with stored model predictions (logits).
    The buffer stores (input, target, logit) triples. During training,
    applies both task loss on stored samples and distillation loss on
    stored logits.

    Args:
        buffer_size (int): Maximum replay buffer size
        replay_weight (float): Weight for replay task loss (alpha)
        distill_weight (float): Weight for logit distillation loss (beta)
        replay_batch_size (int): Samples to draw from buffer per step
    """

    def __init__(self, buffer_size=10000, replay_weight=0.5,
                 distill_weight=0.5, replay_batch_size=5000):
        self.name = "der_pp"
        self.buffer = ReplayBuffer(max_size=buffer_size, store_logits=True)
        self.replay_weight = replay_weight
        self.distill_weight = distill_weight
        self.replay_batch_size = replay_batch_size

    def before_window(self, model, window_idx, window_inputs, window_targets, device):
        pass

    def compute_loss(self, model, criterion, outputs, targets, window_inputs, device):
        current_loss = criterion(outputs, targets)

        if len(self.buffer) == 0:
            return current_loss

        # Sample from buffer (includes stored logits)
        replay_data = self.buffer.sample(self.replay_batch_size, device=device)
        replay_inputs, replay_targets, stored_logits = replay_data

        replay_outputs = model(replay_inputs)

        # DER++ loss: task loss on replay + distillation loss on stored logits
        replay_task_loss = criterion(replay_outputs, replay_targets)
        distill_loss = nn.MSELoss()(replay_outputs, stored_logits)

        return (
            current_loss
            + self.replay_weight * replay_task_loss
            + self.distill_weight * distill_loss
        )

    def after_window(self, model, window_idx, window_inputs, window_targets, device):
        # Store samples with current model's predictions as logits
        model.eval()
        with torch.no_grad():
            logits = model(window_inputs)
        model.train()
        self.buffer.add_window_batch(window_inputs, window_targets, logits=logits)

    def get_config(self):
        return {
            "strategy": self.name,
            "buffer_size": self.buffer.max_size,
            "replay_weight": self.replay_weight,
            "distill_weight": self.distill_weight,
            "replay_batch_size": self.replay_batch_size,
        }


class CombinedStrategy:
    """
    Hybrid: Experience Replay + Knowledge Distillation + EWC Regularization.

    Combines three complementary approaches:
    1. Replay buffer provides data-level memory
    2. Self-distillation provides output-level consistency
    3. EWC provides parameter-level stability

    Args:
        buffer_size (int): Maximum replay buffer size
        replay_weight (float): Weight for replay loss
        distill_weight (float): Weight for distillation loss
        ewc_lambda (float): Strength of EWC penalty
        replay_batch_size (int): Samples to draw from buffer per step
    """

    def __init__(self, buffer_size=10000, replay_weight=0.3,
                 distill_weight=0.3, ewc_lambda=500.0, replay_batch_size=5000):
        self.name = "combined"
        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.replay_weight = replay_weight
        self.distill_weight = distill_weight
        self.ewc_lambda = ewc_lambda
        self.replay_batch_size = replay_batch_size

        self.teacher_outputs = None
        self.has_teacher = False
        self.fisher_dict = {}
        self.old_params_dict = {}
        self.tasks_seen = 0

    def before_window(self, model, window_idx, window_inputs, window_targets, device):
        if window_idx > 0:
            # Record teacher predictions for distillation
            model.eval()
            with torch.no_grad():
                self.teacher_outputs = model(window_inputs).detach()
            self.has_teacher = True
            model.train()

    def compute_loss(self, model, criterion, outputs, targets, window_inputs, device):
        total_loss = criterion(outputs, targets)

        # Replay loss
        if len(self.buffer) > 0:
            replay_data = self.buffer.sample(self.replay_batch_size, device=device)
            replay_inputs, replay_targets = replay_data
            replay_outputs = model(replay_inputs)
            total_loss = total_loss + self.replay_weight * criterion(
                replay_outputs, replay_targets
            )

        # Distillation loss
        if self.has_teacher:
            distill_loss = nn.MSELoss()(outputs, self.teacher_outputs)
            total_loss = total_loss + self.distill_weight * distill_loss

        # EWC penalty
        if self.tasks_seen > 0:
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if name in self.fisher_dict:
                    ewc_loss += (
                        self.fisher_dict[name]
                        * (param - self.old_params_dict[name]) ** 2
                    ).sum()
            total_loss = total_loss + (self.ewc_lambda / 2.0) * ewc_loss

        return total_loss

    def after_window(self, model, window_idx, window_inputs, window_targets, device):
        # Update replay buffer
        self.buffer.add_window_batch(window_inputs, window_targets)

        # Compute and accumulate Fisher
        fisher = self._compute_fisher(model, window_inputs, window_targets, device)
        if self.tasks_seen == 0:
            self.fisher_dict = fisher
        else:
            for name in fisher:
                self.fisher_dict[name] = (
                    self.fisher_dict[name] * self.tasks_seen + fisher[name]
                ) / (self.tasks_seen + 1)

        self.old_params_dict = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }
        self.tasks_seen += 1

    def _compute_fisher(self, model, inputs, targets, device):
        fisher = {}
        for name, param in model.named_parameters():
            fisher[name] = torch.zeros_like(param)

        model.eval()
        criterion = nn.MSELoss()
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] = param.grad.detach() ** 2

        model.zero_grad()
        return fisher

    def get_config(self):
        return {
            "strategy": self.name,
            "buffer_size": self.buffer.max_size,
            "replay_weight": self.replay_weight,
            "distill_weight": self.distill_weight,
            "ewc_lambda": self.ewc_lambda,
            "replay_batch_size": self.replay_batch_size,
        }
