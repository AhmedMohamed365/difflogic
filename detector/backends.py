"""
Backend wrappers for difflogic classification models.

Provides a unified ``predict(batch_inputs) -> batch_probs`` interface for
both PyTorch models and compiled CompiledLogicNet models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    pass


class TorchBackend:
    """Wraps a PyTorch difflogic (or any torch.nn.Module) classifier.

    The model must be in ``eval()`` mode and accept float32 tensors of shape
    ``(N, input_dim)`` where ``input_dim`` is the flattened input size.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` in eval mode.
    device:
        Device string such as ``"cpu"`` or ``"cuda"``.
    """

    def __init__(self, model: "torch.nn.Module", device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, batch_inputs: np.ndarray) -> np.ndarray:
        """Run inference on a batch of pre-processed inputs.

        Parameters
        ----------
        batch_inputs:
            Float32 numpy array of shape ``(N, input_dim)``.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(N, num_classes)`` with class scores.
        """
        x = torch.from_numpy(batch_inputs).float().to(self.device)
        with torch.no_grad():
            out = self.model(x)
        if isinstance(out, torch.Tensor):
            return out.cpu().numpy().astype(np.float32)
        return np.array(out, dtype=np.float32)


class CompiledBackend:
    """Wraps a ``CompiledLogicNet`` for fast CPU inference.

    Parameters
    ----------
    compiled_net:
        A compiled ``CompiledLogicNet`` instance (already compiled via
        ``compiled_net.compile(...)``).
    """

    def __init__(self, compiled_net):
        self.compiled_net = compiled_net

    def predict(self, batch_inputs: np.ndarray) -> np.ndarray:
        """Run inference on a batch of Boolean inputs.

        Parameters
        ----------
        batch_inputs:
            Boolean (or float that will be rounded/cast to bool) numpy array
            of shape ``(N, input_dim)``.

        Returns
        -------
        np.ndarray
            Float32 array of shape ``(N, num_classes)`` with class scores.
        """
        x = batch_inputs.astype(np.bool_)
        out = self.compiled_net(x)
        if isinstance(out, torch.Tensor):
            return out.cpu().numpy().astype(np.float32)
        return np.array(out, dtype=np.float32)
