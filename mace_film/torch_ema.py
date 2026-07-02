"""Minimal local fallback for torch_ema.

This keeps the local MACE checkout importable in environments where the
`torch_ema` package is not installed. It only implements the subset used by
`mace.tools.train`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Iterator, List, Optional, Tuple

import torch


class ExponentialMovingAverage:
    def __init__(self, parameters: Iterable[torch.nn.Parameter], decay: float = 0.999):
        self.decay = float(decay)
        self._parameters = [p for p in parameters if p is not None]
        self._shadow_params = [p.detach().clone() for p in self._parameters]
        self._backup_params: Optional[List[torch.Tensor]] = None

    def update(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None:
        params = self._parameters if parameters is None else [p for p in parameters if p is not None]
        for shadow, param in zip(self._shadow_params, params):
            shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def copy_to(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None:
        params = self._parameters if parameters is None else [p for p in parameters if p is not None]
        for shadow, param in zip(self._shadow_params, params):
            param.data.copy_(shadow.data)

    def store(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None:
        params = self._parameters if parameters is None else [p for p in parameters if p is not None]
        self._backup_params = [p.detach().clone() for p in params]

    def restore(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> None:
        if self._backup_params is None:
            return
        params = self._parameters if parameters is None else [p for p in parameters if p is not None]
        for backup, param in zip(self._backup_params, params):
            param.data.copy_(backup.data)
        self._backup_params = None

    @contextmanager
    def average_parameters(self, parameters: Optional[Iterable[torch.nn.Parameter]] = None) -> Iterator[None]:
        self.store(parameters)
        self.copy_to(parameters)
        try:
            yield
        finally:
            self.restore(parameters)
