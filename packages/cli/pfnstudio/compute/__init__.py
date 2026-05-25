"""Compute adapters. Each adapter knows how to submit a Run to a backend."""

from .base import ComputeAdapter
from .hf_spaces import HFSpacesAdapter
from .local import LocalAdapter
from .modal import ModalAdapter
from .remote import RemoteAdapter
from .runpod import RunPodAdapter
from .vast import VastAdapter

ADAPTERS: dict[str, type[ComputeAdapter]] = {
    "local": LocalAdapter,
    "vast": VastAdapter,
    "remote": RemoteAdapter,
    "modal": ModalAdapter,
    "runpod": RunPodAdapter,
    "hf_spaces": HFSpacesAdapter,
}


def get_adapter(target: str) -> ComputeAdapter:
    if target not in ADAPTERS:
        raise KeyError(f"Unknown compute target '{target}'. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[target]()


__all__ = ["ADAPTERS", "ComputeAdapter", "get_adapter"]
