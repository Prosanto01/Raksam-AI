"""Universal device interface for Raksam V3.

A device is a body. An adapter translates Raksam's abstract actions into the
native protocol of that body. Add adapters for phones, robots, cars, machines,
or custom hardware without changing the brain.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Device:
    id: str
    name: str
    kind: str
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

class DeviceAdapter(ABC):
    def __init__(self, device: Device): self.device=device
    @abstractmethod
    def observe(self) -> str: raise NotImplementedError
    @abstractmethod
    def execute(self, action: dict) -> str: raise NotImplementedError
    def describe(self) -> str:
        return f"{self.device.id} ({self.device.kind}): {', '.join(self.device.capabilities)}"
