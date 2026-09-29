"""Universal Raksam device bus."""
from __future__ import annotations
from dataclasses import asdict
import importlib, yaml
from devices.base import Device
from devices.pc import PCAdapter
from devices.android_adb import AndroidADBAdapter
from devices.http_device import HTTPDeviceAdapter

class DeviceRegistry:
    def __init__(self, config_path="config/devices.yaml"):
        with open(config_path, encoding="utf-8") as f: cfg=yaml.safe_load(f) or {}
        self.devices={}; self.adapters={}
        for raw in cfg.get("devices", []):
            d=Device(id=raw["id"],name=raw.get("name",raw["id"]),kind=raw["kind"],capabilities=raw.get("capabilities",[]),metadata=raw.get("metadata",{}))
            self.add(d)
    def add(self, device:Device):
        if device.kind == "pc": adapter=PCAdapter(device)
        elif device.kind == "android_adb": adapter=AndroidADBAdapter(device)
        elif device.kind == "http": adapter=HTTPDeviceAdapter(device)
        else:
            raise ValueError(f"No adapter registered for device kind {device.kind!r}. Add a DeviceAdapter plugin.")
        self.devices[device.id]=device; self.adapters[device.id]=adapter
    def get(self, device_id="this_pc"):
        if device_id not in self.adapters: raise KeyError(f"Unknown device: {device_id}")
        return self.adapters[device_id]
    def describe(self):
        return "\n".join(f"- {d.id}: {d.name} [{d.kind}] capabilities={', '.join(d.capabilities)}" for d in self.devices.values()) or "No devices configured."
    def as_dict(self): return {k:asdict(v) for k,v in self.devices.items()}
