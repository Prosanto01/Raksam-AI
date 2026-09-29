from __future__ import annotations
from devices.base import Device, DeviceAdapter
from perception.screen_state import perceive
from actuation.executor import execute as execute_pc
from eyes.device_scan import scan_device, observation_prompt

class PCAdapter(DeviceAdapter):
    def observe(self):
        return observation_prompt(scan_device(self.device.id, self.device.kind))
    def execute(self, action): return execute_pc(action)
