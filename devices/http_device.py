"""Generic device bridge: any user-owned hardware that exposes observe/action JSON."""
from __future__ import annotations
import json
from urllib.request import Request,urlopen
from devices.base import Device,DeviceAdapter
class HTTPDeviceAdapter(DeviceAdapter):
    def __init__(self,device):
        super().__init__(device); self.url=device.metadata["url"]; self.state_url=device.metadata.get("state_url"); self.timeout=float(device.metadata.get("timeout",10)); self.headers={"Content-Type":"application/json",**device.metadata.get("headers",{})}
    def _request(self,url,method="GET",body=None):
        req=Request(url,data=(json.dumps(body,ensure_ascii=False).encode() if body is not None else None),headers=self.headers,method=method)
        with urlopen(req,timeout=self.timeout) as r: return r.read().decode("utf-8","replace")
    def observe(self): return self._request(self.state_url) if self.state_url else super().describe()
    def execute(self,action): return self._request(self.url,"POST",action) or "device accepted action"
