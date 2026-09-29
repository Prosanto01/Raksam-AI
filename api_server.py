"""Minimal Raksam V3 API gateway.

The same RaksamAgent used by chat.py can be called over HTTP. This server is
intended as the API foundation for the future public website/developer API.
For internet deployment, put it behind HTTPS, authentication, rate limiting,
isolated device sessions, logging, and a production gateway.
"""
from __future__ import annotations
import json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from agent_core import RaksamAgent

HOST=os.getenv("RAKSAM_API_HOST","127.0.0.1")
PORT=int(os.getenv("RAKSAM_API_PORT","8765"))
API_KEY=os.getenv("RAKSAM_API_KEY","")
AGENT=RaksamAgent()

class Handler(BaseHTTPRequestHandler):
    server_version="RaksamV3API/1.0"
    def _json(self,status,payload):
        raw=json.dumps(payload,ensure_ascii=False).encode("utf-8"); self.send_response(status); self.send_header("Content-Type","application/json; charset=utf-8"); self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def _authorized(self):
        if not API_KEY:return True
        return self.headers.get("Authorization","")==f"Bearer {API_KEY}"
    def do_GET(self):
        if self.path=="/health": return self._json(200,{"ok":True,"model":"Raksam Brain V3 (RB3)","created_by":"Prosanto Raksam"})
        if self.path=="/v1/devices":
            if not self._authorized():return self._json(401,{"error":"unauthorized"})
            return self._json(200,{"devices":AGENT.devices.as_dict()})
        return self._json(404,{"error":"not found"})
    def do_POST(self):
        if not self._authorized():return self._json(401,{"error":"unauthorized"})
        if self.path not in {"/v1/chat","/v1/agent/run"}:return self._json(404,{"error":"not found"})
        try:
            n=int(self.headers.get("Content-Length","0")); body=json.loads(self.rfile.read(n) or b"{}")
            message=str(body.get("message",body.get("goal",""))).strip()
            if not message:return self._json(400,{"error":"message/goal is required"})
            dry=bool(body.get("dry_run",False))
            result=AGENT.chat(message,dry_run=dry) if self.path=="/v1/chat" else AGENT.chat(message,dry_run=dry)
            return self._json(200,{"model":"Raksam Brain V3 (RB3)","result":result})
        except Exception as e:return self._json(500,{"error":str(e)})
    def log_message(self,fmt,*args): pass

def main():
    print(f"Raksam API listening on http://{HOST}:{PORT}")
    ThreadingHTTPServer((HOST,PORT),Handler).serve_forever()
if __name__=="__main__": main()
