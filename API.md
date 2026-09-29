# Raksam V3 API foundation

`api_server.py` exposes the same Raksam agent used by the console.

Endpoints:
- `GET /health`
- `GET /v1/devices`
- `POST /v1/chat`
- `POST /v1/agent/run`

Example request body:
```json
{"message":"Open Notepad and write Hello"}
```

For local development:
```powershell
$env:RAKSAM_API_KEY="change-this"
py -3.13 api_server.py
```

For a public service, deploy behind HTTPS and a production gateway with
per-user authentication/API keys, rate limits, quotas, isolated sessions,
observability, and device ownership controls. The API layer is deliberately
separate from the brain so the future website and developer API can use the
same Raksam engine.
