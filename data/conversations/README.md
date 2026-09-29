# Conversation training data

Use JSONL. Each line is one supervised example.

```json
{"user":"Hello Raksam","assistant":"Hello! What would you like to do?"}
{"user":"Remember our plan","history":[{"role":"user","content":"Let's build a multi-device assistant."},{"role":"assistant","content":"We can design one shared brain and a device bus."}],"assistant":"Yes. The plan is to use one Raksam brain with a device bus."}
```

For high-quality results, create thousands/millions of varied examples rather than repeating templates. Include normal conversation, explanations, planning, clarification, correction, multi-turn context, tool/device decisions, and safe refusal behavior.
