# Raksam Own Brain: conversation + actions + devices

The runtime has no external AI backend. Raksam's transformer is the trainable intelligence; adapters are deterministic bridges to devices.

## 1. Conversation data

Put JSONL files in `data/conversations/`:

```json
{"user":"Hello","assistant":"Hello! How can I help?"}
{"user":"What did we just discuss?","assistant":"We were discussing the project architecture.","history":[{"role":"user","content":"Let's redesign Raksam."},{"role":"assistant","content":"Sure."}]}
```

Train a wide range of styles, factual explanations, reasoning traces appropriate for your design, clarification behavior, refusal behavior, planning, and multi-turn dialogue.

## 2. Device/action data

Action examples can include a device:

```json
{"prompt_text":"Open the calculator on my PC","device":"this_pc","action":{"device":"this_pc","action":"launch_app","target":"calc.exe","reason":"user requested calculator","risk":"medium"}}
{"prompt_text":"Tap Send on my Android phone","device":"my_android","action":{"device":"my_android","action":"android_tap","x":540,"y":960,"reason":"tap Send","risk":"medium"}}
```

The same model therefore learns:

- conversation → natural language
- task + device state → structured action
- action result → next decision

## 3. Do not train secrets

Never include passwords, API keys, private keys, authentication cookies, payment information, or other credentials in training data.

## 4. Demonstrations

The existing PC recorder can generate screen/action demonstrations. For new device types, create equivalent demonstration collectors that record:

`goal + device + capabilities + state-before + action + result-after`

The more varied the demonstrations, the less brittle the learned policy becomes.

## 5. Scaling strategy

The current 12-layer/12-head/384-dimension model is only a starting point. A serious independent model should be grown in stages:

1. Clean and balance conversation/action/device datasets.
2. Train tokenizer vocabulary for all target languages and action schemas.
3. Pretrain on a very large general text corpus you have the rights to use.
4. Fine-tune on Raksam conversation, reasoning, tool-use and device data.
5. Add preference/evaluation data and automated regression tests.
6. Add longer context and larger models only when data/compute justify them.
7. Continuously evaluate hallucination, action accuracy, recovery from errors, and safety.

The runtime architecture is designed so these training stages can replace the current small dataset without replacing the device layer.
