# Raksam AI 🇧🇩

**The first AI agent from Bangladesh with a complete perception → reasoning → action loop.**

Raksam AI doesn't just chat. It **sees the screen, understands your command, controls the mouse and keyboard, and learns continuously** from your feedback — across PC, Android, and generic HTTP devices.

> To the best of my knowledge, this is the first agent system from Bangladesh that combines conversational ability with real screen perception and device control. If you know of an earlier project, please open an Issue.

---

## ✨ What Raksam AI Does

| Capability | Description |
|-----------|-------------|
| 🗣️ **Conversation** | Text and voice chat |
| 👁️ **Perception** | Reads screen and UI state (UI Automation + OCR) |
| 🧠 **Reasoning** | Custom 52M parameter Transformer brain (RB3) |
| 🖐️ **Action** | Controls mouse, keyboard, and applications |
| 🔌 **Multi-device** | PC · Android (adb) · Generic HTTP adapters |
| 🔁 **Continual learning** | Learns from user-approved feedback |
| 🌐 **Web learning** | Collects bounded batches from allowed public sources |
| 🎥 **Demonstration recording** | Records user demos as training data |

---

## ⚙️ How It Works

Raksam AI runs a continuous **observe → think → act → verify** loop. Each step is bounded, inspectable, and logged.

### 1. Task intake
You give Raksam a task through any interface — `voice_chat.py` (voice), `chat.py` (text), `main.py` (CLI), or `api_server.py` (HTTP). All four call into `RaksamAgent` in `agent_core.py`, which owns the task loop.

### 2. Observation (read-only)
The configured adapter in `devices/` gathers the current device state. On Windows, `perception/` uses **UI Automation + OCR** to identify the active window and visible controls. Observation is strictly read-only — no action happens at this step.

### 3. Reasoning
`brain/` contains Raksam's own trained neural model and tokenizer. The agent passes it:
- the user's task
- the current observation

…and the model returns a **structured action** (e.g. `click(x, y)`, `type("hello")`, `open("notepad")`).

### 4. Actuation
The device adapter executes the action. The PC adapter translates structured actions into mouse and keyboard input via `hands/`. Android and generic HTTP adapters are available as alternative device targets.

### 5. Verification
`agent/loop.py` repeats the observe–think–act–verify cycle up to a configured step limit. After each action, the agent re-observes and checks whether the action succeeded. If it cannot be verified, the agent either tries a different step or stops.

### 6. Experience logging
Every action and its result is saved in `data/experiences/`. Conversational history can be saved in `data/memory/`. In voice mode, after a task you can approve or reject the outcome — **only user-approved examples** are eligible for later training.

### 7. Training (separate from runtime)
`brain/train.py` trains a candidate model **offline**, using:
- configured conversations
- recorded demonstrations
- collected web material
- eligible, user-approved experiences

**Model weights are never updated while the agent is controlling the device.** This separation prevents runaway feedback loops.

### Loop diagram

```
   ┌──────────────────────────────────────────────┐
   │                                              │
   ▼                                              │
[ Observe ]──►[ Think ]──►[ Act ]──►[ Verify ]───┤
  devices/      brain/     hands/      agent/     │
  perception/             actuation/   loop.py   │
                                                  │
                                    ▲             │
                                    │             │
                             stop if verified ────┘
                             or step limit reached
```

---

## 🏗️ Architecture

Raksam AI is built around **one brain, many bodies**.

The brain is a single **52M parameter causal Transformer** that receives explicitly formatted conversation, task/action, approved knowledge, and reflection inputs. Device and perception modules produce safe serialized observations.

| Component | Role |
|-----------|------|
| `brain/` | RB3 Transformer model, tokenizer, inference, training |
| `agent/` | Task loop, action contract, grounding, verification |
| `perception/` | Screen and UI observation (UI Automation + OCR) |
| `eyes/` | Camera and device discovery |
| `hands/` | Mouse and keyboard control |
| `actuation/` | Action execution |
| `devices/` | PC · Android · generic HTTP adapters |
| `voice/` | Windows speech input and output |
| `memory.py` | Lightweight persistent memory |
| `continual_learning.py` | Replay data + candidate training cycles |

### RB3 Verified Specifications

| Item | Value |
|------|-------|
| Parameters | **52,460,544** trainable |
| Configuration | 8 layers · 768 hidden · 12 query heads · 4 KV heads |
| Activation | 2.8× SwiGLU |
| Context | 128 tokens |
| Embeddings | Tied |
| Tokenizer | 259 symbols |
| Unified training records | 4,608 |
| Training device | CPU (4 threads) |
| Smoke-run train loss | 6.1076 |
| Smoke-run validation loss | 4.7488 |

Full report: [`52M_TRAINING_REPORT.md`](52M_TRAINING_REPORT.md)

---

## 📁 Project Layout

```
Raksam-AI/
├── chat.py                  # Interactive text chat
├── voice_chat.py            # Voice listen → task → feedback loop
├── main.py                  # CLI single-task runner
├── api_server.py            # Local HTTP API
├── agent_core.py            # Brain + devices + memory task loop
├── memory.py                # Persistent memory
├── continual_learning.py    # Builds training examples, runs learn cycles
├── self_learn.py            # Self-learning entry point
│
├── agent/                   # Task loop, action checks, grounding, verification
├── brain/                   # Neural model, inference, training, tokenizer
├── devices/                 # PC · Android · HTTP device adapters
├── perception/              # Screen and UI observations
├── actuation/               # Action execution
├── eyes/                    # Camera and device discovery
├── hands/                   # Mouse and keyboard control
├── voice/                   # Windows speech input/output
├── web_learner/             # Web collection + experience records
├── recorder/                # Demonstration recording
│
├── config/                  # Settings, devices, web learning
└── data/                    # Conversations, demonstrations, memory
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/Prosanto01/Raksam-AI.git
cd Raksam-AI
pip install -r requirements.txt
```

### Interactive text chat
```bash
python chat.py
```

### Voice interface (Windows)
```bash
py -3.13 voice_chat.py
```

After an action task, say **yes** or **no** when asked whether it worked. Only your approved records are included in later candidate training.

### Run a single task
```bash
python main.py "open notepad and write hello"
```

### Local HTTP API
```bash
python api_server.py
```

See [`API.md`](API.md) for endpoints.

---

## 🧪 Training

Training is separate from runtime and runs offline.

```bash
python brain/train.py
```

Before a normal training run, Raksam can collect a bounded batch of public pages from configured sources in `config/web_learning.yaml`. Only HTTP(S) pages from `allowed_domains` are considered; the crawler checks `robots.txt`, waits between requests, and preserves source URLs and retrieval times.

Use `--skip-web-collection` for an offline run.

**The active compatibility checkpoint is preserved.** Weights are not updated while the agent is controlling the device.

---

## 📊 Status & Limitations

- ✅ RB3 architecture verified — exact parameter count, unified loader, forward passes for all four input formats
- ✅ Full pipeline compiles and runs end-to-end
- ✅ Training path executes successfully (forward/backward/optimizer/validation)
- ⚠️ **Not yet fully trained.** A single CPU optimizer step proves the architecture works; it cannot produce a useful general model.
- ⚠️ Full training requires a GPU and a larger licensed dataset.

The active checkpoint (`brain/checkpoints/latest.pt`, 0.40M compatibility model) remains in service until a GPU-trained candidate passes full held-out evaluation.

---

## ⚠️ Safety — Read Before Using

**Raksam AI can do anything on your computer.** It controls your mouse, your keyboard, your applications, and any device connected to your system. It reads your screen. It can open programs, type text, click buttons, and interact with any window or application on your machine — without restriction.

**I, as the developer, have not added any safety layer to Raksam AI.**

There is:
- ❌ No restricted app list
- ❌ No confirmation dialog before actions
- ❌ No sandboxing
- ❌ No permission gate
- ❌ No limit on what Raksam can do

Raksam will execute whatever command it understands — including destructive actions — if asked or if the model decides to.

### This repository is published for **viewing and learning only**

Raksam AI is **not** released for public use. You are welcome to **read the code, study the architecture, and learn from the design** — but you are **not permitted to run, deploy, or use this software**.

If you choose to ignore this and run Raksam AI anyway:

- You do so **entirely at your own risk**
- You accept **full responsibility** for any consequence
- I, as the developer, accept **no responsibility** for any damage, data loss, unauthorized action, or security incident

**Do not run this software.** It is shared as a research and educational artifact only.

---

## 🗺️ Roadmap

- [x] Text chat
- [x] Voice chat (Windows)
- [x] Screen perception (UI Automation + OCR)
- [x] Mouse/keyboard control
- [x] Multi-device adapters (PC · Android · HTTP)
- [x] Continual learning from user feedback
- [x] RB3 architecture (52M params) verified
- [ ] Full GPU training run
- [ ] Bangla voice input/output
- [ ] Multi-monitor support
- [ ] Linux/macOS voice backends

---

## 📚 Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — High-level design
- [`API.md`](API.md) — HTTP API endpoints
- [`BRAIN_V2.md`](BRAIN_V2.md) — Brain evolution
- [`DEVICE_PROTOCOL.md`](DEVICE_PROTOCOL.md) — Device adapter protocol
- [`VOICE_LIVE_LEARNING.md`](VOICE_LIVE_LEARNING.md) — Voice feedback + learning workflow
- [`CONTINUAL_LEARNING.md`](CONTINUAL_LEARNING.md) — Learning pipeline
- [`SELF_LEARNING.md`](SELF_LEARNING.md) — Self-learning entry point
- [`OWN_BRAIN_TRAINING.md`](OWN_BRAIN_TRAINING.md) — Training guide

---

## 📜 Copyright

Copyright © 2026 **Prosanto Raksam**. All rights reserved.

This project and its source code are the intellectual property of the author.
No part of this repository may be copied, modified, distributed, or used
in any form without explicit written permission from the author.

Viewing and learning from this code is permitted.
Reuse, redistribution, running, or commercial use is strictly prohibited.
