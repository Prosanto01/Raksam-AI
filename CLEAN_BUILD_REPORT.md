# Raksam AI V3 — Clean Build Report

Created by: Prosanto Raksam

## What was fixed
- Runtime `Turn` objects are accepted by chat history formatting.
- Conversation answers can retrieve exact approved training examples before neural generation.
- Unknown/out-of-curriculum questions return a clear learning fallback instead of a random hallucination.
- `in <app> write <text>` and `open <app> and write <text>` plans are supported.
- Typing now focuses the foreground editable/document control before sending keyboard input.
- Application launch waits for the new window and attempts to focus it.
- UI verification understands the real `ScreenState` structure.
- Action generation has reliable local heuristics for common open/type/click/key actions.
- Training data was cleaned to a curated conversation/action curriculum.
- Runtime caches, old demonstration screenshots, old experience logs, and Python cache files were removed.

## Training
- Fresh from-scratch checkpoint.
- 4 completed CPU training epochs in this build.
- 4,578 usable training examples.
- Best recorded validation loss: 0.3081.
- Model: Raksam Brain V3 (RB3), 0.40M parameters.
- No Ollama, OpenAI model, Gemini model, or other external AI runtime is used.

## Important limitation
The included neural model is intentionally small so it can run and train on CPU. It is not comparable to frontier models yet. The project is structured so the model and training curriculum can be scaled later. GUI execution still needs to be tested on the user's actual Windows desktop because this build environment cannot control the user's physical screen.

## Start
```powershell
cd "C:\MY AI\Raksam AI"
py -3.13 chat.py
```

## Continue training
```powershell
py -3.13 -m brain.train --resume --epochs 1 --batch-size 32
```

For a clean new training run, delete `brain/checkpoints/latest.pt` and `brain/checkpoints/optimizer.pt`, then run the trainer without `--resume`.
