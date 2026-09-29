# Voice and live learning

Run `voice_chat.py` on Windows. Raksam listens for one task, acts through the
same observe-think-act-verify loop, speaks its result, then asks whether it was
correct. A **yes** moves that task's recorded steps into the approved training
queue; a **no** marks them rejected. The raw log retains both outcomes for
audit.

Approved records contain the goal, screen observation before the action,
structured action, result, and verification status. They are used by a later
candidate training run. This is real-time experience collection and feedback,
not unsafe live weight updates while the AI controls a computer.
