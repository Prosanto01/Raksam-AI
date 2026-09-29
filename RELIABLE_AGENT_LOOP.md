# Reliable agent loop

This build strengthens the links between the brain, eyes, and hands:

1. The brain may propose only a structured action. Invalid or unsupported
   actions stop before reaching a device.
2. A named click is grounded to a visible UI Automation element; without a
   visible target or coordinates it is not executed.
3. Meaningful actions must produce an observable UI change. A click, type,
   navigation, or key press is no longer treated as successful merely because
   no Python exception occurred.
4. Repeating the same unverified action twice stops the task instead of
   blindly looping.
5. A `done` action cannot finish a task before at least one action has been
   visibly verified.
6. Every agent step is saved to the raw experience log with its verification
   result. Only human-approved records enter supervised training.

These safeguards make failure explicit so it can be demonstrated, recorded,
and used in future training. They cannot replace real-device testing or make
an under-trained model understand an app it has never observed.
