# Raksam Brain V3 (RB3) device bodies

Raksam has one brain and many possible bodies. A body implements `DeviceAdapter`
with two operations: `observe()` and `execute(action)`.

Built-in bodies:
- `pc`: Windows screen/UI/mouse/keyboard/applications.
- `android_adb`: Android screen/focus + touch/text/key/launch.
- `http`: generic JSON bridge for user-owned robots, machines, IoT, test rigs,
  vehicle simulators/interfaces, and custom hardware.

To support a new device, implement an adapter in `devices/` and register its
kind in `devices/registry.py`. The brain and agent loop do not need to know the
vendor or operating system.

## Public-web learning before training

Before a normal `brain/train.py` run, Raksam can collect a bounded batch of
public pages from the owner-configured sources in `config/web_learning.yaml`.
Only HTTP(S) pages from `allowed_domains` are considered; the crawler checks
`robots.txt`, waits between requests, preserves each source URL and retrieval
time in `data/web_knowledge/manifest.jsonl`, and writes usable text to dated
JSONL files in `data/web_knowledge/`.  Those records are included by the
training dataset in the same run.

Keep the source list limited to material you have the right to use. The
crawler does not sign in, bypass access controls, or execute downloaded code.
Use `--skip-web-collection` with `brain/train.py` for an offline run.

## Voice and live learning

On Windows, run `py -3.13 voice_chat.py` to speak a task and hear Raksam's
response. After an action task, say **yes** or **no** when asked whether it
worked. The app saves the screen state before each action plus the result;
only your approved records are included in later candidate training. Voice
input uses Windows Speech Recognition and voice output uses Windows Speech
Synthesis. It does not modify model weights while controlling your device.

For physical vehicles/industrial equipment, use an appropriate hardware
controller and enforce device-level limits; do not expose an unauthenticated
raw actuator endpoint to the public internet.
