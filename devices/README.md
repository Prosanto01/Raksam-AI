# Raksam V3 device bodies

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

For physical vehicles/industrial equipment, use an appropriate hardware
controller and enforce device-level limits; do not expose an unauthenticated
raw actuator endpoint to the public internet.
