# Raksam V3 universal device protocol

Raksam does not need to know every brand of device. It needs a link that exposes
an observation channel and an action channel.

## Abstract body

```text
observe() -> current state
execute(action) -> result
```

The brain can therefore stay device-independent:

```text
User goal -> Brain -> abstract action -> Device Adapter -> physical device
                         ^                         |
                         |---- new observation ----|
```

## Adding a body

Implement `DeviceAdapter` in `devices/` and register its `kind` in
`devices/registry.py`.

Examples:
- PC: screen/UI + mouse/keyboard
- Android: screen/focus + touch/keys
- Robot: cameras/sensors + motors/arm controller
- Machine: sensors + PLC/controller bridge
- IoT: telemetry + command bridge
- Custom hardware: any authenticated local/private protocol

The generic HTTP adapter is intentionally protocol-neutral. The external
controller decides how its hardware is actually driven.
