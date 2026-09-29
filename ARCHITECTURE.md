# Raksam AI architecture — independent multi-device agent

## Principle

Raksam's neural network is the intelligence layer. It is trained from scratch and is responsible for language, task interpretation, planning, action selection, and learning from examples. It does not call another AI model at runtime.

Deterministic software surrounds it for perception, memory, permissions and hardware protocols.

```text
User / Voice / App / Event
          |
          v
   +-------------------+
   | RAKSAM BRAIN      |
   | language + plan   |
   | + action policy   |
   +---------+---------+
             |
       structured intent
             |
   +---------v---------+
   | DEVICE / TOOL BUS  |
   +----+------+--------+
        |      |       |
       PC   Android  HTTP/IoT
        |      |       |
        +------+-------+
               |
        result + new state
               |
               v
        memory / next step
```

## Why adapters are not separate AIs

A device adapter is a protocol translator, not an intelligence system. For example, the Android adapter translates `android_tap(x,y)` into an ADB command. This keeps the model independent of any particular operating system or vendor.

## Extending to other devices

Add a new adapter implementing `DeviceAdapter.execute(action)` and register its capabilities in `devices/registry.py`. Examples include:

- smart-home hubs
- robots
- cameras
- industrial/lab controllers
- game consoles
- vehicle test systems
- custom microcontroller gateways

The adapter must enforce its own protocol validation, while Raksam's central permission gate remains the final policy layer.

## Scaling the brain

The included model is a research-scale starting point, not a frontier model. A path toward a much stronger independent Raksam is:

**data → tokenizer → pretraining → instruction tuning → action demonstrations → preference/evaluation training → tool/device learning → continual evaluation**.

The runtime architecture does not require changing when the model grows from millions to billions of parameters.
