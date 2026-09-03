status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: implement the IsaacLab backend plan via intuitive-flow
current_slice: staged IsaacLab articulation reset/cooking diagnosis; PhysX
  friction integration remains the simulator gate
blocker_kind: articulation_reset_hang
  blocker_fingerprint: with the complete runtime PYTHONPATH, the Microduck
  InteractiveScene reaches scene_ready but native sim.reset() does not return;
  the same boundary persists after removing all 75 USD collision APIs
last_proven_evidence: >-
  The derived `microduck-isaaclab:3.0.0-beta2.patch1-isaacsim6.0.1` image
  starts Isaac Sim 6.0.1 headless with CUDA, Python 3.12.13, Torch 2.10.0+cu128,
  and Warp 1.13.0. SimulationContext steps five frames. The current MJCF
  importer API converts both bundled `nv_ant.xml` and the Microduck walk MJCF
  reproducibly. Microduck inspection reports 169 active prims, 15 rigid bodies,
  75 collision geometries, one articulation root at
  `/microduck/Geometry/trunk_base`, all 14 policy joints with limits, and no
  missing joints. It also reports `drive_configured=false`: every converted
  joint has zero PhysX drive stiffness and damping, despite maxForce metadata.
completed: >-
  Added isolated runtime and Docker launchers, pinned IsaacLab source checkout,
  lazy package/task registry, 61D/14D policy ABI with golden fixture, MJCF
  mechanical report, current Isaac Sim MJCF conversion/diagnostic path, a
  machine-readable USD inspection script, and a Torch BAM XL330/M6 numerical
  core and an explicit IsaacLab `BamActuatorCfg`/`BamActuator` wrapper. The
  225-point voltage/torque/friction grid matches the reference BAM
  implementation to floating-point precision (max error 1.4e-16). The wrapper
  imports and the Microduck asset config resolves in the actual 6.0.1 runtime;
  a 2-environment `BamActuator.compute()` smoke also returns finite efforts.
  CPU/runtime contracts: 30 passed. InteractiveScene probing identified that
  IsaacLab 3.0 beta also requires the bundled
  `/workspace/IsaacLab/source/isaaclab_contrib` path on PYTHONPATH. PhysX
  articulation loading additionally probes `isaaclab_newton`, so the runtime
  command now includes the IsaacLab `newton`, `ov`, and `ovphysx` source paths.
  A staged articulation probe reaches App, imports, SimulationContext, and
  `scene_ready` with the project `/src` path included. Both canonical and
  `--no-collisions` probes then block inside native `sim.reset()` before
  `sim_reset`, `articulation_ready`, or `step_ok`; this is an explicit
  articulation-reset/cooking blocker, not a successful articulation test.
next_action: >-
  Diagnose the native reset hang with a minimal/reduced-collision diagnostic
  asset or direct PhysX articulation setup; do not replace the canonical USD.
  Once reset completes, verify the wrapper on an instantiated articulation and
  define how its friction budget is applied to PhysX. Keep the USD
  `drive_configured=false` finding visible until that bench passes.
next_proof: >-
  `scripts/isaaclab/docker-run.sh -lc 'PYTHONPATH=... /isaac-sim/python.sh
  scripts/isaaclab/inspect_usd.py .cache/isaaclab-assets/microduck_walk.usd
  --headless --output .cache/isaaclab-assets/microduck_walk.usd.report.json'`,
  followed by the staged `scripts/isaaclab/articulation_probe.py`, actuator
  numerical tests, and existing mjlab regression checks.
stop_condition: >-
  Do not claim simulator parity or start PPO until PhysX drive/BAM behavior is
  explicitly mapped and the deterministic asset/actuator acceptance is green.
no_touch_scope: existing mjlab code, dependency lock, production runtime
parked_todos: >-
  Plan Tasks E-I (BAM bench, physics battery, Velocity-Flat smoke, walking run,
  continuation decision) remain pending; DCMotorCfg is not accepted as BAM
  parity.
