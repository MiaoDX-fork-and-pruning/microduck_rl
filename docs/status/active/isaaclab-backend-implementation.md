status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: implement the IsaacLab backend plan via intuitive-flow
current_slice: explicit IsaacLab BAM wrapper; PhysX friction integration remains
  the simulator gate
blocker_kind: physx_drive_mapping_unresolved
  blocker_fingerprint: converted Microduck USD has 14 joints and limits but zero authored PhysX drive stiffness/damping; InteractiveScene spawn has not completed in a 240s probe
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
  `/workspace/IsaacLab/source/isaaclab_contrib` path on PYTHONPATH.
  A staged articulation probe reaches App, imports, and SimulationContext, but
  remains before `scene_ready` during InteractiveScene construction at 240s;
  this is unproven scene-spawn behavior, not a successful articulation test.
next_action: >-
  Verify the wrapper on an instantiated IsaacLab articulation and define how
  its friction budget is applied to PhysX. Run a one-joint dynamic bench for
  target steps, velocity response, voltage sag, and friction scaling. Keep the
  USD `drive_configured=false` finding visible until that bench passes. Use the
  complete IsaacLab source path set, including `isaaclab_contrib`, for the next
  InteractiveScene/articulation probe.
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
