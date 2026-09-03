status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: implement the IsaacLab backend plan via intuitive-flow
current_slice: BAM actuator numerical parity; PhysX friction integration remains
  the simulator gate
blocker_kind: physx_drive_mapping_unresolved
  blocker_fingerprint: converted Microduck USD has 14 joints and limits but zero authored PhysX drive stiffness/damping
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
  core. The 225-point voltage/torque/friction grid matches the reference BAM
  implementation to floating-point precision (max error 1.4e-16). Focused
  IsaacLab/BAM contract suite: 25 passed.
next_action: >-
  Implement the IsaacLab explicit actuator wrapper around the proven math and
  define how its friction budget is applied to PhysX. Verify drive stiffness,
  effort limits, armature, and friction behavior in a one-joint dynamic bench.
  Keep DCMotorCfg marked as temporary skeleton only.
next_proof: >-
  `scripts/isaaclab/docker-run.sh -lc 'PYTHONPATH=... /isaac-sim/python.sh
  scripts/isaaclab/inspect_usd.py .cache/isaaclab-assets/microduck_walk.usd
  --headless --output .cache/isaaclab-assets/microduck_walk.usd.report.json'`,
  followed by actuator numerical tests and existing mjlab regression checks.
stop_condition: >-
  Do not claim simulator parity or start PPO until PhysX drive/BAM behavior is
  explicitly mapped and the deterministic asset/actuator acceptance is green.
no_touch_scope: existing mjlab code, dependency lock, production runtime
parked_todos: >-
  Plan Tasks E-I (BAM bench, physics battery, Velocity-Flat smoke, walking run,
  continuation decision) remain pending; DCMotorCfg is not accepted as BAM
  parity.
