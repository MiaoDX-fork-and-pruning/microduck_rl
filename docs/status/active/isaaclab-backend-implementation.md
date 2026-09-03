status: ACTIVE
source_plan: docs/isaaclab_backend_implementation_plan.md
control_plane: /root
latest_intent: implement the IsaacLab backend plan via intuitive-flow
current_slice: switching to the officially paired IsaacLab 3.0 beta / Isaac Sim 6.0.1 runtime
blocker_kind: mjcf_importer_no_artifact
  blocker_fingerprint: prior Isaac Sim 5.0 MJCFCreateAsset hang; superseded by runtime switch
last_proven_evidence: >-
  Revalidated this continuation: `python` cannot import isaaclab, isaacsim, or
  omni.isaac.lab; `command -v` finds no isaaclab/isaacsim; host Python is
  Derived image probe exits 0 with CUDA available, Python 3.11.13, Torch
  2.7.0+cu128, IsaacLab package 0.44.9 from source release v2.2.0, and Isaac
  Sim 5.0.0-rc.45 on the RTX 3090. Empty-stage smoke also exits successfully
  after 3 SimulationContext steps. Both IsaacLab MjcfConverter and the
  supported standalone `MJCFCreateImportConfig`/`MJCFCreateAsset` API leave no
  USD file or success marker; the latter reaches `[app ready]` then exceeds a
  90-second bound. A 4 KB cached `.usd` is only the default empty stage and
  has not passed robot prim/joint inspection, so it is not conversion proof.
  Passing the documented `dest_path` argument did not change this behavior;
  the output remains 4,150 bytes.
  A clean run with NVIDIA's bundled `nv_ant.xml` fixture blocks inside the
  native `MJCFCreateAsset` call: the Python process remains in
  `futex_wait_queue` before the wrapper's post-command timeout loop. Three
  stale conversion containers were removed and the result reproduced from a
  single clean container, ruling out competing Isaac Sim instances.
completed: >-
  Added isolated runtime and Docker launchers, pinned IsaacLab source checkout,
  lazy package/task registry, 61D/14D policy ABI with golden fixture, and MJCF
  asset parity report. Focused deterministic suite currently passes 14 tests;
  conversion and diagnostic contract tests pass as well.
next_action: >-
  Finish pulling/building the official Isaac Sim 6.0.1 image, run the headless
  probe and bundled `nv_ant.xml` conversion, then retry Microduck. Do not
  hand-maintain a USD without a reproducible source conversion.
next_proof: >-
  `scripts/isaaclab/docker-run.sh ./python.sh -u scripts/isaaclab/probe.py`
  against a completed official Isaac Sim image, plus existing mjlab regression
  checks.
stop_condition: >-
  Do not create the backend skeleton or claim implementation completion until
  Task A acceptance is proven.
no_touch_scope: existing mjlab code, dependency lock, production runtime
parked_todos: full plan Tasks A-I remain pending
