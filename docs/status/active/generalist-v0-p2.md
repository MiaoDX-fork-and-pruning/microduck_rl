Status: ACTIVE
Phase: P2 walk conditioned BC baseline

Completed slice:
- Frozen `generalist-v0` schema v2 adapter with 71D input and 14D raw action.
- G0 teacher collection from immutable `velstand_flat` and `velocity_flat` P0 traces.
- Offline BC MLP smoke completed with 3,900 samples (3 epochs, seed 7).

Evidence:
- Contract tests: `tests/test_generalist_schema.py` (6 passed).
- BC smoke output was written to the untracked `/tmp/p2-walk-bc-smoke` directory.
- Smoke metrics: train MSE `0.0067272`, validation MSE `0.0043456`.
- 30-epoch deterministic reruns produced identical model hashes and metrics:
  train MSE `0.00125484`, validation MSE `0.00111708`.
  Per-behavior validation MSE: stand `0.00005858`, locomotion `0.00120211`.
- The dataset contains 3,900 finite samples and is serializable as NPZ with
  explicit 71D/14D shapes.

Remaining P2 work:
- Freeze a versioned teacher manifest outside Git with checkpoint/ONNX hashes.
- Add Rust-readable golden vectors and export metadata for the 71D student contract.
- Compare the BC candidate in MuJoCo against P1 per-behavior and transition baselines.
- Do not add sit/kick/roulade until the G0 baseline and transition gates are reviewed.

No-touch: specialist 61D ABI, production runtime defaults, official scheduler, and roller track.
