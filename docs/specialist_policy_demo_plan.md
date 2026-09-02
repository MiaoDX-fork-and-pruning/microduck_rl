# Specialist policy program

Status: **training and per-policy acceptance complete; integrated demo pending**

The specialist program produced and accepted 13 independent policies under the
shared 61D observation and 14D action contract. The immutable final selection is
recorded in
[`cloudml/specialist-final-checkpoints-remediated-facd4f4.json`](../cloudml/specialist-final-checkpoints-remediated-facd4f4.json),
with reproducibility details in
[`docs/human/training-reproducibility.md`](human/training-reproducibility.md).

The remaining work is packaging and demonstrating the accepted artifacts. It is
specified separately in the executable plan:

[`docs/plans/specialist-demo-implementation.md`](plans/specialist-demo-implementation.md)

## Scope

- Preserve the accepted checkpoint inventory; later VelStand continuation does
  not replace the frozen accepted checkpoint without a new evaluation.
- Build deterministic, scenario-driven Track A and Track B demo slices.
- Export and index PT/ONNX artifacts, parity evidence, videos, and hashes.
- Keep the 61D ABI and deployment semantics unchanged.

## Completed evidence

- S0 preflight, 13-task smoke matrix, immutable source/image package.
- S1 training and S2 per-policy evaluation: 13 of 13 policies accepted.
- S3 representative ONNX export and golden-action parity.
- Canonical scenario, artifact validator, router tests, and gallery tooling.

## Out of scope

Generalist distillation, 71D schemas, hardware deployment, Rough/Backlash
robustness passes, left-kick expansion, and new training runs are separate work.
