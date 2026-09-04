"""Probe production Velocity-Flat foot-material writes on PhysX."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch

from isaaclab.app import AppLauncher


def _tensor(value):
    return getattr(value, "torch", value)


def _stats(value: torch.Tensor) -> dict[str, object]:
    value = _tensor(value).detach().float().cpu()
    return {
        "shape": list(value.shape),
        "finite": bool(torch.isfinite(value).all().item()),
        "min": float(value.min().item()) if value.numel() else None,
        "max": float(value.max().item()) if value.numel() else None,
        "mean": float(value.mean().item()) if value.numel() else None,
    }


def _material_tensor(robot) -> torch.Tensor:
    import warp as wp

    return wp.to_torch(robot.root_view.get_material_properties()).detach().clone()


def _selected_shape_ids(impl) -> list[int]:
    """Resolve the production term's public ankle bodies to PhysX shape IDs."""

    if impl.num_shapes_per_body is None or impl._backend_body_ids is None:
        raise AssertionError("foot material term did not retain a body-subset selection")
    shape_ids: list[int] = []
    for body_id in impl._backend_body_ids:
        body_id = int(body_id)
        start = sum(impl.num_shapes_per_body[:body_id])
        shape_ids.extend(range(start, start + impl.num_shapes_per_body[body_id]))
    return shape_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    launcher = AppLauncher(args)
    app = launcher.app
    env = None
    result: dict[str, object] = {
        "task": "IsaacLab-Velocity-Flat-MicroDuck",
        "device": args.device,
        "status": "BLOCKED",
        "response": "not_tested",
    }
    try:
        import gymnasium as gym

        from isaaclab_microduck.tasks import register_tasks
        from isaaclab_microduck.tasks.velocity_flat import make_velocity_flat_env_cfg

        register_tasks()
        cfg = make_velocity_flat_env_cfg(num_envs=args.num_envs)
        cfg.seed = args.seed
        env = gym.make("IsaacLab-Velocity-Flat-MicroDuck", cfg=cfg)
        base_env = env.unwrapped
        env.reset(seed=args.seed)

        manager_name = base_env.sim.physics_manager.__name__
        if manager_name.lower() == "ovphysxmanager" or "physx" not in manager_name.lower():
            raise AssertionError(f"expected production PhysX manager, got {manager_name!r}")

        event_cfg = base_env.event_manager.get_term_cfg("randomize_foot_material")
        robot = base_env.scene[event_cfg.params["asset_cfg"].name]
        cached_term = getattr(base_env, "_velocity_flat_foot_material_term", None)
        if cached_term is None:
            raise AssertionError("production randomize_foot_material event did not cache its material term")
        impl = cached_term._impl
        if type(impl).__name__ != "_RandomizeRigidBodyMaterialPhysx":
            raise AssertionError(f"expected production PhysX material implementation, got {type(impl).__name__}")

        selected_body_ids = [int(body_id) for body_id in event_cfg.params["asset_cfg"].body_ids]
        selected_body_names = [robot.body_names[body_id] for body_id in selected_body_ids]
        if set(selected_body_names) != {"ankle_left", "ankle_right"}:
            raise AssertionError(f"unexpected material bodies: {selected_body_names}")
        selected_shape_ids = _selected_shape_ids(impl)
        if not selected_shape_ids:
            raise AssertionError("production foot material selection resolved no PhysX shapes")

        # IsaacLab's production PhysX material term consumes CPU int32 IDs
        # because it forwards them directly to the Warp tensor API.
        env_ids = torch.arange(args.num_envs, device="cpu", dtype=torch.int32)
        cases = []
        previous_selected = None
        for label, coefficient in (("low", 0.1), ("high", 1.2)):
            # The production PhysX term samples buckets only at construction.
            # Fix its existing bucket to isolate the write/readback behavior
            # while still invoking the exact cached task term.
            impl.material_buckets = torch.tensor(
                [[coefficient, coefficient, 0.0]], dtype=torch.float32, device="cpu"
            )
            before = _material_tensor(robot)
            event_cfg.func(
                base_env,
                env_ids,
                static_friction_range=(coefficient, coefficient),
                dynamic_friction_range=(coefficient, coefficient),
                restitution_range=(0.0, 0.0),
                num_buckets=1,
                asset_cfg=event_cfg.params["asset_cfg"],
            )
            after = _material_tensor(robot)
            selected = after[:, selected_shape_ids, :]
            unselected_ids = [index for index in range(after.shape[1]) if index not in selected_shape_ids]
            unselected_unchanged = bool(
                torch.equal(before[:, unselected_ids, :], after[:, unselected_ids, :])
            )
            finite = bool(torch.isfinite(selected).all().item())
            in_range = bool(
                torch.allclose(selected[..., 0], torch.full_like(selected[..., 0], coefficient))
                and torch.allclose(selected[..., 1], torch.full_like(selected[..., 1], coefficient))
                and torch.allclose(selected[..., 2], torch.zeros_like(selected[..., 2]))
            )
            cases.append(
                {
                    "name": label,
                    "requested_coefficient": coefficient,
                    "before": _stats(before[:, selected_shape_ids, :]),
                    "after": _stats(selected),
                    "readback_changed": bool(not torch.equal(before[:, selected_shape_ids, :], selected)),
                    "finite": finite,
                    "in_range": in_range,
                    "unselected_shapes_unchanged": unselected_unchanged,
                }
            )
            if previous_selected is not None:
                cases[-1]["different_from_previous_case"] = bool(not torch.equal(previous_selected, selected))
            previous_selected = selected.clone()

        checks = {
            "production_physx_backend": manager_name.lower() != "ovphysxmanager" and "physx" in manager_name.lower(),
            "cached_production_term": type(impl).__name__ == "_RandomizeRigidBodyMaterialPhysx",
            "both_ankles_selected": set(selected_body_names) == {"ankle_left", "ankle_right"},
            "selected_shapes_nonempty": bool(selected_shape_ids),
            "all_readbacks_finite": all(case["finite"] for case in cases),
            "all_readbacks_in_range": all(case["in_range"] for case in cases),
            "only_selected_shapes_changed": all(case["unselected_shapes_unchanged"] for case in cases),
            "low_high_readbacks_differ": bool(cases[1]["different_from_previous_case"]),
        }
        if not all(checks.values()):
            raise AssertionError(f"production foot material contract failed: {checks}")
        result.update(
            {
                "status": "READBACK_PROVEN",
                "physics_manager": manager_name,
                "material_impl": type(impl).__name__,
                "body_names": selected_body_names,
                "body_ids": selected_body_ids,
                "shape_ids": selected_shape_ids,
                "root_view_type": type(robot.root_view).__name__,
                "cases": cases,
                "checks": checks,
                "response": "not_tested",
                "response_note": (
                    "Production PhysX coefficient writes/readback are proven. "
                    "Tangential contact response remains a separate deterministic fixture."
                ),
            }
        )
    except BaseException as exc:
        result.update(
            {
                "status": "BLOCKED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "response": "unavailable",
            }
        )
    finally:
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded)
        print(encoded, end="")
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
