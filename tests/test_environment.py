from pathlib import Path


def test_t2_walk_task_imports_without_linux_only_deps():
    from tasks.locomotion.robots.t2 import T2WalkTaskCfg

    cfg = T2WalkTaskCfg()
    assert cfg.policy.checkpoint_path == "robots/t2/models/t2_walk.pt"
    assert len(cfg.policy.policy_joint_names) == 23


def test_t2_policy_checkpoint_runs_on_cpu_with_expected_shapes():
    import torch
    import tasks.locomotion as loco
    from booster_deploy.utils.policy_runner import create_policy_runner

    path = Path(loco.__file__).parent / "robots/t2/models/t2_walk.pt"
    runner = create_policy_runner(str(path), torch.device("cpu"), use_actor_module=False)
    # ang_vel 3 + gravity 3 + command 3 + dof_pos 23 + dof_vel 23 + last_action 23 + heading_err 1 + standing 1
    obs_dim = 80
    out = runner(torch.zeros(10 * obs_dim)).reshape(-1)
    assert out.numel() == 23
    assert torch.isfinite(out).all()


def test_t2_mjcf_joint_and_actuator_order_matches_robot_cfg():
    import mujoco
    from booster_assets import BOOSTER_ASSETS_DIR
    from booster_deploy.robots.t2 import T2_31DOF_CFG

    path = T2_31DOF_CFG.mjcf_path.replace("{BOOSTER_ASSETS_DIR}", str(BOOSTER_ASSETS_DIR))
    model = mujoco.MjModel.from_xml_path(path)
    joints = [model.joint(i).name for i in range(1, model.njnt)]  # joint 0 is the free joint
    actuators = [model.actuator(i).name for i in range(model.nu)]
    assert joints == list(T2_31DOF_CFG.joint_names)
    assert actuators == list(T2_31DOF_CFG.joint_names)
