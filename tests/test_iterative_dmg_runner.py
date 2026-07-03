import collections
import importlib
import os
import sys
import tempfile
import types

import numpy as np
import torch


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


def load_module():
    time_step = collections.namedtuple(
        "TimeStep", ["observation", "reward", "done", "info"]
    )

    diffusion_policy_module = types.ModuleType("diffusion_policy")
    common_module = types.ModuleType("diffusion_policy.common")
    pytorch_util_module = types.ModuleType("diffusion_policy.common.pytorch_util")
    workspace_module = types.ModuleType("diffusion_policy.workspace")
    base_workspace_module = types.ModuleType(
        "diffusion_policy.workspace.base_workspace"
    )
    policy_module = types.ModuleType("diffusion_policy.policy")
    base_policy_module = types.ModuleType("diffusion_policy.policy.base_image_policy")
    gym_util_module = types.ModuleType("diffusion_policy.gym_util")
    video_module = types.ModuleType(
        "diffusion_policy.gym_util.video_recording_wrapper"
    )
    model_module = types.ModuleType("diffusion_policy.model")
    model_common_module = types.ModuleType("diffusion_policy.model.common")
    rotation_module = types.ModuleType(
        "diffusion_policy.model.common.rotation_transformer"
    )

    scripts_module = types.ModuleType("scripts")
    wrapper_module = types.ModuleType("scripts.robomimic_dmg_wrapper")

    class DummyEnvSwitchable:
        pass

    class DummyBaseWorkspace:
        pass

    class DummyBaseImagePolicy:
        pass

    class DummyVideoRecorder:
        @staticmethod
        def create_h264(**kwargs):
            return None

    class DummyRotationTransformer:
        def __init__(self, *args, **kwargs):
            pass

    def dict_apply(data, fn):
        return {key: fn(value) for key, value in data.items()}

    pytorch_util_module.dict_apply = dict_apply
    base_workspace_module.BaseWorkspace = DummyBaseWorkspace
    base_policy_module.BaseImagePolicy = DummyBaseImagePolicy
    video_module.VideoRecorder = DummyVideoRecorder
    rotation_module.RotationTransformer = DummyRotationTransformer

    diffusion_policy_module.common = common_module
    diffusion_policy_module.workspace = workspace_module
    diffusion_policy_module.policy = policy_module
    diffusion_policy_module.gym_util = gym_util_module
    diffusion_policy_module.model = model_module
    common_module.pytorch_util = pytorch_util_module
    workspace_module.base_workspace = base_workspace_module
    policy_module.base_image_policy = base_policy_module
    gym_util_module.video_recording_wrapper = video_module
    model_module.common = model_common_module
    model_common_module.rotation_transformer = rotation_module

    wrapper_module.DMG_env_switchable = DummyEnvSwitchable
    wrapper_module.to_camel_case = lambda name: name
    scripts_module.robomimic_dmg_wrapper = wrapper_module

    stubbed_modules = {
        "diffusion_policy": diffusion_policy_module,
        "diffusion_policy.common": common_module,
        "diffusion_policy.common.pytorch_util": pytorch_util_module,
        "diffusion_policy.workspace": workspace_module,
        "diffusion_policy.workspace.base_workspace": base_workspace_module,
        "diffusion_policy.policy": policy_module,
        "diffusion_policy.policy.base_image_policy": base_policy_module,
        "diffusion_policy.gym_util": gym_util_module,
        "diffusion_policy.gym_util.video_recording_wrapper": video_module,
        "diffusion_policy.model": model_module,
        "diffusion_policy.model.common": model_common_module,
        "diffusion_policy.model.common.rotation_transformer": rotation_module,
        "scripts": scripts_module,
        "scripts.robomimic_dmg_wrapper": wrapper_module,
    }
    original_modules = {
        name: sys.modules.get(name)
        for name in stubbed_modules
    }

    try:
        sys.modules.update(stubbed_modules)
        sys.modules.pop("iterative_dmg_runner", None)
        module = importlib.import_module("iterative_dmg_runner")
    finally:
        for name, original in original_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    return module, time_step


def test_inference_once_aligns_action_chunks_to_bc_start():
    module, time_step = load_module()
    evaluator = module.Robosuite_Evaluator.__new__(module.Robosuite_Evaluator)

    obs = {"state": np.array([1.0, 2.0], dtype=np.float32)}
    captured = {}

    evaluator.t = 10
    evaluator.t_bc_start = 10
    evaluator.max_timesteps = 20
    evaluator.query_cycle = 4
    evaluator.n_obs_steps = 1
    evaluator.device = torch.device("cpu")
    evaluator.obs_shape_meta = {"state": {"shape": (2,), "type": "low_dim"}}
    evaluator.obs_history = {"state": np.zeros((20, 2), dtype=np.float32)}
    evaluator.ts = time_step(obs, 0.0, False, {})
    evaluator.record_frame = lambda current_obs: None

    class FakePolicy:
        def predict_action(self, obs_dict):
            return {
                "action": torch.tensor(
                    [[[10.0], [11.0], [12.0], [13.0]]], dtype=torch.float32
                )
            }

    def step_ts(action):
        captured["action"] = action
        return time_step(obs, 0.0, False, {})

    evaluator.policy = FakePolicy()
    evaluator.step_ts = step_ts

    done = evaluator.inference_once(render=False)

    assert done is False
    np.testing.assert_array_equal(
        captured["action"], np.array([10.0], dtype=np.float32)
    )


def test_replay_tamp_step_updates_history_and_time():
    module, time_step = load_module()
    evaluator = module.Robosuite_Evaluator.__new__(module.Robosuite_Evaluator)

    current_obs = {"state": np.array([1.0, 2.0], dtype=np.float32)}
    next_obs = {"state": np.array([3.0, 4.0], dtype=np.float32)}

    evaluator.max_framerate = 10_000
    evaluator.t = 0
    evaluator.obs_shape_meta = {"state": {"shape": (2,), "type": "low_dim"}}
    evaluator.obs_history = {"state": np.zeros((4, 2), dtype=np.float32)}
    evaluator.ts = time_step(current_obs, 0.0, False, {})
    evaluator.record_frame = lambda current_obs: None
    evaluator.env = types.SimpleNamespace(render=lambda: None)
    evaluator.step_ts = lambda action: time_step(next_obs, 0.0, False, {})

    evaluator.replay_tamp_step(np.array([1.0], dtype=np.float32))

    np.testing.assert_array_equal(
        evaluator.obs_history["state"][0], current_obs["state"]
    )
    np.testing.assert_array_equal(evaluator.ts.observation["state"], next_obs["state"])
    assert evaluator.t == 1


def test_load_checkpoint_passes_env_options_without_legacy_hw_kwargs():
    module, _ = load_module()
    evaluator = module.Robosuite_Evaluator.__new__(module.Robosuite_Evaluator)

    class FakePolicy:
        def eval(self):
            return self

        def to(self, device):
            return self

        def parameters(self):
            return []

    fake_policy = FakePolicy()

    workspace = types.SimpleNamespace(
        load_payload=lambda payload, exclude_keys=None, include_keys=None: None
    )

    cfg = types.SimpleNamespace(
        _target_="fake.workspace",
        policy="fake.policy",
        optimizer="fake.optimizer",
        training=types.SimpleNamespace(use_ema=False),
        name="diffusion-test",
        task=types.SimpleNamespace(
            shape_meta=types.SimpleNamespace(
                obs={"state": {"shape": (2,), "type": "low_dim"}},
                action=types.SimpleNamespace(shape=(7,)),
            )
        ),
        n_action_steps=4,
        n_obs_steps=2,
    )

    captured = {}

    def fake_parent_init(self, *args, **kwargs):
        if "H" in kwargs or "W" in kwargs or "cam_names" in kwargs:
            raise TypeError("legacy image sizing kwargs should not be forwarded")
        captured["args"] = args
        captured["kwargs"] = kwargs

    parent_cls = module.Robosuite_Evaluator.__mro__[1]
    original_parent_init = getattr(parent_cls, "__init__", None)
    original_torch_load = module.torch.load
    original_get_class = module.hydra.utils.get_class
    original_instantiate = module.hydra.utils.instantiate

    with tempfile.NamedTemporaryFile() as handle:
        evaluator.output = "/tmp"
        evaluator.checkpoint_dict = {"task": handle.name}
        evaluator.cur_env_name = "task"
        evaluator.num_inference_steps = 10
        evaluator.max_timesteps = 20
        evaluator.env_initialized = False

        try:
            parent_cls.__init__ = fake_parent_init
            module.torch.load = lambda *args, **kwargs: {"cfg": cfg}
            module.hydra.utils.get_class = lambda target: (
                lambda cfg, output_dir=None: workspace
            )
            module.hydra.utils.instantiate = lambda spec, *args, **kwargs: (
                fake_policy if spec == "fake.policy" else object()
            )

            evaluator.load_checkpoint(
                controller_name="OSC_POSE",
                env_options={"camera_names": ["agentview"]},
            )
        finally:
            if original_parent_init is None:
                delattr(parent_cls, "__init__")
            else:
                parent_cls.__init__ = original_parent_init
            module.torch.load = original_torch_load
            module.hydra.utils.get_class = original_get_class
            module.hydra.utils.instantiate = original_instantiate

    assert captured["kwargs"]["env_options"] == {"camera_names": ["agentview"]}
    assert captured["kwargs"]["max_timesteps"] == evaluator.max_timesteps
