"""
Usage:
python eval_aloha.py \
    -i data/outputs/2024.07.18/19.44.55_act_aloha_starbucks/checkpoints/latest.ckpt \
    -o data/eval/aloha_starbucks/ \
    -t 500
"""

import os
import pathlib
import time
import threading
import numpy as np
import copy
import json
import numpy as np
import torch
import dill
import hydra
import dm_env

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from aloha_pkg.aloha_scripts.real_env import make_real_env

from aloha_pkg.aloha_scripts.constants import DT, PUPPET_GRIPPER_JOINT_NORMALIZE_FN, PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN, PUPPET_POS2JOINT
from diffusion_policy.real_world.video_recorder import save_videos




def get_seq_obs(obs_history, t, n_obs_steps, max_timesteps):
    if max_timesteps <= 0:
        raise ValueError(f"max_timesteps must be positive, got {max_timesteps}")
    if n_obs_steps > max_timesteps:
        raise ValueError(
            f"n_obs_steps ({n_obs_steps}) cannot exceed ring buffer length "
            f"max_timesteps ({max_timesteps})"
        )
    # Clamp negative logical indices to 0 to replicate the first observation as padding
    # for early steps (t < n_obs_steps - 1). After buffer wrap, modulo maps each
    # logical timestep to its physical ring slot.
    logical_indices = np.maximum(0, np.arange(t - n_obs_steps + 1, t + 1))
    physical_indices = logical_indices % max_timesteps
    obs_dict_np = dict()
    for k, v in obs_history.items():
        obs_dict_np[k] = v[physical_indices]
    return obs_dict_np

class DP_Evaluator():
    def __init__(self, checkpoint_dict, output, max_timesteps, num_inference_steps, scale, with_planning= False, render_obs_keys=None, fps=30, hdf5_image_size=(128, 128), save_hdf5=False):
        self.max_timesteps = max_timesteps
        self.checkpoint_dict = checkpoint_dict
        self.output = output
        self.num_inference_steps = num_inference_steps

        self.with_planning = with_planning
        self.image_list = []
        self.camera_names = self._resolve_camera_names()
        if 'cam_high' not in self.camera_names:
            raise ValueError("DP_Evaluator requires 'cam_high' in camera_names for video recording.")
        self.render_obs_keys = [] if render_obs_keys is None else list(render_obs_keys)
        env_camera_names = list(dict.fromkeys(self.camera_names + self.render_obs_keys))

        self.fps = fps
        self.save_hdf5 = save_hdf5
        self.hdf5_path = None
        if self.save_hdf5:
            if len(hdf5_image_size) != 2:
                raise ValueError(f"hdf5_image_size must be (width, height), got {hdf5_image_size}")
            self.hdf5_image_size = tuple(hdf5_image_size)
            self.hdf5_buffers = {cam: [] for cam in self.render_obs_keys}
        else:
            self.hdf5_image_size = None
            self.hdf5_buffers = None
        # setup experiment
        self.env = make_real_env(
            init_node=True,
            downsample_scale=scale,
            setup_robots=not self.with_planning,
            camera_names=env_camera_names,
        )

        # Freeze-control related state (optional, per-skill).
        self.allow_freeze = False
        self.last_action = None
        # Text overlay for video frames (e.g., freeze indicators).
        self.overlay_text = None
        # Startup gripper limiter: cap the first N policy timesteps after reset.
        self.startup_gripper_max_delta = 0.02
        self.startup_gripper_limit_timesteps = 10
        self.startup_gripper_limit_step = 0
        self.startup_gripper_limit_active = False
        self.startup_gripper_reference = None

        self._prefetch_lock = threading.Lock()
        self._prefetch_thread = None
        self._prefetch_skill_requested = None
        self._prefetch_ready_skill = None
        self._prefetch_bundle = None
        self._prefetch_error = None
        self._prefetch_done = threading.Event()

        skill_names = list(checkpoint_dict.keys())
        if not skill_names:
            raise ValueError("checkpoint_dict must contain at least one skill")
        self._default_skill = skill_names[0]
        self.load_skill_weights(self._default_skill)
        self.reset_loaded_skill(reset_grippers=False)

    def _resolve_camera_names(self):
        default_camera_names = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
        skill_names = list(self.checkpoint_dict.keys())
        if len(skill_names) == 0:
            return default_camera_names

        first_ckpt = self.checkpoint_dict[skill_names[0]]
        payload = torch.load(open(first_ckpt, 'rb'), pickle_module=dill)
        cfg = payload['cfg']

        if hasattr(cfg, "task") and hasattr(cfg.task, "dataset") and hasattr(cfg.task.dataset, "camera_names"):
            return list(cfg.task.dataset.camera_names)
        return default_camera_names

    def cancel_prefetch(self, join_timeout_s=5.0):
        """Invalidate in-flight prefetch and wait for the worker to finish (no-op if idle)."""
        with self._prefetch_lock:
            self._prefetch_skill_requested = None
            self._prefetch_ready_skill = None
            self._prefetch_bundle = None
            self._prefetch_error = None
            self._prefetch_done.set()
            thr = self._prefetch_thread
            self._prefetch_thread = None
        if thr is not None:
            thr.join(timeout=join_timeout_s)
            if thr.is_alive():
                raise RuntimeError(
                    "DP_Evaluator.prefetch worker did not stop within "
                    f"{join_timeout_s}s; refusing to continue with stale state"
                )

    def _build_skill_bundle(self, skill_name):
        if skill_name not in self.checkpoint_dict:
            raise KeyError(f"Unknown skill {skill_name!r} for DP checkpoint_dict")
        payload = torch.load(
            open(self.checkpoint_dict[skill_name], "rb"), pickle_module=dill
        )
        cfg = payload["cfg"]

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=self.output)
        workspace: BaseWorkspace
        if "model" not in workspace.__dict__.keys():
            workspace.model = hydra.utils.instantiate(cfg.policy)
        if "ema_model" not in workspace.__dict__.keys() and cfg.training.use_ema:
            workspace.ema_model = copy.deepcopy(workspace.model)
        if "optimizer" not in workspace.__dict__.keys():
            workspace.optimizer = hydra.utils.instantiate(
                cfg.optimizer, workspace.model.parameters()
            )
        workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)

        policy: BaseImagePolicy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device("cuda")
        policy.eval().to(device)
        policy.num_inference_steps = self.num_inference_steps

        obs_shape_meta = cfg.task.shape_meta.obs
        query_cycle = cfg.n_action_steps
        n_obs_steps = cfg.n_obs_steps
        allow_freeze = False

        return {
            "skill_name": skill_name,
            "policy": policy,
            "obs_shape_meta": obs_shape_meta,
            "query_cycle": query_cycle,
            "n_obs_steps": n_obs_steps,
            "allow_freeze": allow_freeze,
            "device": device,
        }

    def _apply_skill_bundle(self, bundle):
        self.cur_skill = bundle["skill_name"]
        self.policy = bundle["policy"]
        self.obs_shape_meta = bundle["obs_shape_meta"]
        self.query_cycle = bundle["query_cycle"]
        self.n_obs_steps = bundle["n_obs_steps"]
        if self.max_timesteps <= 0:
            raise ValueError(f"max_timesteps must be positive, got {self.max_timesteps}")
        if self.n_obs_steps > self.max_timesteps:
            raise ValueError(
                f"n_obs_steps ({self.n_obs_steps}) cannot exceed max_timesteps "
                f"({self.max_timesteps}) for rolling obs_history"
            )
        self.allow_freeze = bundle["allow_freeze"]
        self.device = bundle["device"]
        self._clear_inference_action_cache()

    def _clear_inference_action_cache(self):
        """Invalidate cached policy action chunk (policy weights unchanged)."""
        self.np_action_seq = None

    def _policy_step_offset(self):
        qc = self.query_cycle
        if qc <= 0:
            raise RuntimeError(f"query_cycle must be positive, got {qc}")
        return (self.t - self.t_bc_start) % qc

    def mark_bc_segment_start(self):
        """Align diffusion policy chunk phase with current global timestep (call at each Graphstate / LfD start)."""
        self.t_bc_start = self.t
        self.np_action_seq = None

    def load_skill_weights(self, skill_name):
        """Load policy weights and obs metadata for skill_name only (no env / buffer / timestep changes)."""
        self.cancel_prefetch()
        bundle = self._build_skill_bundle(skill_name)
        self._apply_skill_bundle(bundle)

    def prefetch_skill(self, skill_name):
        """Start loading skill_name in a background thread; results are consumed via consume_prefetched_weights."""
        if skill_name not in self.checkpoint_dict:
            raise KeyError(f"Unknown skill {skill_name!r} for DP prefetch")

        def worker(target_skill):
            err = None
            bundle = None
            try:
                bundle = self._build_skill_bundle(target_skill)
            except Exception as e:
                err = e
            with self._prefetch_lock:
                if self._prefetch_skill_requested != target_skill:
                    return
                self._prefetch_ready_skill = target_skill
                self._prefetch_bundle = bundle
                self._prefetch_error = err
                self._prefetch_done.set()

        self.cancel_prefetch()
        with self._prefetch_lock:
            self._prefetch_done.clear()
            self._prefetch_skill_requested = skill_name
            self._prefetch_ready_skill = None
            self._prefetch_bundle = None
            self._prefetch_error = None
            self._prefetch_thread = threading.Thread(
                target=worker, args=(skill_name,), daemon=True
            )
            self._prefetch_thread.start()

    def consume_prefetched_weights(self, skill_name, wait_timeout_s=600.0):
        """Wait for prefetch of skill_name and apply weights on this thread. Raises on mismatch or load error."""
        if not self._prefetch_done.wait(timeout=wait_timeout_s):
            raise RuntimeError(
                f"Timed out waiting {wait_timeout_s}s for prefetch of {skill_name!r}"
            )
        with self._prefetch_lock:
            ready = self._prefetch_ready_skill
            bundle = self._prefetch_bundle
            err = self._prefetch_error
            thr = self._prefetch_thread
            self._prefetch_thread = None
            self._prefetch_skill_requested = None
            self._prefetch_ready_skill = None
            self._prefetch_bundle = None
            self._prefetch_error = None
            self._prefetch_done.clear()

        if thr is not None:
            thr.join(timeout=5.0)
            if thr.is_alive():
                raise RuntimeError("Prefetch thread did not terminate after consume_prefetched_weights")

        if ready is None and err is None:
            raise RuntimeError(
                f"Prefetch was cancelled or produced no result while waiting for {skill_name!r}"
            )
        if ready != skill_name:
            raise RuntimeError(
                f"Prefetch mismatch: expected {skill_name!r}, got {ready!r}"
            )
        if err is not None:
            raise RuntimeError(f"Prefetch failed for {skill_name!r}") from err
        if bundle is None:
            raise RuntimeError(f"Prefetch produced no bundle for {skill_name!r}")
        self._apply_skill_bundle(bundle)

    def set_skill(self, skill_name, reset_grippers=True):
        self.cancel_prefetch()
        self.load_skill_weights(skill_name)
        self.reset_all(reset_grippers=reset_grippers)

    def load_checkpoint(self):
        """Backward-compatible alias: load weights for current cur_skill."""
        self.load_skill_weights(self.cur_skill)

    def reset_loaded_skill(self, reset_grippers=False, reinit_buffers=True):
        """Episode boundary for the already-loaded skill: optional gripper reboot, (re)allocate obs buffers, t=0."""
        self.ts = self.env.reset(fake=self.with_planning)
        print(f"Reset DP env! (reset_grippers={reset_grippers}, reinit_buffers={reinit_buffers})")
        if reset_grippers:
            self.env.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            self.env.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
        if reinit_buffers:
            if self.obs_shape_meta is None:
                raise RuntimeError("obs_shape_meta is not set; load_skill_weights before reset_loaded_skill")
            self.obs_history = dict()
            for key in self.obs_shape_meta.keys():
                self.obs_history[key] = np.zeros(
                    (self.max_timesteps, *self.obs_shape_meta[key].shape),
                    dtype=np.float32,
                )
        self.t = 0
        self.t_bc_start = 0
        self.last_action = None
        self.startup_gripper_reference = (
            self._get_joint_normalized_gripper_reference_from_obs()
        )
        self.startup_gripper_limit_step = 0
        self.startup_gripper_limit_active = True
        self._clear_inference_action_cache()

    def reset_all(self, reset_grippers=True):
        """Full reset including observation buffers (matches legacy set_skill path)."""
        self.reset_loaded_skill(reset_grippers=reset_grippers, reinit_buffers=True)

    def refresh_ts_from_env(self):
        """Refresh policy timestep observation from the real env (no physics step)."""
        self.ts = dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.env.get_reward(),
            discount=None,
            observation=self.env.get_observation(),
        )

    def collect_obs_for_tamp_step(self):
        """TAMP connector replay: sync obs from env, write rolling obs_history, then t += 1."""
        if self.obs_history is None:
            raise RuntimeError("obs_history is not allocated; call reset_loaded_skill first")
        self.refresh_ts_from_env()
        self.collect_obs()
        self.t += 1

    def collect_obs(self):
        ## NOTE: we need to load a dp snapshot first to get obs_shape_meta
        obs_slot = self.t % self.max_timesteps
        for k, v in self.obs_shape_meta.items():
            if v["type"] == "rgb":
                self.obs_history[k][obs_slot] = np.moveaxis(
                    self.ts.observation["images"][k].astype(np.float32) / 255.0, -1, 0
                )
            else:
                self.obs_history[k][obs_slot] = self.ts.observation[k].astype(np.float32)
        return

    def _get_joint_normalized_gripper_reference_from_obs(self):
        qpos = self.ts.observation["qpos"]
        gripper_pos_norm = np.array([qpos[6], qpos[13]], dtype=np.float32)
        if not np.all(np.isfinite(gripper_pos_norm)):
            raise ValueError(
                "Expected finite gripper qpos observation, "
                f"got {gripper_pos_norm}"
            )
        # bridge: position-normalized -> raw position -> raw joint -> joint-normalized
        gripper_pos = PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN(gripper_pos_norm)
        gripper_joint = PUPPET_POS2JOINT(gripper_pos)
        return PUPPET_GRIPPER_JOINT_NORMALIZE_FN(gripper_joint).astype(np.float32)

    def _apply_freeze_blend(self, full_action, base_action):
        if not (self.allow_freeze and full_action.shape[-1] >= 16):
            return base_action

        # Treat the last two dimensions as continuous freeze strengths alpha in [0, 1]
        # and blend between the current DP command and the last held command:
        # u = (1 - alpha) * u_DP + alpha * u_hold
        freeze = np.clip(full_action[14:16], 0.0, 1.0)
        left_alpha = float(freeze[0])
        right_alpha = float(freeze[1])
        self.overlay_text = f"L:{left_alpha:.2f} R:{right_alpha:.2f} allow_freeze:{self.allow_freeze}"

        left_slice = slice(0, 7)
        right_slice = slice(7, 14)
        if self.last_action is not None:
            if left_alpha > 0.0:
                base_action[left_slice] = (
                    (1.0 - left_alpha) * base_action[left_slice]
                    + left_alpha * self.last_action[left_slice]
                )
            if right_alpha > 0.0:
                base_action[right_slice] = (
                    (1.0 - right_alpha) * base_action[right_slice]
                    + right_alpha * self.last_action[right_slice]
                )
        return base_action

    def _apply_startup_gripper_limiter(self, base_action):
        if not self.startup_gripper_limit_active:
            return base_action

        gripper_indices = np.array([6, 13], dtype=np.int64)
        self.startup_gripper_reference = (
            self._get_joint_normalized_gripper_reference_from_obs()
        )

        target = base_action[gripper_indices]
        delta = target - self.startup_gripper_reference
        clipped_delta = np.clip(
            delta,
            -self.startup_gripper_max_delta,
            self.startup_gripper_max_delta,
        )
        base_action[gripper_indices] = self.startup_gripper_reference + clipped_delta
        self.startup_gripper_limit_step += 1
        if self.startup_gripper_limit_step >= self.startup_gripper_limit_timesteps:
            self.startup_gripper_limit_active = False
        return base_action

    def inference(self):
        with torch.inference_mode():
            if self.with_planning:
                self.refresh_ts_from_env()
            # process previous ts
            self.collect_obs()
            obs_dict_np = get_seq_obs(self.obs_history, self.t, self.n_obs_steps, self.max_timesteps)
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

            idx = self._policy_step_offset()
            if idx == 0:
                action_dict = self.policy.predict_action(obs_dict)
                self.np_action_seq = (
                    action_dict["action"][0].detach().to("cpu").numpy()
                )  # [T, Da]
            elif self.np_action_seq is None:
                raise RuntimeError(
                    "Missing cached policy actions with non-zero policy offset — "
                    "call mark_bc_segment_start() before the first inference() after TAMP "
                    f"(t={self.t}, t_bc_start={self.t_bc_start}, query_cycle={self.query_cycle})"
                )

            horizon = len(self.np_action_seq)
            if idx >= horizon:
                raise RuntimeError(
                    f"Inference action index out of range: idx={idx}, horizon={horizon}, "
                    f"t={self.t}, t_bc_start={self.t_bc_start}, query_cycle={self.query_cycle}"
                )
            full_action = self.np_action_seq[idx]

            # Always keep the real env interface at 14D (joint + gripper).
            if full_action.shape[-1] > 14:
                base_action = full_action[:14].copy()
            else:
                base_action = full_action.copy()

            base_action = self._apply_freeze_blend(full_action, base_action)
            # base_action = self._apply_startup_gripper_limiter(base_action)

            # step env with 14D action
            self.ts = self.env.step(base_action)

            # cache last applied 14D action for future freeze
            self.last_action = base_action.copy()

            self.t += 1
        return False

    def append_image(self):
        cam_high_image = self.env.image_recorder.cam_high_image
        # Overlay freeze indicators on the saved video frame, if available.
        try:
            import cv2

            img = cam_high_image.copy()
            # Choose custom text if provided, otherwise use the latest overlay_text.

            if self.overlay_text is not None:
                cv2.putText(
                    img,
                    self.overlay_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            self.image_list.append({"cam_high": img})
        except Exception:
            # Fallback: save raw image if overlay fails for any reason.
            self.image_list.append({"cam_high": cam_high_image})

        if self.save_hdf5 and self.render_obs_keys and self.hdf5_buffers is not None:
            try:
                import cv2

                all_cam_images = self.env.image_recorder.get_images()
                for cam_name in self.render_obs_keys:
                    frame = all_cam_images.get(cam_name)
                    if frame is None:
                        continue
                    resized_frame = cv2.resize(
                        frame,
                        self.hdf5_image_size,
                        interpolation=cv2.INTER_AREA,
                    )
                    self.hdf5_buffers[cam_name].append(np.ascontiguousarray(resized_frame))
            except Exception as e:
                print(f"Warning: Failed to capture HDF5 frame: {e}")

    def exit(self, save_dir):
        # save_videos(self.image_list, DT, video_path=os.path.join(save_dir, f'rollout.mp4'))
        import cv2
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cur_time = time.strftime("%d_%H.%M.%S", time.localtime())
        vid_save_path = os.path.join(save_dir,  'test_' +cur_time +'.avi')

        height, width, _ = self.image_list[0]['cam_high'].shape
        fps = 30  # Adjust based on your camera settings
        self.video_writer = cv2.VideoWriter(
            vid_save_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            ( width,height)
        )
        for image in self.image_list:
            rgb_image = cv2.cvtColor(image['cam_high'], cv2.COLOR_BGR2RGB)
            # transpose image width and height
            self.video_writer.write(rgb_image)
        self.video_writer.release()
        print(f"Saved video to {vid_save_path}")

        if self.save_hdf5 and self.hdf5_buffers is not None and any(self.hdf5_buffers.values()):
            self.hdf5_path = os.path.join(save_dir, 'test_' + cur_time + '.hdf5')
            try:
                import h5py
                with h5py.File(self.hdf5_path, 'w') as hf:
                    hf.attrs['fps'] = self.fps
                    hf.attrs['max_timesteps'] = self.max_timesteps
                    num_recorded = 0
                    for cam_name, cam_frames in self.hdf5_buffers.items():
                        if not cam_frames:
                            continue
                        hf.create_dataset(
                            cam_name,
                            data=np.stack(cam_frames, axis=0),
                            compression='gzip',
                        )
                        num_recorded = max(num_recorded, len(cam_frames))
                    hf.attrs['num_frames'] = num_recorded
                print(f"HDF5 saved to: {self.hdf5_path}")
            except Exception as e:
                print(f"Warning: Failed to write HDF5 file: {e}")



def wrapper_test():

    output = './data/eval/transfer_cup/'
    checkpoint_dict = {\
        # 'handoff_cup': '/ssd1/chenyizhou/dp_ckpts/handoff_cup/long_chunk/epoch=1975-train_loss=0.0000.ckpt', \
        # 'clean_cup': '/ssd1/chenyizhou/dp_ckpts/clean_cup/long_chunk/fm_3view_place.ckpt',\
        'screwdriver_noisy': '/ssd1/chenyizhou/dp_ckpts/aloha_screwdriver_noisy/dp_3view/latest.ckpt',\
        # 'screwdriver_noisy': '/ssd1/chenyizhou/dp_ckpts/aloha_screwdriver_noisy/fm_3view/latest_headdown.ckpt',
        # 'hang_cup': '/ssd1/chenyizhou/dp_ckpts/aloha_hang_cup/latest_fm.ckpt',
        # 'two_arm_pour': '/ssd1/chenyizhou/dp_ckpts/two_arm_pour/original_long_horizon_new.ckpt',
        # 'two_arm_pour_freeze': '/ssd1/chenyizhou/dp_ckpts/two_arm_pour/freeze_long_horizon.ckpt'

                       }
    skill_names = list(checkpoint_dict.keys())

    max_timesteps = 1400
    num_inference_steps = 10
    scale = 4

    dp = DP_Evaluator(checkpoint_dict, output, max_timesteps, \
                      num_inference_steps, scale)

    for skill in skill_names:
        # dp.set_skill(skill)
        for i in range(max_timesteps):
            dp.inference()
            dp.append_image()

    dp.exit(output)


if __name__ == '__main__':
    # main()
    wrapper_test()
