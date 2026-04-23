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
import numpy as np
import copy
import json
import numpy as np
import torch
import dill
import hydra


from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

from aloha_pkg.aloha_scripts.real_env import make_real_env

from aloha_pkg.aloha_scripts.constants import DT
from diffusion_policy.real_world.video_recorder import save_videos




def get_seq_obs(obs_history, t, n_obs_steps):
    obs_dict_np = dict()
    if t < n_obs_steps - 1:
        for k, v in obs_history.items():
            obs_dict_np[k] = np.array([v[0]] * n_obs_steps, dtype=np.float32)
            obs_dict_np[k][n_obs_steps-t-1:n_obs_steps] = v[0:t+1]
    else:
        for k, v in obs_history.items():
            obs_dict_np[k] = v[t-n_obs_steps+1:t+1]
    return obs_dict_np

class DP_Evaluator():
    def __init__(self, checkpoint_dict, output, max_timesteps, num_inference_steps, scale, with_planning= False):
        self.max_timesteps = max_timesteps
        self.checkpoint_dict = checkpoint_dict
        self.output = output
        self.num_inference_steps = num_inference_steps

        # skill_names = list(checkpoint_dict.keys())
        # self.cur_skill = skill_names[0]
        self.with_planning = with_planning  
        self.image_list = []
        self.camera_names = self._resolve_camera_names()
        if 'cam_high' not in self.camera_names:
            raise ValueError("DP_Evaluator requires 'cam_high' in camera_names for video recording.")
        # setup experiment
        self.env = make_real_env(
            init_node=True,
            downsample_scale=scale,
            setup_robots=not self.with_planning,
            camera_names=self.camera_names,
        )

        # Freeze-control related state (optional, per-skill).
        self.allow_freeze = False
        self.last_action = None
        # Text overlay for video frames (e.g., freeze indicators).
        self.overlay_text = None

        ## set a default skill to get obs_shape_meta
        skill_names = list(checkpoint_dict.keys())
        self.set_skill(skill_names[0])

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


    def set_skill(self, skill_name, reset_grippers= True):
        self.cur_skill = skill_name
        self.load_checkpoint()        
        self.reset_all(reset_grippers = reset_grippers)

    def load_checkpoint(self):
        # load checkpoint
        payload = torch.load(open(self.checkpoint_dict[self.cur_skill], 'rb'), pickle_module=dill)
        cfg = payload['cfg']

        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=self.output)
        workspace: BaseWorkspace
        # in case that model, ema_model & opt are not defined in __init__ (e.g. ddp)
        if "model" not in workspace.__dict__.keys():
            workspace.model = hydra.utils.instantiate(cfg.policy)
        if "ema_model" not in workspace.__dict__.keys() and cfg.training.use_ema:
            workspace.ema_model = copy.deepcopy(workspace.model)
        if "optimizer" not in workspace.__dict__.keys():
            workspace.optimizer = hydra.utils.instantiate(
                cfg.optimizer, workspace.model.parameters()
            )
        workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)

        # get policy from workspace
        # if 'diffusion' in cfg.name:
        ## diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        self.device = torch.device('cuda')
        policy.eval().to(self.device)

        ## set inference params
        policy.num_inference_steps = self.num_inference_steps #16 # [DDIM inference iterations]
        # else:
        #     raise RuntimeError("Unsupported policy type: ", cfg.name)
        
        self.policy= policy
        self.obs_shape_meta = cfg.task.shape_meta.obs

        ## multi-step params for policy
        self.query_cycle = cfg.n_action_steps
        self.n_obs_steps = cfg.n_obs_steps

        # Whether this skill uses extra freeze channels in the action (14 + 2).
        self.allow_freeze = False
        # task_cfg = getattr(cfg, "task", None)
        # if task_cfg is not None:
        #     dataset_cfg = getattr(task_cfg, "dataset", None)
        #     # OmegaConf containers support key access like a dict.
        #     if dataset_cfg is not None and "allow_freeze" in dataset_cfg:
        #         self.allow_freeze = bool(dataset_cfg["allow_freeze"])

    def reset_all(self, reset_grippers = True):
        self.ts = self.env.reset(fake=self.with_planning)
        # inference_time_list = []
        # ep_t0 = time.perf_counter()
        print(f"Reset DP env!")
        if reset_grippers:
            self.env.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
            self.env.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
            
            ## obs history for extracting multi-step obs
            self.obs_history = dict()
            for key in self.obs_shape_meta.keys():
                self.obs_history[key] = np.zeros(
                    (self.max_timesteps, *self.obs_shape_meta[key].shape),
                    dtype=np.float32
                )
        self.t = 0
        # Reset last applied action used for freeze control.
        self.last_action = None

    def collect_obs(self):
        if self.t >= self.max_timesteps:
            print('Reached max timesteps in collect_obs')
            return
        ## NOTE: we need to load a dp snapshort first to get obs_shape_meta
        for k, v in self.obs_shape_meta.items():
            if v["type"] == "rgb":
                self.obs_history[k][self.t] = np.moveaxis(
                    self.ts.observation["images"][k].astype(np.float32) / 255.0, -1, 0
                )
            else:
                self.obs_history[k][self.t] = self.ts.observation[k].astype(np.float32)
        return

    def inference(self):
        if self.t >= self.max_timesteps:
            return True
        with torch.inference_mode():
            # process previous ts
            self.collect_obs()
            obs_dict_np = get_seq_obs(self.obs_history, self.t, self.n_obs_steps)
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))

            # query policy to extract action sequence: (B=1, T, Da)
            # t0 = time.perf_counter()
            if self.t % self.query_cycle == 0:
                action_dict = self.policy.predict_action(obs_dict)
                self.np_action_seq = (
                    action_dict["action"][0].detach().to("cpu").numpy()
                )  # [T, Da]

            full_action = self.np_action_seq[self.t % self.query_cycle]

            # Always keep the real env interface at 14D (joint + gripper).
            if full_action.shape[-1] > 14:
                base_action = full_action[:14].copy()
            else:
                base_action = full_action.copy()

            # Optionally apply per-arm freeze using the last 2 dims (left/right).
            if self.allow_freeze and full_action.shape[-1] >= 16:
                # Treat the last two dimensions as continuous freeze strengths alpha in [0, 1]
                # and blend between the current DP command and the last held command:
                # u = (1 - alpha) * u_DP + alpha * u_hold
                freeze = np.clip(full_action[14:16], 0.0, 1.0)
                left_alpha = float(freeze[0])
                right_alpha = float(freeze[1])
                # Cache overlay text for visualization/logging.
                self.overlay_text = f"L:{left_alpha:.2f} R:{right_alpha:.2f} allow_freeze:{self.allow_freeze}"

                left_slice = slice(0, 7)
                right_slice = slice(7, 14)

                if self.last_action is not None:
                    # Left arm blend
                    if left_alpha > 0.0:
                        base_action[left_slice] = (
                            (1.0 - left_alpha) * base_action[left_slice]
                            + left_alpha * self.last_action[left_slice]
                        )
                    # Right arm blend
                    if right_alpha > 0.0:
                        base_action[right_slice] = (
                            (1.0 - right_alpha) * base_action[right_slice]
                            + right_alpha * self.last_action[right_slice]
                        )

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



def wrapper_test():

    output = './data/eval/transfer_cup/'
    checkpoint_dict = {\
        # 'handoff_cup': '/ssd1/chenyizhou/dp_ckpts/handoff_cup/long_chunk/epoch=1975-train_loss=0.0000.ckpt', \
        # 'clean_cup': '/ssd1/chenyizhou/dp_ckpts/clean_cup/long_chunk/epoch=1900-train_loss=0.0001.ckpt',\
        'screwdriver_noisy': '/ssd1/chenyizhou/dp_ckpts/aloha_screwdriver_noisy/fm_3view/latest.ckpt',\
        # 'screwdriver_noisy': '/ssd1/chenyizhou/dp_ckpts/aloha_screwdriver_noisy/long_chunk/epoch=1925-train_loss=0.0001.ckpt',
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
