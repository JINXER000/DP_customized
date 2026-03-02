import os
import time
import numpy as np
import click
import cv2
import copy
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
from einops import rearrange
import scipy.spatial.transform as st
# from diffusion_policy.real_world.real_env import RealEnv
# from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.real_world.real_inference_util import (
    get_real_obs_resolution, 
    get_real_obs_dict)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.cv2_util import get_image_transform

from aloha_pkg.aloha_scripts.robot_utils import move_grippers
from aloha_pkg.aloha_scripts.real_env import make_real_env

import ipdb

OmegaConf.register_new_resolver("eval", eval, replace=True)

## data configuration settings
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
# @click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
# @click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_timesteps', '-md', default=500, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=50, type=float, help="Control frequency in Hz.")
@click.option('--num_inference_steps', '-n', default=16, type=int, help="DDIM inference iterations.")
@click.option('--scale', '-s', default=4, type=int, help="Image downsample scale")
def main(input, 
         output,
    # vis_camera_idx, 
    # steps_per_inference, 
    max_timesteps,
    frequency,
    num_inference_steps,
    scale):

    ### load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill) ## payload = {'cfg', 'state_dicts', 'pickles'}
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    ### in case that model, ema_model & opt are not defined in __init__ (e.g. ddp)
    if "model" not in workspace.__dict__.keys():
        workspace.model = hydra.utils.instantiate(cfg.policy)
    if "ema_model" not in workspace.__dict__.keys() and cfg.training.use_ema:
        workspace.ema_model = copy.deepcopy(workspace.model)
    if "optimizer" not in workspace.__dict__.keys():
        workspace.optimizer = hydra.utils.instantiate(
            cfg.optimizer, workspace.model.parameters()
        )
    workspace.load_payload(payload, exclude_keys=["optimizer"], include_keys=None)

    ### load policy
    action_offset = 0
    delta_action = False
    if 'diffusion' in cfg.name:
        # diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = num_inference_steps #16 # [DDIM inference iterations]
        #policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    elif 'train_consistency_unet_image' in cfg.name:
        # consistency model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        # set inference params
        policy.num_inference_steps = num_inference_steps #
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)
    
    ### hyper-parameters
    state_dim = cfg.task.shape_meta.obs.qpos.shape[0] ## qpos shape
    camera_names = cfg.task.dataset.camera_names
    shape_meta = cfg.task.shape_meta
    c, h, w = shape_meta.obs.cam_high.shape ## [c, h, w]

    ### setup experiment
    dt = 1/frequency ## Warning: ALOHA constants.py 中的 DT = 0.02, so set frequency = 50

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)  # why (w,h) here?
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("max_timesteps:", max_timesteps)
    print("action_offset:", action_offset) ## what is action offset?

    ## load aloha env
    env = make_real_env(init_node=True, downsample_scale=scale)
    env_max_reward = 0

    ## rollout
    max_timesteps = int(max_timesteps * 1) ## may increase for real-world tasks

    num_rollouts = 1
    episode_returns = []
    highest_rewards = []
    video_recorder = VideoRecorder()  # Initialize video recorder

    for rollout_idx in range(num_rollouts):
        rollout_idx += 1

        print(f"Rollout {rollout_idx} - Collecting observations...")

        ## reset env
        ts = env.reset()
        t_idx = n_obs_steps

        qpos_history = np.zeros(
            (max_timesteps+n_obs_steps, state_dim), dtype=np.float32
        ) ## [max_timesteps, state_dim]
        images_history = dict()
        for cam_name in camera_names:
            images_history[cam_name] = np.zeros(
                (max_timesteps+n_obs_steps, c, h, w), dtype=np.float32
            ) ## [max_timesteps, c, h, w]

        qpos_history, images_history = collect_obs(ts, t_idx, n_obs_steps, qpos_history, images_history, camera_names, shape_meta)
        ep_t0 = time.perf_counter()
        step_time_list = []
        with torch.inference_mode():
            ## loop max_timesteps
            while True:
                ''' construct observations_seq = {"images", "qpos"} '''
                obs_dict_np = dict()
                obs_dict_np["qpos"] = qpos_history[t_idx-n_obs_steps:t_idx] ## [n_obs_steps, state_dim]")
                for cam_name in camera_names:
                    obs_dict_np[cam_name] = images_history[cam_name][t_idx-n_obs_steps:t_idx] ## [n_obs_steps, c, h, w]

                # print(f"observation_range = {t_idx-n_obs_steps}:{t_idx}")

                ''' get action sequence '''
                t0 = time.perf_counter()
                obs_dict = dict_apply(obs_dict_np,
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                t1 = time.perf_counter()
                # print(f"Execution Policy: {t1 - t0:.4f} [s]")
                step_time_list.append(t1 - t0)
                # ipdb.set_trace()

                action_seq = result['action'][0].detach().to('cpu').numpy()

                ''' implement action sequence '''
                for action in action_seq:
                    ts = env.step(action)
                    video_recorder.append_image(env)  # Collect image for video

                    qpos_history, images_history = collect_obs(ts, t_idx, n_obs_steps, qpos_history, images_history, camera_names, shape_meta)

                    t_idx += 1
                    if t_idx == max_timesteps+n_obs_steps:
                        # ipdb.set_trace()
                        break

                if t_idx >= max_timesteps+n_obs_steps:
                    break

    ### move grippers
    PUPPET_GRIPPER_JOINT_OPEN = 1.4910
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
    ep_t1 = time.perf_counter()
    print(f"Average step time: {np.mean(step_time_list[1:]):.4f} +/- {np.std(step_time_list[1:]):.4f} s")
    print(f"Total time: {ep_t1 - ep_t0:.4f} s")

    # Save video
    video_recorder.save_video(output)
                        

def collect_obs(ts, idx, n_obs_steps, qpos_history, images_history, camera_names, shape_meta):
    obs = ts.observation
    ## get qpos input
    qpos = np.array(obs['qpos']) ## [state_dim,]
    if idx == n_obs_steps:
        qpos_history[idx-n_obs_steps:idx] = qpos # broadcast here
    else:
        qpos_history[idx] = qpos

    ## get image input
    curr_image_dict = get_image(ts, camera_names, shape_meta) # it returns a dict
    if idx == n_obs_steps:
        for cam_name in camera_names:
            images_history[cam_name][idx-n_obs_steps:idx] = curr_image_dict[cam_name]
    else:
        for cam_name in camera_names:
            images_history[cam_name][idx] = curr_image_dict[cam_name]

    return qpos_history, images_history



def get_image(ts, camera_names, shape_meta):
    """
    @return
        curr_images_dict: {
            cam_name: image (c,h,w)
        }
    """
    curr_images_dict = dict()
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w') / 255.0
        assert curr_image.shape == tuple((shape_meta.obs[cam_name]).shape), \
            f"{curr_image.shape} vs. {tuple((shape_meta.obs[cam_name]).shape)}"
        curr_images_dict[cam_name] = curr_image # [0, 1]^(c,h,w)

    return curr_images_dict






class VideoRecorder:
    def __init__(self):
        self.image_list = []
        self.video_writer = None

    def append_image(self, env):
        cam_high_image = env.image_recorder.cam_high_image
        self.image_list.append({'cam_high': cam_high_image})

    def save_video(self, save_dir):
        if not self.image_list:
            print("No images to save")
            return

        import cv2
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        cur_time = time.strftime("%d_%H.%M.%S", time.localtime())
        vid_save_path = os.path.join(save_dir, 'test_' + cur_time + '.avi')

        height, width, _ = self.image_list[0]['cam_high'].shape
        fps = 30  # Adjust based on your camera settings
        self.video_writer = cv2.VideoWriter(
            vid_save_path,
            cv2.VideoWriter_fourcc(*'XVID'),
            fps,
            (width, height)
        )
        for image in self.image_list:
            rgb_image = cv2.cvtColor(image['cam_high'], cv2.COLOR_BGR2RGB)
            # transpose image width and height
            self.video_writer.write(rgb_image)
        self.video_writer.release()
        print(f"Saved video to {vid_save_path}")


if __name__ == '__main__':
    main()


