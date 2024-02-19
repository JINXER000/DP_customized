# %%
import time
import numpy as np
import click
import cv2
import numpy as np
import torch
import dill
import hydra
import pathlib
import skvideo.io
from omegaconf import OmegaConf
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

import ipdb
ipdb.set_trace()


OmegaConf.register_new_resolver("eval", eval, replace=True)

## data configuration settings
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--max_timesteps', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=60, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--num_inference_steps', '-n', default=16, type=int, help="DDIM inference iterations.")
def main(input, output,
    vis_camera_idx, 
    max_timesteps, 
    max_duration,
    frequency):

    ### load checkpoint
    ckpt_path = input
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill) ## payload = {'cfg', 'state_dicts', 'pickles'}
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

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
        policy.num_inference_steps = num_inference_steps # [DDIM inference iterations]
        policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
    else:
        raise RuntimeError("Unsupported policy type: ", cfg.name)
    
    ### setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    n_obs_steps = cfg.n_obs_steps
    print("n_obs_steps: ", n_obs_steps)
    print("max_timesteps:", max_timesteps)
    print("action_offset:", action_offset) ## what is action offset?

    ipdb.set_trace()

    ## load aloha env

    from aloha.aloha_scripts.robot_utils import move_grippers
    from aloha.aloha_scripts.real_env import make_real_env

    env = make_real_env(init_node=True)
    env_max_reward = 0

    ## rollout
    max_timesteps = int(max_timesteps * 1) ## may increase for real-world tasks

    num_rollouts = 5
    episode_returns = []
    highest_rewards = []

    for rollout_idx in range(num_rollouts):
        rollout_idx += 1
        print(f"Rollout {rollout_idx}")
        
        ts = env.reset() # reset env

        qpos_history = torch.zero((1, max_timesteps, state_dim)).cuda()
        image_list = []
        qpos_list = []
        target_qpos_list = []
        rewards = []

        with torch.inference_mode():

            for t in range(max_timesteps):
            ## loop max_timesteps
                
                ## process previous timesteps to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                ipdb.set_trace()

                ## To-Do: 需要仔细debug一下，输入形式？输出形式？
                with torch.no_grad():
                    s = time.time()
                    # obs_dict_np = get_real_obs_dict(
                    #     env_obs=obs, shape_meta=cfg.task.shape_meta)
                    # obs_dict = dict_apply(obs_dict_np, 
                    #     lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    result = policy.predict_action(qpos, curr_image)
                    # this action starts from the first obs step
                    action = result['action'][0].detach().to('cpu').numpy()
                    print('Inference latency:', time.time() - s)

                ## env.step
                target_qpos = action ## Note: 由于 aloha_dataset 并没有对数据进行 normalize，所以此处可不用进行 pre-processing
                ts = env.step(target_qpos)

                # ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                # rewards.append(ts.reward)


        ## move grippers
        move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
        pass

        ## statistics
        save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))



def save_videos(video, dt, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        h, w, _ = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        fps = int(1/dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = image[:, :, [2, 1, 0]] # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'Saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        fps = int(1 / dt)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'Saved video to: {video_path}')


if __name__ == '__main__':
    main()


