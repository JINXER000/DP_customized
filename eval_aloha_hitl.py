"""
Usage:
python eval_aloha_hitl.py \
    -i data/outputs/2024.07.18/19.44.55_act_aloha_starbucks/checkpoints/latest.ckpt \
    -o data/eval/aloha_starbucks/ \
    -t 500
"""

import os
import pathlib
import time
import numpy as np
import click
import copy
import json
import numpy as np
import torch
import dill
import hydra
from omegaconf import OmegaConf
from einops import rearrange
from pynput import keyboard
from enum import Enum

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.video_recorder import save_videos
from diffusion_policy.common.hitl_recorder import HitlRecorder

from aloha.aloha_scripts.robot_utils import move_grippers, move_arms, torque_on, torque_off
from aloha.aloha_scripts.robot_utils import get_arm_gripper_positions, get_arm_joint_positions
from aloha.aloha_scripts.real_env import make_real_env, get_action
from aloha.aloha_scripts.constants import DT, PUPPET_GRIPPER_JOINT_NORMALIZE_FN, MASTER_GRIPPER_JOINT_UNNORMALIZE_FN

from interbotix_xs_modules.arm import InterbotixManipulatorXS


OmegaConf.register_new_resolver("eval", eval, replace=True)

## data configuration settings
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--max_timesteps', '-t', default=500, help='Max duration for each epoch in seconds.')
@click.option('--num_inference_steps', '-n', default=10, type=int, help="DDIM inference iterations.")
@click.option('--scale', '-s', default=4, type=int, help="Image downsample scale")
@click.option('--round', '-r', default=1, type=int, help="Collect HITL data for i-th round training.")
def main(
    input, 
    output,
    max_timesteps,
    num_inference_steps,
    scale,
    round
):
    # make output directory
    output_root = output
    output += time.strftime("%Y.%m.%d/%H.%M.%S", time.localtime())
    if os.path.exists(output):
        click.confirm(f"Output path {output} already exists! Overwrite?", abort=True)
    pathlib.Path(output).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(input, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output)
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
    if 'diffusion' in cfg.name:
        ## diffusion model
        policy: BaseImagePolicy
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device('cuda')
        policy.eval().to(device)

        ## set inference params
        policy.num_inference_steps = num_inference_steps #16 # [DDIM inference iterations]
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
    
    # hyper-parameters
    ## observation
    state_dim = cfg.task.shape_meta.obs.qpos.shape[0] ## qpos shape
    camera_names = cfg.task.dataset.camera_names
    obs_shape_meta = cfg.task.shape_meta.obs
    c, h, w = obs_shape_meta.cam_high.shape ## [c, h, w]

    ## multi-step params for policy
    query_cycle = cfg.n_action_steps
    n_obs_steps = cfg.n_obs_steps

    # setup experiment
    env = make_real_env(init_node=True, downsample_scale=scale)
    master_bot_left = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                              robot_name=f'master_left', init_node=False)
    master_bot_right = InterbotixManipulatorXS(robot_model="wx250s", group_name="arm", gripper_name="gripper",
                                               robot_name=f'master_right', init_node=False)

    ## prepare recorder
    prefix = cfg.task.dataset.weight_type + '_' if round > 1 else ''
    output_round_r = output_root + f'/{prefix}round_{round}/'
    pathlib.Path(output_round_r).mkdir(parents=True, exist_ok=True)  # data for r-th round training
    hitl_recorder = HitlRecorder(output_round_r, camera_names, max_timesteps)

    ## reset env
    initialize_master_arms(master_bot_left, master_bot_right)    
    ts = env.reset()
    inference_time_list = []
    image_list = []
    ep_t0 = time.perf_counter()
    print(f"Rollout begins!")

    ## keyboard listener for changing system state and human intervention
    global mode, prev_mode
    mode = prev_mode = Mode.AUTO

    def on_press(key):
        # defines state transitions for mode.
        global mode, prev_mode
        if key == keyboard.KeyCode.from_char('p') or key == keyboard.KeyCode.from_char('P'):
            if mode != Mode.PAUSE:
                prev_mode = mode
                mode = Mode.PAUSE
                print("PAUSE. Press 'A' to continue or 'H' to take over.")
        elif key == keyboard.KeyCode.from_char('a') or key == keyboard.KeyCode.from_char('A'):
            if mode == Mode.PAUSE:
                prev_mode = mode
                mode = Mode.AUTO
                print("AUTO. Neural net policy continues.")
        elif key == keyboard.KeyCode.from_char('h') or key == keyboard.KeyCode.from_char('H'):
            if mode == Mode.PAUSE:
                prev_mode = mode
                mode = Mode.HUMAN
                print("HUMAN. Human takes over control.")
    
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    ## obs history for extracting multi-step obs
    obs_history = dict()
    for key in obs_shape_meta.keys():
        obs_history[key] = np.zeros(
            (max_timesteps, *obs_shape_meta[key].shape),
            dtype=np.float32
        )

    with torch.inference_mode():
        last_pred_step = 0
        for t in range(max_timesteps):
            # process previous ts
            obs = ts.observation
            image_list.append(obs["original_images"])  ## record images
            collect_obs(obs_shape_meta, obs_history, t, obs)
            obs_dict_np = get_seq_obs(obs_history, t, n_obs_steps)
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

            # define robot behaviours upon transition
            if mode == Mode.PAUSE:
                if prev_mode == Mode.AUTO:
                    sync_master_with_puppet(master_bot_left, master_bot_right, env.puppet_bot_left, env.puppet_bot_right)
                    print("Synced master with puppet. Ready for intervention.")
                elif prev_mode == Mode.HUMAN:
                    torque_on(master_bot_left)
                    torque_on(master_bot_right)
            elif mode == Mode.HUMAN:
                if prev_mode == Mode.PAUSE:
                    torque_off(master_bot_left)
                    torque_off(master_bot_right)
            elif mode == Mode.AUTO:
                pass
            prev_mode = mode

            # Wait when paused
            while mode == Mode.PAUSE:
                time.sleep(DT / 5)
            
            if mode == Mode.AUTO and prev_mode == Mode.PAUSE:
                last_pred_step = t  # need predict action when PAUSE -> AUTO

            # Get action according to mode
            t0 = time.perf_counter()
            ## AUTO: query policy to extract action: (B=1, Da)
            if mode == Mode.AUTO:
                t_sub = t - last_pred_step  # starts from AUTO segments
                if t_sub % query_cycle == 0:
                    action_dict = policy.predict_action(obs_dict)
                    np_action_seq = action_dict['action'][0].detach().to('cpu').numpy() # T,Da
                action = np_action_seq[t_sub % query_cycle]
            ## HUMAN: take qpos of masters as action
            elif mode == Mode.HUMAN:
                action = get_action(master_bot_left, master_bot_right)

            t1 = time.perf_counter()

            # store data
            hitl_recorder.store_one_step(ts, action, mode.value)

            # step env
            ts = env.step(action)

            # collect statistics
            inference_time_list.append(t1 - t0)

            postfix = ""
            if mode == Mode.AUTO:
                postfix = f"AUTO, sub-step {t_sub % query_cycle} / {query_cycle}"
            elif mode == Mode.HUMAN:
                postfix = f"HUMAN"
            print(f"Step {t:>3d}, {t1 - t0:.4f} [s], {postfix}")

    listener.stop()

    # move grippers to open
    PUPPET_GRIPPER_JOINT_OPEN = 0.3
    move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open

    # collect stats
    ep_t1 = time.perf_counter()
    print(f'Avg inference time: {np.mean(inference_time_list[1:]):.4f} +/- {np.std(inference_time_list[1:]):.4f} s')
    print(f'Episode time: {ep_t1 - ep_t0:.4f} s')

    json_log = dict()
    json_log["checkpoint"] = input
    output_path = os.path.join(output, 'eval_log.json')
    json.dump(json_log, open(output_path, 'w'), indent=2, sort_keys=True)

    # save rollout videos
    save_videos(image_list, DT, video_path=os.path.join(output, f'rollout.mp4'))

    # save HITL rollout traj for further training
    hitl_recorder.store_episode()


# ----------------- Helper functions and classes -----------------
class Mode(Enum):
    AUTO = 1
    PAUSE = 2
    HUMAN = 3


def initialize_master_arms(master_bot_left, master_bot_right):
    master_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    master_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    master_bot_left.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_left.dxl.robot_set_operating_modes("single", "gripper", "position")
    master_bot_right.dxl.robot_set_operating_modes("group", "arm", "position")
    master_bot_right.dxl.robot_set_operating_modes("single", "gripper", "position")
    torque_on(master_bot_left)
    torque_on(master_bot_right)


def sync_master_with_puppet(master_bot_left, master_bot_right, puppet_bot_left, puppet_bot_right):
    # Without rebooting gripper motor, masters will not follow the commands
    # The opening or closing of gripper is bang-bang control even qpos is set to a value
    # Therefore, we recommend to intervene only when gripper is totally open or closed
    master_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    master_bot_right.dxl.robot_reboot_motors("single", "gripper", True)

    # get joint states of puppets
    puppet_left_arm_qpos = get_arm_joint_positions(puppet_bot_left)
    puppet_right_arm_qpos = get_arm_joint_positions(puppet_bot_right)
    puppet_left_gripper_qpos = get_arm_gripper_positions(puppet_bot_left)
    puppet_right_gripper_qpos = get_arm_gripper_positions(puppet_bot_right)

    PUPPET2MASTER_JOINT_FN = lambda x: MASTER_GRIPPER_JOINT_UNNORMALIZE_FN(PUPPET_GRIPPER_JOINT_NORMALIZE_FN(x))

    # then move master to the same joint states
    move_arms(
        [master_bot_left, master_bot_right],
        [puppet_left_arm_qpos, puppet_right_arm_qpos],
        move_time=1.5
    )
    move_grippers(
        [master_bot_left, master_bot_right],
        [PUPPET2MASTER_JOINT_FN(puppet_left_gripper_qpos), PUPPET2MASTER_JOINT_FN(puppet_right_gripper_qpos)],
        move_time=1.0
    )


def collect_obs(obs_shape_meta, obs_history, t, obs):
    for k, v in obs_shape_meta.items():
        if v["type"] == "rgb":
            obs_history[k][t] = np.moveaxis(
                obs["images"][k].astype(np.float32) / 255.0, -1, 0
            )
        else:
            obs_history[k][t] = obs[k].astype(np.float32)
    return


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


if __name__ == '__main__':
    main()
